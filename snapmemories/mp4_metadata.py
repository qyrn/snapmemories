import os
import struct
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

from snapmemories.models import GeoPoint

MP4_EPOCH_OFFSET_SECONDS = 2_082_844_800
CONTAINER_BOXES = frozenset({b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"udta"})
TIMED_BOXES = frozenset({b"mvhd", b"tkhd", b"mdhd"})
LOCATION_BOX = b"\xa9xyz"
UNDETERMINED_LANGUAGE = 0x15C7
MAX_UINT32 = 0xFFFFFFFF
COPY_BUFFER_BYTES = 1024 * 1024


class Mp4FormatError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Box:
    kind: bytes
    start: int
    header_size: int
    size: int

    @property
    def end(self) -> int:
        return self.start + self.size

    @property
    def payload_start(self) -> int:
        return self.start + self.header_size


def write_capture_metadata(path: Path, taken_at: datetime, location: GeoPoint | None) -> None:
    with path.open("r+b") as stream:
        top_level = list(_read_top_level_boxes(stream))
        moov = next((box for box in top_level if box.kind == b"moov"), None)
        if moov is None:
            raise Mp4FormatError("No moov box found.")
        stream.seek(moov.start)
        original = stream.read(moov.size)

    updated = bytearray(original)
    mp4_seconds = int(taken_at.timestamp()) + MP4_EPOCH_OFFSET_SECONDS
    _visit(updated, 0, len(updated), lambda box: _set_creation_time(updated, box, mp4_seconds))

    new_moov = bytes(updated)
    if location is not None:
        new_moov = _with_location(new_moov, moov.start, location) or new_moov

    _store_moov(path, moov, top_level[-1].end, new_moov)


def _read_top_level_boxes(stream: BinaryIO) -> Iterator[Box]:
    stream.seek(0, os.SEEK_END)
    file_size = stream.tell()
    position = 0
    while position + 8 <= file_size:
        stream.seek(position)
        box = _parse_header(stream.read(16), position, file_size)
        yield box
        position = box.end


def _parse_header(header: bytes, start: int, limit: int) -> Box:
    size, kind = struct.unpack(">I4s", header[:8])
    header_size = 8
    if size == 1:
        size = struct.unpack(">Q", header[8:16])[0]
        header_size = 16
    elif size == 0:
        size = limit - start
    if size < header_size or start + size > limit:
        raise Mp4FormatError(f"Invalid box size for {kind!r}.")
    return Box(kind=kind, start=start, header_size=header_size, size=size)


def _children(data: bytes | bytearray, start: int, end: int) -> Iterator[Box]:
    position = start
    while position + 8 <= end:
        box = _parse_header(bytes(data[position : position + 16]), position, end)
        yield box
        position = box.end


def _visit(data: bytearray, start: int, end: int, visitor: Callable[[Box], None]) -> None:
    for box in _children(data, start, end):
        visitor(box)
        if box.kind in CONTAINER_BOXES:
            _visit(data, box.payload_start, box.end, visitor)


def _set_creation_time(data: bytearray, box: Box, mp4_seconds: int) -> None:
    if box.kind not in TIMED_BOXES:
        return
    version = data[box.payload_start]
    if version == 1:
        struct.pack_into(">QQ", data, box.payload_start + 4, mp4_seconds, mp4_seconds)
    elif mp4_seconds <= MAX_UINT32:
        struct.pack_into(">II", data, box.payload_start + 4, mp4_seconds, mp4_seconds)


def _with_location(moov: bytes, moov_offset: int, location: GeoPoint) -> bytes | None:
    root = _parse_header(moov[:16], 0, len(moov))
    location_box = _location_box(location)
    children = list(_children(moov, root.payload_start, root.end))
    user_data = next((box for box in children if box.kind == b"udta"), None)

    if user_data is None:
        new_children = moov[root.payload_start :] + _box(b"udta", location_box)
    else:
        kept = b"".join(
            moov[child.start : child.end]
            for child in _children(moov, user_data.payload_start, user_data.end)
            if child.kind != LOCATION_BOX
        )
        new_children = (
            moov[root.payload_start : user_data.start]
            + _box(b"udta", kept + location_box)
            + moov[user_data.end :]
        )

    rebuilt = bytearray(_box(b"moov", new_children))
    growth = len(rebuilt) - len(moov)
    if not _shift_chunk_offsets(rebuilt, moov_offset, growth):
        return None
    return bytes(rebuilt)


def _location_box(location: GeoPoint) -> bytes:
    text = f"{location.latitude:+08.4f}{location.longitude:+09.4f}/".encode("ascii")
    return _box(LOCATION_BOX, struct.pack(">HH", len(text), UNDETERMINED_LANGUAGE) + text)


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", len(payload) + 8, kind) + payload


def _shift_chunk_offsets(moov: bytearray, moov_offset: int, growth: int) -> bool:
    tables: list[Box] = []

    def collect_tables(box: Box) -> None:
        if box.kind in (b"stco", b"co64"):
            tables.append(box)

    _visit(moov, 0, len(moov), collect_tables)
    for table in tables:
        count = struct.unpack_from(">I", moov, table.payload_start + 4)[0]
        wide = table.kind == b"co64"
        entry_format = ">Q" if wide else ">I"
        entry_size = 8 if wide else 4
        first_entry = table.payload_start + 8
        for index in range(count):
            position = first_entry + index * entry_size
            offset = struct.unpack_from(entry_format, moov, position)[0]
            if offset < moov_offset:
                continue
            shifted = offset + growth
            if not wide and shifted > MAX_UINT32:
                return False
            struct.pack_into(entry_format, moov, position, shifted)
    return True


def _store_moov(path: Path, moov: Box, file_end: int, new_moov: bytes) -> None:
    same_size = len(new_moov) == moov.size
    if same_size or moov.end == file_end:
        with path.open("r+b") as stream:
            stream.seek(moov.start)
            stream.write(new_moov)
            if not same_size:
                stream.truncate()
        return

    rewritten = path.with_name(path.name + ".rewrite")
    with path.open("rb") as source, rewritten.open("wb") as target:
        _copy_range(source, target, 0, moov.start)
        target.write(new_moov)
        _copy_range(source, target, moov.end, file_end)
    rewritten.replace(path)


def _copy_range(source: BinaryIO, target: BinaryIO, start: int, end: int) -> None:
    source.seek(start)
    remaining = end - start
    while remaining > 0:
        chunk = source.read(min(COPY_BUFFER_BYTES, remaining))
        if not chunk:
            break
        target.write(chunk)
        remaining -= len(chunk)
