import contextlib
import io
import os
import shutil
import threading
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import IO

import requests
from PIL import Image

from snapmemories.downloader import DownloadError, download_memory
from snapmemories.export_archive import list_embedded_memories
from snapmemories.library import Library, LibraryRecord
from snapmemories.media_format import MediaFormat, sniff_format
from snapmemories.models import (
    ArchiveMember,
    EmbeddedMemory,
    GeoPoint,
    ImportItem,
    MediaKind,
    RemoteMemory,
)
from snapmemories.mp4_metadata import Mp4FormatError, write_capture_metadata
from snapmemories.photo import render_photo

COPY_BUFFER_BYTES = 1024 * 1024
PHOTO_EXTENSIONS = {
    MediaFormat.JPEG: "jpg",
    MediaFormat.PNG: "png",
    MediaFormat.WEBP: "webp",
    MediaFormat.HEIC: "heic",
}
IMAGE_ERRORS = (OSError, ValueError, SyntaxError, Image.DecompressionBombError)


@dataclass(frozen=True, slots=True)
class _Media:
    main_bytes: bytes | None
    main_file: Path | None
    video_extension: str
    overlay: bytes | None


class MemoryImporter:
    def __init__(
        self, library: Library, work_directory: Path, on_bytes: Callable[[int], None]
    ) -> None:
        self._library = library
        self._work_directory = work_directory
        self._on_bytes = on_bytes
        self._archives = threading.local()
        self._opened: list[zipfile.ZipFile] = []
        self._opened_lock = threading.Lock()

    def close(self) -> None:
        with self._opened_lock:
            for archive in self._opened:
                archive.close()
            self._opened.clear()

    def import_item(self, item: ImportItem, session: requests.Session | None) -> LibraryRecord:
        local_time = item.taken_at.astimezone()
        scratch = self._work_directory / f"{threading.get_ident()}-{os.urandom(4).hex()}"
        try:
            media = self._fetch(item, scratch, session)
            if item.kind == MediaKind.VIDEO:
                target = self._save_video(media, local_time, item.location)
            else:
                target = self._save_photo(media, local_time, item.location)
        finally:
            scratch.unlink(missing_ok=True)

        timestamp = item.taken_at.timestamp()
        os.utime(target, (timestamp, timestamp))
        record = LibraryRecord(
            item_id=item.item_id,
            relative_path=self._library.relative(target),
            kind=item.kind,
            taken_at=local_time.isoformat(),
            latitude=item.location.latitude if item.location else None,
            longitude=item.location.longitude if item.location else None,
        )
        self._library.add(record)
        return record

    def _fetch(self, item: ImportItem, scratch: Path, session: requests.Session | None) -> _Media:
        if isinstance(item.source, EmbeddedMemory):
            return self._read_embedded(item.source, item.kind, scratch)
        if session is None:
            raise DownloadError("Network session unavailable")
        return self._download(item.source, item.kind, scratch, session)

    def _read_embedded(self, memory: EmbeddedMemory, kind: MediaKind, scratch: Path) -> _Media:
        overlay = self._read_member(memory.overlay) if memory.overlay else None
        if kind == MediaKind.PHOTO:
            main = self._read_member(memory.main)
            self._on_bytes(len(main))
            return _Media(main, None, "", overlay)
        archive = self._archive(memory.main.archive)
        with archive.open(memory.main.name) as source, scratch.open("wb") as target:
            self._copy(source, target)
        extension = PurePosixPath(memory.main.name).suffix.lower().lstrip(".") or "mp4"
        return _Media(None, scratch, extension, overlay)

    def _download(
        self, memory: RemoteMemory, kind: MediaKind, scratch: Path, session: requests.Session
    ) -> _Media:
        download_memory(memory, scratch, session, self._on_bytes)
        with scratch.open("rb") as stream:
            media_format = sniff_format(stream.read(16))
        if media_format == MediaFormat.ZIP:
            return self._unpack_downloaded_bundle(scratch, kind)
        if media_format == MediaFormat.UNKNOWN:
            raise DownloadError("Snapchat sent a file that is not a photo or a video")
        if kind == MediaKind.VIDEO or media_format == MediaFormat.MP4:
            return _Media(None, scratch, "mp4", None)
        return _Media(scratch.read_bytes(), None, "", None)

    def _unpack_downloaded_bundle(self, bundle: Path, kind: MediaKind) -> _Media:
        with zipfile.ZipFile(bundle) as archive:
            memories = list_embedded_memories(bundle, archive.infolist())
            if not memories:
                raise DownloadError("Snapchat sent an empty bundle")
            memory = memories[0]
            overlay = archive.read(memory.overlay.name) if memory.overlay else None
            if kind == MediaKind.PHOTO and memory.kind == MediaKind.PHOTO:
                return _Media(archive.read(memory.main.name), None, "", overlay)
            extracted = bundle.with_name(bundle.name + ".video")
            with archive.open(memory.main.name) as source, extracted.open("wb") as target:
                shutil.copyfileobj(source, target, COPY_BUFFER_BYTES)
        extracted.replace(bundle)
        extension = PurePosixPath(memory.main.name).suffix.lower().lstrip(".") or "mp4"
        return _Media(None, bundle, extension, overlay)

    def _save_photo(self, media: _Media, taken_at: datetime, location: GeoPoint | None) -> Path:
        main = media.main_bytes
        if main is None:
            raise DownloadError("Photo content missing")
        content = _render_with_fallback(main, media.overlay, taken_at, location)
        extension = PHOTO_EXTENSIONS.get(sniff_format(content[:16]), "jpg")
        return self._write_bytes(content, taken_at, extension)

    def _save_video(self, media: _Media, taken_at: datetime, location: GeoPoint | None) -> Path:
        if media.main_file is None:
            raise DownloadError("Video content missing")
        with contextlib.suppress(Mp4FormatError, OSError):
            write_capture_metadata(media.main_file, taken_at, location)
        target = self._library.reserve_path(taken_at, media.video_extension)
        try:
            shutil.move(media.main_file, target)
        finally:
            self._library.release_path(target)
        if media.overlay is not None:
            self._save_video_overlay(media.overlay, taken_at)
        return target

    def _save_video_overlay(self, overlay: bytes, taken_at: datetime) -> None:
        try:
            with Image.open(io.BytesIO(overlay)) as layer:
                output = io.BytesIO()
                layer.save(output, format="PNG")
        except IMAGE_ERRORS:
            return
        self._write_bytes(output.getvalue(), taken_at, "png", suffix="_overlay")

    def _write_bytes(
        self, content: bytes, taken_at: datetime, extension: str, suffix: str = ""
    ) -> Path:
        target = self._library.reserve_path(taken_at, extension, suffix)
        partial = target.with_name(target.name + ".part")
        try:
            partial.write_bytes(content)
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
            self._library.release_path(target)
        return target

    def _read_member(self, member: ArchiveMember) -> bytes:
        return self._archive(member.archive).read(member.name)

    def _archive(self, path: Path) -> zipfile.ZipFile:
        cache: dict[Path, zipfile.ZipFile] = self._archives.__dict__.setdefault("by_path", {})
        archive = cache.get(path)
        if archive is None:
            archive = zipfile.ZipFile(path)
            cache[path] = archive
            with self._opened_lock:
                self._opened.append(archive)
        return archive

    def _copy(self, source: IO[bytes], target: IO[bytes]) -> None:
        while chunk := source.read(COPY_BUFFER_BYTES):
            target.write(chunk)
            self._on_bytes(len(chunk))


def _render_with_fallback(
    main: bytes, overlay: bytes | None, taken_at: datetime, location: GeoPoint | None
) -> bytes:
    attempts = (overlay, None) if overlay is not None else (None,)
    for candidate in attempts:
        try:
            return render_photo(main, candidate, taken_at, location)
        except IMAGE_ERRORS:
            continue
    return main
