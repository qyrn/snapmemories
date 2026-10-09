import struct
from datetime import UTC, datetime
from pathlib import Path

import pytest

from snapmemories.models import GeoPoint
from snapmemories.mp4_metadata import (
    MP4_EPOCH_OFFSET_SECONDS,
    Mp4FormatError,
    write_capture_metadata,
)
from tests.factories import mp4_bytes

TAKEN = datetime(2026, 2, 14, 16, 28, 15, tzinfo=UTC)
MEDIA = b"\x07" * 64


def chunk_offset(data: bytes) -> int:
    position = data.index(b"stco")
    return int(struct.unpack_from(">I", data, position + 12)[0])


def creation_time(data: bytes, kind: bytes) -> int:
    position = data.index(kind)
    return int(struct.unpack_from(">I", data, position + 8)[0])


@pytest.mark.parametrize("moov_first", [True, False])
def test_sets_creation_times_and_location_keeping_media_reachable(
    tmp_path: Path, moov_first: bool
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(mp4_bytes(moov_first=moov_first, media=MEDIA))

    write_capture_metadata(video, TAKEN, GeoPoint(45.44657, 4.389548))

    data = video.read_bytes()
    expected = int(TAKEN.timestamp()) + MP4_EPOCH_OFFSET_SECONDS
    for kind in (b"mvhd", b"tkhd", b"mdhd"):
        assert creation_time(data, kind) == expected
    assert b"\xa9xyz" in data
    assert b"+45.4466+004.3895/" in data
    offset = chunk_offset(data)
    assert data[offset : offset + len(MEDIA)] == MEDIA


def test_without_location_the_file_size_is_unchanged(tmp_path: Path) -> None:
    original = mp4_bytes(media=MEDIA)
    video = tmp_path / "clip.mp4"
    video.write_bytes(original)

    write_capture_metadata(video, TAKEN, None)

    data = video.read_bytes()
    assert len(data) == len(original)
    assert data.endswith(MEDIA)


def test_rejects_files_without_movie_box(tmp_path: Path) -> None:
    video = tmp_path / "broken.mp4"
    video.write_bytes(struct.pack(">I4s", 16, b"mdat") + b"\x00" * 8)

    with pytest.raises(Mp4FormatError):
        write_capture_metadata(video, TAKEN, None)
