import io
from datetime import datetime, timedelta, timezone

import pytest
from PIL import ExifTags, Image

from snapmemories.jpeg_exif import replace_exif_segment, with_capture_metadata
from snapmemories.models import GeoPoint
from tests.factories import jpeg_bytes

PARIS_SUMMER = timezone(timedelta(hours=2))


def scan_data(jpeg: bytes) -> bytes:
    return jpeg[jpeg.index(b"\xff\xda") :]


def test_writes_dates_offsets_and_gps_without_reencoding() -> None:
    original = jpeg_bytes()
    taken = datetime(2025, 6, 1, 2, 11, 34, tzinfo=PARIS_SUMMER)

    result = with_capture_metadata(original, taken, GeoPoint(48.89948, -6.049386))

    assert scan_data(result) == scan_data(original)
    with Image.open(io.BytesIO(result)) as image:
        exif = image.getexif()
    details = exif.get_ifd(ExifTags.IFD.Exif)
    gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
    assert details[ExifTags.Base.DateTimeOriginal] == "2025:06:01 02:11:34"
    assert details[ExifTags.Base.OffsetTimeOriginal] == "+02:00"
    assert gps[ExifTags.GPS.GPSLatitudeRef] == "N"
    assert gps[ExifTags.GPS.GPSLongitudeRef] == "W"
    degrees, minutes, seconds = gps[ExifTags.GPS.GPSLatitude]
    assert float(degrees) + float(minutes) / 60 + float(seconds) / 3600 == pytest.approx(48.89948)


def test_replaces_an_existing_exif_block_instead_of_stacking() -> None:
    taken = datetime(2025, 1, 1, tzinfo=PARIS_SUMMER)
    once = with_capture_metadata(jpeg_bytes(), taken, None)

    twice = with_capture_metadata(once, taken, None)

    assert twice.count(b"Exif\x00\x00") == 1


def test_rejects_non_jpeg_data() -> None:
    with pytest.raises(ValueError, match="JPEG"):
        replace_exif_segment(b"\x89PNG", b"Exif\x00\x00")
