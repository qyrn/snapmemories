import io
from datetime import datetime
from fractions import Fraction

from PIL import ExifTags, Image
from PIL.TiffImagePlugin import IFDRational

from snapmemories.models import GeoPoint

START_OF_IMAGE = b"\xff\xd8"
APP0 = 0xE0
APP1 = 0xE1
START_OF_SCAN = 0xDA
END_OF_IMAGE = 0xD9
EXIF_HEADER = b"Exif\x00\x00"
MAX_SEGMENT_LENGTH = 0xFFFF
EXIF_DATE_FORMAT = "%Y:%m:%d %H:%M:%S"
SECONDS_PRECISION = 10_000


def with_capture_metadata(jpeg: bytes, taken_at: datetime, location: GeoPoint | None) -> bytes:
    with Image.open(io.BytesIO(jpeg)) as image:
        exif = image.getexif()

    stamp = taken_at.strftime(EXIF_DATE_FORMAT)
    offset = format_utc_offset(taken_at)
    exif[ExifTags.Base.DateTime] = stamp
    exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
    exif_ifd[ExifTags.Base.DateTimeOriginal] = stamp
    exif_ifd[ExifTags.Base.DateTimeDigitized] = stamp
    exif_ifd[ExifTags.Base.OffsetTime] = offset
    exif_ifd[ExifTags.Base.OffsetTimeOriginal] = offset
    exif_ifd[ExifTags.Base.OffsetTimeDigitized] = offset

    if location is not None:
        gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
        gps_ifd[ExifTags.GPS.GPSVersionID] = b"\x02\x03\x00\x00"
        gps_ifd[ExifTags.GPS.GPSLatitudeRef] = "N" if location.latitude >= 0 else "S"
        gps_ifd[ExifTags.GPS.GPSLatitude] = degrees_to_rationals(location.latitude)
        gps_ifd[ExifTags.GPS.GPSLongitudeRef] = "E" if location.longitude >= 0 else "W"
        gps_ifd[ExifTags.GPS.GPSLongitude] = degrees_to_rationals(location.longitude)

    return replace_exif_segment(jpeg, exif.tobytes())


def format_utc_offset(moment: datetime) -> str:
    offset = moment.utcoffset()
    total_minutes = int(offset.total_seconds() // 60) if offset is not None else 0
    sign = "+" if total_minutes >= 0 else "-"
    hours, minutes = divmod(abs(total_minutes), 60)
    return f"{sign}{hours:02d}:{minutes:02d}"


def degrees_to_rationals(value: float) -> tuple[IFDRational, IFDRational, IFDRational]:
    absolute = abs(value)
    degrees = int(absolute)
    minutes_float = (absolute - degrees) * 60
    minutes = int(minutes_float)
    seconds = round((minutes_float - minutes) * 60 * SECONDS_PRECISION)
    return (
        IFDRational(degrees, 1),
        IFDRational(minutes, 1),
        IFDRational(Fraction(seconds, SECONDS_PRECISION)),
    )


def replace_exif_segment(jpeg: bytes, exif_payload: bytes) -> bytes:
    if not jpeg.startswith(START_OF_IMAGE):
        raise ValueError("Not a JPEG file.")
    if not exif_payload.startswith(EXIF_HEADER):
        exif_payload = EXIF_HEADER + exif_payload
    if len(exif_payload) + 2 > MAX_SEGMENT_LENGTH:
        raise ValueError("EXIF block too large.")

    leading: list[bytes] = []
    kept: list[bytes] = []
    position = len(START_OF_IMAGE)
    while position + 4 <= len(jpeg) and jpeg[position] == 0xFF:
        marker = jpeg[position + 1]
        if marker == 0xFF:
            position += 1
            continue
        if marker in (START_OF_SCAN, END_OF_IMAGE):
            break
        length = int.from_bytes(jpeg[position + 2 : position + 4], "big")
        segment = jpeg[position : position + 2 + length]
        position += 2 + length
        if marker == APP1 and segment[4:10] == EXIF_HEADER:
            continue
        if marker == APP0 and not kept:
            leading.append(segment)
        else:
            kept.append(segment)

    exif_segment = bytes((0xFF, APP1)) + (len(exif_payload) + 2).to_bytes(2, "big") + exif_payload
    return b"".join((START_OF_IMAGE, *leading, exif_segment, *kept, jpeg[position:]))
