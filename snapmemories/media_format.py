from enum import StrEnum


class MediaFormat(StrEnum):
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    HEIC = "heic"
    MP4 = "mp4"
    ZIP = "zip"
    UNKNOWN = "unknown"


HEIC_BRANDS = frozenset({b"heic", b"heix", b"mif1", b"msf1", b"hevc"})


def sniff_format(header: bytes) -> MediaFormat:
    if header.startswith(b"\xff\xd8\xff"):
        return MediaFormat.JPEG
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return MediaFormat.PNG
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return MediaFormat.WEBP
    if header.startswith(b"PK\x03\x04"):
        return MediaFormat.ZIP
    if header[4:8] == b"ftyp":
        return MediaFormat.HEIC if header[8:12] in HEIC_BRANDS else MediaFormat.MP4
    return MediaFormat.UNKNOWN
