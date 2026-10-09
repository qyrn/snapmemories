import io
import json
import struct
import zipfile
from datetime import datetime
from pathlib import Path

from PIL import Image


def jpeg_bytes(
    size: tuple[int, int] = (64, 96), color: tuple[int, int, int] = (200, 30, 30)
) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format="JPEG", quality=90)
    return output.getvalue()


def overlay_webp_bytes(size: tuple[int, int] = (32, 48)) -> bytes:
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    for x in range(size[0] // 2):
        for y in range(size[1] // 2):
            layer.putpixel((x, y), (0, 0, 255, 255))
    output = io.BytesIO()
    layer.save(output, format="WEBP", lossless=True)
    return output.getvalue()


def box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", len(payload) + 8, kind) + payload


def full_box(kind: bytes, version: int, payload: bytes) -> bytes:
    return box(kind, bytes((version, 0, 0, 0)) + payload)


def mp4_bytes(moov_first: bool = True, media: bytes = b"\x00" * 64) -> bytes:
    ftyp = box(b"ftyp", b"isom\x00\x00\x02\x00isomiso2mp41")
    mvhd = full_box(b"mvhd", 0, struct.pack(">IIII", 1, 1, 1000, 1000) + b"\x00" * 80)
    tkhd = full_box(b"tkhd", 0, struct.pack(">III", 1, 1, 1) + b"\x00" * 68)
    mdhd = full_box(b"mdhd", 0, struct.pack(">IIII", 1, 1, 1000, 1000) + b"\x00" * 4)

    def build(chunk_offset: int) -> bytes:
        stco = full_box(b"stco", 0, struct.pack(">II", 1, chunk_offset))
        stbl = box(b"stbl", stco)
        minf = box(b"minf", stbl)
        mdia = box(b"mdia", mdhd + minf)
        trak = box(b"trak", tkhd + mdia)
        return box(b"moov", mvhd + trak)

    mdat = box(b"mdat", media)
    if moov_first:
        moov_size = len(build(0))
        return ftyp + build(len(ftyp) + moov_size + 8) + mdat
    return ftyp + mdat + build(len(ftyp) + 8)


def zip_member(name: str, content: bytes, stored_at: datetime) -> tuple[zipfile.ZipInfo, bytes]:
    info = zipfile.ZipInfo(name, date_time=stored_at.timetuple()[:6])
    info.compress_type = zipfile.ZIP_DEFLATED
    return info, content


def write_export_zip(
    path: Path,
    entries: list[dict[str, str]],
    members: list[tuple[zipfile.ZipInfo, bytes]],
) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("json/memories_history.json", json.dumps({"Saved Media": entries}))
        for info, content in members:
            archive.writestr(info, content)
    return path


def history_entry(
    date: str, media_type: str, location: str = "Latitude, Longitude: 0.0, 0.0"
) -> dict[str, str]:
    return {
        "Date": date,
        "Media Type": media_type,
        "Location": location,
        "Download Link": "",
        "Media Download Url": "",
    }
