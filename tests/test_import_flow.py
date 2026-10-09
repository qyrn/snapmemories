import io
import json
import zipfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from PIL import ExifTags, Image
from pytest import MonkeyPatch

from snapmemories.export_archive import read_export
from snapmemories.job import ImportJob
from snapmemories.library import Library
from snapmemories.models import RemoteMemory
from snapmemories.planning import build_plan
from tests.factories import (
    history_entry,
    jpeg_bytes,
    mp4_bytes,
    overlay_webp_bytes,
    write_export_zip,
    zip_member,
)


def build_export(tmp_path: Path) -> Path:
    return write_export_zip(
        tmp_path / "mydata.zip",
        [
            history_entry(
                "2025-06-01 00:11:34 UTC", "Image", "Latitude, Longitude: 48.89948, 6.049386"
            ),
            history_entry("2026-05-08 17:01:16 UTC", "Image"),
            history_entry("2026-02-14 16:28:15 UTC", "Video", "Latitude, Longitude: 45.4, 4.3"),
            history_entry("2026-03-01 10:00:00 UTC", "Image"),
        ],
        [
            zip_member(
                "memories/2025-06-01_A-main.jpg", jpeg_bytes(), datetime(2025, 6, 1, 0, 11, 34)
            ),
            zip_member(
                "memories/2026-05-08_B-main.jpg", jpeg_bytes(), datetime(2026, 5, 8, 17, 1, 16)
            ),
            zip_member(
                "memories/2026-05-08_B-overlay.png",
                overlay_webp_bytes(),
                datetime(2026, 5, 8, 17, 1, 16),
            ),
            zip_member(
                "memories/2026-02-14_C-main.mp4", mp4_bytes(), datetime(2026, 2, 14, 16, 28, 14)
            ),
        ],
    )


def run_import(export: Path, output: Path, work: Path) -> dict[str, object]:
    job = ImportJob(build_plan(read_export([export])), Library(output), work)
    job.start()
    job.wait(timeout=30)
    return job.snapshot()


def test_imports_embedded_memories_then_resumes(tmp_path: Path) -> None:
    export = build_export(tmp_path)
    output = tmp_path / "Memories"
    work = tmp_path / "work"
    work.mkdir()

    first = run_import(export, output, work)
    second = run_import(export, output, work)

    assert (first["saved"], first["failed"], first["missing"]) == (3, 0, 1)
    assert (second["saved"], second["skipped"]) == (0, 3)
    assert list(work.iterdir()) == []
    catalog = (output / ".snapmemories" / "catalog.jsonl").read_text(encoding="utf-8")
    assert len(catalog.splitlines()) == 3
    assert {json.loads(line)["kind"] for line in catalog.splitlines()} == {"photo", "video"}


def test_overlay_is_merged_and_gps_written(tmp_path: Path) -> None:
    export = build_export(tmp_path)
    output = tmp_path / "Memories"
    (tmp_path / "work").mkdir()

    run_import(export, output, tmp_path / "work")

    photos = sorted(output.rglob("*.jpg"))
    assert len(photos) == 2
    with Image.open(io.BytesIO(photos[0].read_bytes())) as located:
        gps = located.getexif().get_ifd(ExifTags.IFD.GPSInfo)
        assert gps[ExifTags.GPS.GPSLatitudeRef] == "N"
    with Image.open(photos[1]) as merged:
        top_left = merged.convert("RGB").getpixel((2, 2))
        bottom_right = merged.convert("RGB").getpixel((60, 90))
    assert isinstance(top_left, tuple) and top_left[2] > 200
    assert isinstance(bottom_right, tuple) and bottom_right[0] > 150
    assert len(list(output.rglob("*.mp4"))) == 1


def test_downloaded_bundles_are_unpacked_and_merged(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("abc-main.jpg", jpeg_bytes())
        archive.writestr("abc-overlay.png", overlay_webp_bytes())

    def fake_download(
        memory: RemoteMemory, target: Path, session: object, on_bytes: Callable[[int], None]
    ) -> None:
        target.write_bytes(bundle.getvalue())

    monkeypatch.setattr("snapmemories.importer.download_memory", fake_download)
    export = tmp_path / "memories_history.json"
    entry = history_entry("2024-01-02 03:04:05 UTC", "Image")
    entry["Media Download Url"] = "https://example.test/media?mid=abc"
    export.write_text(json.dumps({"Saved Media": [entry]}), encoding="utf-8")
    (tmp_path / "work").mkdir()

    result = run_import(export, tmp_path / "Memories", tmp_path / "work")

    assert (result["saved"], result["failed"]) == (1, 0)
    [photo] = (tmp_path / "Memories").rglob("*.jpg")
    with Image.open(photo) as merged:
        pixel = merged.convert("RGB").getpixel((2, 2))
    assert isinstance(pixel, tuple) and pixel[2] > 200
