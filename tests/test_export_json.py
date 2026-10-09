import json
from datetime import UTC, datetime

import pytest

from snapmemories.export_json import ExportFormatError, parse_location, parse_memories_history
from snapmemories.models import GeoPoint, MediaKind


def test_parses_current_snapchat_format() -> None:
    raw = json.dumps(
        {
            "Saved Media": [
                {
                    "Date": "2026-02-14 16:28:15 UTC",
                    "Media Type": "Video",
                    "Location": "Latitude, Longitude: 45.44657, 4.389548",
                    "Download Link": "",
                    "Media Download Url": "https://example.test/media",
                }
            ]
        }
    ).encode()

    [entry] = parse_memories_history(raw)

    assert entry.taken_at == datetime(2026, 2, 14, 16, 28, 15, tzinfo=UTC)
    assert entry.kind == MediaKind.VIDEO
    assert entry.location == GeoPoint(45.44657, 4.389548)
    assert entry.is_downloadable


def test_skips_entries_with_unreadable_dates() -> None:
    raw = json.dumps([{"Date": "yesterday", "Media Type": "Image"}]).encode()

    assert parse_memories_history(raw) == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Latitude, Longitude: 0.0, 0.0", None),
        ("Latitude, Longitude: -33.86, 151.2", GeoPoint(-33.86, 151.2)),
        ("Latitude, Longitude: 95.0, 10.0", None),
        ({"Latitude": "48.85", "Longitude": 2.35}, GeoPoint(48.85, 2.35)),
        ("", None),
        (None, None),
    ],
)
def test_location_parsing(value: object, expected: GeoPoint | None) -> None:
    assert parse_location(value) == expected


def test_rejects_invalid_json() -> None:
    with pytest.raises(ExportFormatError):
        parse_memories_history(b"{not json")
