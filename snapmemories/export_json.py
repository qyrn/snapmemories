import json
import re
from datetime import UTC, datetime
from typing import cast

from snapmemories.models import GeoPoint, MediaKind, MemoryEntry

ENTRY_LIST_KEYS = ("Saved Media", "SavedMedia", "saved_media", "Memories", "memories")
DATE_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f")
COORDINATES_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)")
VIDEO_TYPES = frozenset({"VIDEO", "MOVIE"})


class ExportFormatError(ValueError):
    pass


def parse_memories_history(raw: bytes) -> list[MemoryEntry]:
    try:
        document: object = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExportFormatError("memories_history.json is not valid JSON.") from error

    entries: list[MemoryEntry] = []
    for item in _find_entry_list(document):
        if isinstance(item, dict):
            entry = _parse_entry(cast(dict[str, object], item))
            if entry is not None:
                entries.append(entry)
    return entries


def _find_entry_list(document: object) -> list[object]:
    if isinstance(document, list):
        return cast(list[object], document)
    if not isinstance(document, dict):
        raise ExportFormatError("memories_history.json has an unexpected structure.")
    mapping = cast(dict[str, object], document)
    for key in ENTRY_LIST_KEYS:
        value = mapping.get(key)
        if isinstance(value, list):
            return cast(list[object], value)
    for value in mapping.values():
        if isinstance(value, list):
            return cast(list[object], value)
    raise ExportFormatError("No memories list found in memories_history.json.")


def _parse_entry(item: dict[str, object]) -> MemoryEntry | None:
    taken_at = parse_snapchat_date(_text(item, "Date"))
    if taken_at is None:
        return None
    media_type = _text(item, "Media Type").strip().upper()
    return MemoryEntry(
        taken_at=taken_at,
        kind=MediaKind.VIDEO if media_type in VIDEO_TYPES else MediaKind.PHOTO,
        location=parse_location(item.get("Location")),
        media_download_url=_text(item, "Media Download Url").strip(),
        download_link=_text(item, "Download Link").strip(),
    )


def _text(item: dict[str, object], key: str) -> str:
    value = item.get(key)
    return value if isinstance(value, str) else ""


def parse_snapchat_date(value: str) -> datetime | None:
    cleaned = value.strip().removesuffix("UTC").removesuffix("Z").strip()
    for date_format in DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, date_format).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def parse_location(value: object) -> GeoPoint | None:
    latitude: float | None = None
    longitude: float | None = None
    if isinstance(value, str):
        match = COORDINATES_PATTERN.search(value)
        if match:
            latitude, longitude = float(match.group(1)), float(match.group(2))
    elif isinstance(value, dict):
        mapping = cast(dict[str, object], value)
        latitude = _coordinate(mapping.get("Latitude", mapping.get("latitude")))
        longitude = _coordinate(mapping.get("Longitude", mapping.get("longitude")))
    if latitude is None or longitude is None:
        return None
    if latitude == 0 and longitude == 0:
        return None
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return None
    return GeoPoint(latitude, longitude)


def _coordinate(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
