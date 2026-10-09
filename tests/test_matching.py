from datetime import UTC, datetime, timedelta
from pathlib import Path

from snapmemories.matching import match_embedded
from snapmemories.models import ArchiveMember, EmbeddedMemory, GeoPoint, MediaKind, MemoryEntry


def entry(moment: datetime, kind: MediaKind = MediaKind.PHOTO) -> MemoryEntry:
    return MemoryEntry(moment, kind, GeoPoint(1, 1), "", "")


def embedded(key: str, archived_at: datetime, kind: MediaKind = MediaKind.PHOTO) -> EmbeddedMemory:
    member = ArchiveMember(Path("export.zip"), f"memories/{key}-main.jpg", 10)
    return EmbeddedMemory(key, kind, member, None, archived_at)


def test_matches_utc_archive_times_with_dos_rounding() -> None:
    taken = datetime(2025, 6, 16, 10, 43, 33, tzinfo=UTC)
    memory = embedded("a", datetime(2025, 6, 16, 10, 43, 32))

    result = match_embedded([entry(taken)], [memory])

    assert result.pairs == [(memory, entry(taken))]
    assert result.unmatched_entries == []


def test_prefers_the_dominant_offset_for_local_archive_times() -> None:
    first = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    second = first + timedelta(minutes=15)
    third = datetime(2025, 1, 2, 8, 0, 0, tzinfo=UTC)
    local = timedelta(hours=1)
    memories = [
        embedded("first", (first + local).replace(tzinfo=None)),
        embedded("second", (second + local).replace(tzinfo=None)),
        embedded("third", (third + local).replace(tzinfo=None)),
    ]

    result = match_embedded([entry(first), entry(second), entry(third)], memories)

    matched = {memory.key: matched_entry for memory, matched_entry in result.pairs}
    assert matched["first"] == entry(first)
    assert matched["second"] == entry(second)
    assert result.archive_offset == local


def test_never_matches_a_different_media_kind() -> None:
    taken = datetime(2025, 1, 1, tzinfo=UTC)
    memory = embedded("video", taken.replace(tzinfo=None), MediaKind.VIDEO)

    result = match_embedded([entry(taken, MediaKind.PHOTO)], [memory])

    assert result.pairs == [(memory, None)]
    assert len(result.unmatched_entries) == 1
