from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from snapmemories.models import EmbeddedMemory, MemoryEntry

OFFSET_STEP_SECONDS = 900
MAX_OFFSET_SECONDS = 14 * 3600
TOLERANCE_SECONDS = 2


@dataclass(frozen=True, slots=True)
class MatchResult:
    pairs: list[tuple[EmbeddedMemory, MemoryEntry | None]]
    unmatched_entries: list[MemoryEntry]
    archive_offset: timedelta


def match_embedded(entries: list[MemoryEntry], embedded: list[EmbeddedMemory]) -> MatchResult:
    candidates = {
        memory.key: [
            (index, offset)
            for index, entry in enumerate(entries)
            if entry.kind == memory.kind
            and (offset := _archive_offset(memory.archived_at, entry.taken_at)) is not None
        ]
        for memory in embedded
    }
    offset_popularity = Counter(offset for options in candidates.values() for _, offset in options)
    used: set[int] = set()
    pairs: list[tuple[EmbeddedMemory, MemoryEntry | None]] = []
    for memory in sorted(embedded, key=lambda item: item.archived_at):
        options = [option for option in candidates[memory.key] if option[0] not in used]
        if not options:
            pairs.append((memory, None))
            continue
        index, _ = max(
            options,
            key=lambda option: (offset_popularity[option[1]], -abs(option[1])),
        )
        used.add(index)
        pairs.append((memory, entries[index]))

    dominant = offset_popularity.most_common(1)
    return MatchResult(
        pairs=pairs,
        unmatched_entries=[entry for index, entry in enumerate(entries) if index not in used],
        archive_offset=timedelta(seconds=dominant[0][0] if dominant else 0),
    )


def _archive_offset(archived_at: datetime, taken_at: datetime) -> int | None:
    delta = (archived_at - taken_at.astimezone(UTC).replace(tzinfo=None)).total_seconds()
    rounded = round(delta / OFFSET_STEP_SECONDS) * OFFSET_STEP_SECONDS
    if abs(delta - rounded) > TOLERANCE_SECONDS or abs(rounded) > MAX_OFFSET_SECONDS:
        return None
    return rounded
