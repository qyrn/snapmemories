import hashlib
from dataclasses import dataclass
from datetime import UTC, timedelta
from urllib.parse import parse_qs, urlsplit

from snapmemories.export_archive import ExportContents
from snapmemories.matching import match_embedded
from snapmemories.models import EmbeddedMemory, ImportItem, MediaKind, MemoryEntry, RemoteMemory

ESTIMATED_PHOTO_BYTES = 3_000_000
ESTIMATED_VIDEO_BYTES = 15_000_000


@dataclass(frozen=True, slots=True)
class ImportPlan:
    items: list[ImportItem]
    missing: int

    @property
    def photos(self) -> int:
        return sum(1 for item in self.items if item.kind == MediaKind.PHOTO)

    @property
    def videos(self) -> int:
        return sum(1 for item in self.items if item.kind == MediaKind.VIDEO)

    @property
    def located(self) -> int:
        return sum(1 for item in self.items if item.location is not None)

    @property
    def needs_network(self) -> bool:
        return any(isinstance(item.source, RemoteMemory) for item in self.items)

    @property
    def estimated_bytes(self) -> int:
        return sum(_estimated_size(item) for item in self.items)


def build_plan(contents: ExportContents) -> ImportPlan:
    match = match_embedded(contents.entries, contents.embedded)
    items = [_embedded_item(memory, entry, match.archive_offset) for memory, entry in match.pairs]
    missing = 0
    for entry in match.unmatched_entries:
        if entry.is_downloadable:
            items.append(_remote_item(entry))
        else:
            missing += 1
    items.sort(key=lambda item: item.taken_at)
    return ImportPlan(items=items, missing=missing)


def _embedded_item(
    memory: EmbeddedMemory, entry: MemoryEntry | None, archive_offset: timedelta
) -> ImportItem:
    taken_at = (
        entry.taken_at
        if entry is not None
        else (memory.archived_at - archive_offset).replace(tzinfo=UTC)
    )
    return ImportItem(
        item_id=memory.key,
        kind=memory.kind,
        taken_at=taken_at,
        location=entry.location if entry else None,
        source=memory,
    )


def _remote_item(entry: MemoryEntry) -> ImportItem:
    return ImportItem(
        item_id=remote_item_id(entry),
        kind=entry.kind,
        taken_at=entry.taken_at,
        location=entry.location,
        source=RemoteMemory(entry.media_download_url, entry.download_link),
    )


def remote_item_id(entry: MemoryEntry) -> str:
    url = entry.download_link or entry.media_download_url
    media_id = next(iter(parse_qs(urlsplit(url).query).get("mid", [])), "")
    seed = media_id or url
    return "remote-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]


def _estimated_size(item: ImportItem) -> int:
    if isinstance(item.source, EmbeddedMemory):
        overlay = item.source.overlay.size if item.source.overlay else 0
        return item.source.main.size + overlay
    return ESTIMATED_VIDEO_BYTES if item.kind == MediaKind.VIDEO else ESTIMATED_PHOTO_BYTES
