from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path


class MediaKind(StrEnum):
    PHOTO = "photo"
    VIDEO = "video"


@dataclass(frozen=True, slots=True)
class GeoPoint:
    latitude: float
    longitude: float


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    taken_at: datetime
    kind: MediaKind
    location: GeoPoint | None
    media_download_url: str
    download_link: str

    @property
    def is_downloadable(self) -> bool:
        return bool(self.media_download_url or self.download_link)


@dataclass(frozen=True, slots=True)
class ArchiveMember:
    archive: Path
    name: str
    size: int


@dataclass(frozen=True, slots=True)
class EmbeddedMemory:
    key: str
    kind: MediaKind
    main: ArchiveMember
    overlay: ArchiveMember | None
    archived_at: datetime


@dataclass(frozen=True, slots=True)
class RemoteMemory:
    media_download_url: str
    download_link: str


@dataclass(frozen=True, slots=True)
class ImportItem:
    item_id: str
    kind: MediaKind
    taken_at: datetime
    location: GeoPoint | None
    source: EmbeddedMemory | RemoteMemory
