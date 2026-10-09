import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

from snapmemories.export_json import ExportFormatError, parse_memories_history
from snapmemories.models import ArchiveMember, EmbeddedMemory, MediaKind, MemoryEntry

HISTORY_FILE_NAME = "memories_history.json"
MEMBER_PATTERN = re.compile(r"^(?P<key>.+?)-(?P<role>main|overlay)$", re.IGNORECASE)
PHOTO_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".heic"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov"})


@dataclass(frozen=True, slots=True)
class ExportContents:
    entries: list[MemoryEntry]
    embedded: list[EmbeddedMemory]


@dataclass(slots=True)
class _MemberGroup:
    main: zipfile.ZipInfo | None = None
    overlay: zipfile.ZipInfo | None = None


def read_export(sources: list[Path]) -> ExportContents:
    entries: list[MemoryEntry] | None = None
    embedded: list[EmbeddedMemory] = []
    for source in sources:
        if source.suffix.lower() == ".json":
            entries = parse_memories_history(source.read_bytes())
            continue
        try:
            with zipfile.ZipFile(source) as archive:
                history = _find_history(archive)
                if history is not None and entries is None:
                    entries = parse_memories_history(archive.read(history))
                embedded.extend(list_embedded_memories(source, archive.infolist()))
        except zipfile.BadZipFile as error:
            raise ExportFormatError(f"{source.name} is not a valid ZIP file.") from error
    if entries is None and not embedded:
        raise ExportFormatError(
            "No memories found. Make sure you dropped the ZIP sent by Snapchat "
            "with the Memories option checked."
        )
    return ExportContents(entries=entries or [], embedded=_deduplicate(embedded))


def _find_history(archive: zipfile.ZipFile) -> str | None:
    for name in archive.namelist():
        if PurePosixPath(name).name == HISTORY_FILE_NAME:
            return name
    return None


def _deduplicate(embedded: list[EmbeddedMemory]) -> list[EmbeddedMemory]:
    return list({memory.key: memory for memory in embedded}.values())


def list_embedded_memories(
    archive_path: Path, members: list[zipfile.ZipInfo]
) -> list[EmbeddedMemory]:
    groups: dict[str, _MemberGroup] = {}
    for info in members:
        if info.is_dir():
            continue
        path = PurePosixPath(info.filename)
        extension = path.suffix.lower()
        if extension not in PHOTO_EXTENSIONS and extension not in VIDEO_EXTENSIONS:
            continue
        match = MEMBER_PATTERN.match(path.stem)
        key = match.group("key") if match else path.stem
        group = groups.setdefault(key, _MemberGroup())
        if match and match.group("role").lower() == "overlay":
            group.overlay = info
        else:
            group.main = info

    memories: list[EmbeddedMemory] = []
    for key, group in groups.items():
        if group.main is None:
            continue
        main_extension = PurePosixPath(group.main.filename).suffix.lower()
        memories.append(
            EmbeddedMemory(
                key=key,
                kind=MediaKind.VIDEO if main_extension in VIDEO_EXTENSIONS else MediaKind.PHOTO,
                main=_member(archive_path, group.main),
                overlay=_member(archive_path, group.overlay) if group.overlay else None,
                archived_at=datetime(*group.main.date_time),
            )
        )
    return memories


def _member(archive_path: Path, info: zipfile.ZipInfo) -> ArchiveMember:
    return ArchiveMember(archive=archive_path, name=info.filename, size=info.file_size)
