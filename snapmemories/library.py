import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from snapmemories.models import MediaKind

STATE_DIRECTORY = ".snapmemories"
CATALOG_FILE = "catalog.jsonl"
THUMBNAIL_DIRECTORY = "thumbnails"
FILE_STEM_FORMAT = "%Y-%m-%d_%H-%M-%S"


@dataclass(frozen=True, slots=True)
class LibraryRecord:
    item_id: str
    relative_path: str
    kind: MediaKind
    taken_at: str
    latitude: float | None
    longitude: float | None


class Library:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = threading.Lock()
        self._reserved: set[Path] = set()

    @property
    def state_directory(self) -> Path:
        return self.root / STATE_DIRECTORY

    @property
    def thumbnail_directory(self) -> Path:
        return self.state_directory / THUMBNAIL_DIRECTORY

    @property
    def catalog_path(self) -> Path:
        return self.state_directory / CATALOG_FILE

    def records(self) -> dict[str, LibraryRecord]:
        records: dict[str, LibraryRecord] = {}
        try:
            lines = self.catalog_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return records
        for line in lines:
            record = _parse_record(line)
            if record is not None:
                records[record.item_id] = record
        return records

    def existing_records(self) -> dict[str, LibraryRecord]:
        return {
            item_id: record
            for item_id, record in self.records().items()
            if self.resolve(record) is not None
        }

    def resolve(self, record: LibraryRecord) -> Path | None:
        root = self.root.resolve()
        candidate = (root / record.relative_path).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            return None
        return candidate

    def reserve_path(self, taken_at: datetime, extension: str, suffix: str = "") -> Path:
        folder = self.root / f"{taken_at:%Y}" / f"{taken_at:%Y-%m}"
        folder.mkdir(parents=True, exist_ok=True)
        stem = taken_at.strftime(FILE_STEM_FORMAT) + suffix
        with self._lock:
            attempt = 1
            while True:
                name = stem if attempt == 1 else f"{stem}_{attempt}"
                candidate = folder / f"{name}.{extension}"
                if candidate not in self._reserved and not candidate.exists():
                    self._reserved.add(candidate)
                    return candidate
                attempt += 1

    def release_path(self, path: Path) -> None:
        with self._lock:
            self._reserved.discard(path)

    def add(self, record: LibraryRecord) -> None:
        line = json.dumps(asdict(record), ensure_ascii=False)
        with self._lock:
            self.state_directory.mkdir(parents=True, exist_ok=True)
            with self.catalog_path.open("a", encoding="utf-8") as catalog:
                catalog.write(line + "\n")

    def relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()


def _parse_record(line: str) -> LibraryRecord | None:
    try:
        raw: object = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    try:
        return LibraryRecord(
            item_id=str(raw["item_id"]),
            relative_path=str(raw["relative_path"]),
            kind=MediaKind(raw["kind"]),
            taken_at=str(raw["taken_at"]),
            latitude=_optional_float(raw.get("latitude")),
            longitude=_optional_float(raw.get("longitude")),
        )
    except KeyError, ValueError:
        return None


def _optional_float(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None
