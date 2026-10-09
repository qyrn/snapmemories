import shutil
import threading
from pathlib import Path, PurePath
from typing import IO

from snapmemories.export_archive import read_export
from snapmemories.export_json import ExportFormatError
from snapmemories.job import ImportJob
from snapmemories.library import Library
from snapmemories.planning import ImportPlan, build_plan

ACCEPTED_EXTENSIONS = frozenset({".zip", ".json"})
MAX_UPLOADS = 64
UPLOAD_CHUNK_BYTES = 1024 * 1024
DISK_SAFETY_MARGIN_BYTES = 200 * 1024 * 1024


class SessionError(Exception):
    pass


class ImportSession:
    def __init__(self, work_directory: Path, library: Library) -> None:
        self._work_directory = work_directory
        self._uploads_directory = work_directory / "uploads"
        self._library = library
        self._lock = threading.Lock()
        self._uploads: list[tuple[str, Path]] = []
        self._plan: ImportPlan | None = None
        self._job: ImportJob | None = None

    @property
    def is_busy(self) -> bool:
        return self._job is not None and self._job.is_running

    @property
    def output_directory(self) -> Path:
        return self._library.root

    def add_upload(
        self, display_name: str, stream: IO[bytes], declared_size: int | None
    ) -> list[str]:
        name = PurePath(display_name.replace("\\", "/")).name.strip()
        extension = PurePath(name).suffix.lower()
        if extension not in ACCEPTED_EXTENSIONS:
            raise SessionError(f"{name or 'This file'} is not a .zip file from Snapchat.")
        with self._lock:
            self._ensure_idle()
            if len(self._uploads) >= MAX_UPLOADS:
                raise SessionError("Too many files. Drop only the ZIP files sent by Snapchat.")
            self._uploads_directory.mkdir(parents=True, exist_ok=True)
            if declared_size is not None:
                free = shutil.disk_usage(self._uploads_directory).free
                if free < declared_size + DISK_SAFETY_MARGIN_BYTES:
                    raise SessionError("Not enough free disk space to read this file.")
            target = self._uploads_directory / f"upload-{len(self._uploads) + 1}{extension}"
            self._plan = None

        try:
            with target.open("wb") as output:
                while chunk := stream.read(UPLOAD_CHUNK_BYTES):
                    output.write(chunk)
        except OSError as error:
            target.unlink(missing_ok=True)
            raise SessionError("The file could not be received. Please try again.") from error

        with self._lock:
            self._uploads.append((name, target))
            return [display for display, _ in self._uploads]

    def analyze(self) -> dict[str, object]:
        with self._lock:
            self._ensure_idle()
            sources = [path for _, path in self._uploads]
        if not sources:
            raise SessionError("Drop your Snapchat ZIP file first.")
        try:
            plan = build_plan(read_export(sources))
        except ExportFormatError as error:
            raise SessionError(str(error)) from error
        if not plan.items:
            raise SessionError(_empty_plan_message(plan.missing))

        already_saved = self._library.existing_records()
        with self._lock:
            self._plan = plan
        output = self._library.root
        output.mkdir(parents=True, exist_ok=True)
        free_bytes = shutil.disk_usage(output).free
        return {
            "files": [display for display, _ in self._uploads],
            "photos": plan.photos,
            "videos": plan.videos,
            "total": len(plan.items),
            "located": plan.located,
            "missing": plan.missing,
            "already_saved": sum(1 for item in plan.items if item.item_id in already_saved),
            "needs_network": plan.needs_network,
            "estimated_bytes": plan.estimated_bytes,
            "free_bytes": free_bytes,
            "enough_space": free_bytes >= plan.estimated_bytes + DISK_SAFETY_MARGIN_BYTES,
            "output_directory": str(output),
        }

    def start(self) -> None:
        with self._lock:
            self._ensure_idle()
            if self._plan is None:
                raise SessionError("Analyze your Snapchat export first.")
            self._job = ImportJob(self._plan, self._library, self._work_directory)
            self._job.start()

    def cancel(self) -> None:
        with self._lock:
            if self._job is not None:
                self._job.cancel()

    def progress(self) -> dict[str, object] | None:
        with self._lock:
            job = self._job
        return job.snapshot() if job is not None else None

    def reset(self) -> None:
        with self._lock:
            self._ensure_idle()
            self._job = None
            self._plan = None
            self._uploads.clear()
            shutil.rmtree(self._uploads_directory, ignore_errors=True)

    def _ensure_idle(self) -> None:
        if self.is_busy:
            raise SessionError("An import is already running.")


def _empty_plan_message(missing: int) -> str:
    if missing:
        return (
            f"{missing} memories are listed but their files are not in the dropped ZIP. "
            "If Snapchat split your export into several ZIP files, drop all of them together."
        )
    return "No memories found in this export."
