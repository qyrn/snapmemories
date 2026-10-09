import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import requests

from snapmemories.downloader import ExpiredLinkError, new_session
from snapmemories.importer import MemoryImporter
from snapmemories.library import Library
from snapmemories.models import ImportItem
from snapmemories.planning import ImportPlan

LOCAL_WORKERS = 4
NETWORK_WORKERS = 8
MAX_REPORTED_ERRORS = 200
SPEED_WINDOW_SECONDS = 6.0


class JobPhase(StrEnum):
    RUNNING = "running"
    DONE = "done"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class _Counters:
    saved: int = 0
    skipped: int = 0
    failed: int = 0
    expired: int = 0
    bytes_processed: int = 0
    errors: list[str] = field(default_factory=list)


class ImportJob:
    def __init__(self, plan: ImportPlan, library: Library, work_directory: Path) -> None:
        self._plan = plan
        self._library = library
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._counters = _Counters()
        self._phase = JobPhase.RUNNING
        self._started_at = time.monotonic()
        self._finished_at: float | None = None
        self._speed_samples: deque[tuple[float, int]] = deque()
        self._sessions = threading.local()
        self._importer = MemoryImporter(library, work_directory, self._count_bytes)
        self._thread = threading.Thread(target=self._run, name="import-job", daemon=True)

    @property
    def is_running(self) -> bool:
        return self._phase == JobPhase.RUNNING

    def start(self) -> None:
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()

    def wait(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def snapshot(self) -> dict[str, object]:
        now = time.monotonic()
        with self._lock:
            counters = self._counters
            total = len(self._plan.items)
            processed = counters.saved + counters.skipped + counters.failed + counters.expired
            self._speed_samples.append((now, counters.bytes_processed))
            while self._speed_samples and now - self._speed_samples[0][0] > SPEED_WINDOW_SECONDS:
                self._speed_samples.popleft()
            oldest_time, oldest_bytes = self._speed_samples[0]
            window = now - oldest_time
            speed = (counters.bytes_processed - oldest_bytes) / window if window > 0 else 0.0
            elapsed = (self._finished_at or now) - self._started_at
            worked = processed - counters.skipped
            remaining = total - processed
            eta = elapsed / worked * remaining if worked > 0 and remaining > 0 else 0.0
            return {
                "phase": self._phase.value,
                "total": total,
                "processed": processed,
                "saved": counters.saved,
                "skipped": counters.skipped,
                "failed": counters.failed,
                "expired": counters.expired,
                "missing": self._plan.missing,
                "bytes_per_second": round(speed),
                "elapsed_seconds": round(elapsed),
                "eta_seconds": round(eta),
                "errors": list(counters.errors),
                "output_directory": str(self._library.root),
            }

    def _run(self) -> None:
        already_saved = self._library.existing_records()
        pending = [item for item in self._plan.items if item.item_id not in already_saved]
        with self._lock:
            self._counters.skipped = len(self._plan.items) - len(pending)
        workers = NETWORK_WORKERS if self._plan.needs_network else LOCAL_WORKERS
        try:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="import") as pool:
                for item in pending:
                    pool.submit(self._import_one, item)
        finally:
            self._importer.close()
            with self._lock:
                self._phase = JobPhase.CANCELLED if self._cancel.is_set() else JobPhase.DONE
                self._finished_at = time.monotonic()

    def _import_one(self, item: ImportItem) -> None:
        if self._cancel.is_set():
            return
        try:
            self._importer.import_item(item, self._session() if self._plan.needs_network else None)
        except ExpiredLinkError:
            with self._lock:
                self._counters.expired += 1
            return
        except Exception as error:
            self._record_failure(item, error)
            return
        with self._lock:
            self._counters.saved += 1

    def _record_failure(self, item: ImportItem, error: Exception) -> None:
        label = item.taken_at.astimezone().strftime("%Y-%m-%d %H:%M")
        reason = str(error) or type(error).__name__
        with self._lock:
            self._counters.failed += 1
            if len(self._counters.errors) < MAX_REPORTED_ERRORS:
                self._counters.errors.append(f"{label} ({item.kind.value}): {reason[:160]}")

    def _session(self) -> requests.Session:
        session: requests.Session | None = getattr(self._sessions, "value", None)
        if session is None:
            session = new_session()
            self._sessions.value = session
        return session

    def _count_bytes(self, amount: int) -> None:
        with self._lock:
            self._counters.bytes_processed += amount
