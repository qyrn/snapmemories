import threading
import time


class Heartbeat:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_beat = time.monotonic()

    def beat(self) -> None:
        with self._lock:
            self._last_beat = time.monotonic()

    def seconds_since_last_beat(self) -> float:
        with self._lock:
            return time.monotonic() - self._last_beat
