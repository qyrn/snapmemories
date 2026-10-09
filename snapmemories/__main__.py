import shutil
import socket
import tempfile
import threading
import time
from pathlib import Path

import requests
from werkzeug.serving import make_server

from snapmemories.desktop import open_in_browser
from snapmemories.heartbeat import Heartbeat
from snapmemories.library import Library
from snapmemories.security import local_origins
from snapmemories.server import APP_IDENTIFIER, create_app
from snapmemories.session import ImportSession

HOST = "127.0.0.1"
PREFERRED_PORT = 7842
OUTPUT_DIRECTORY = Path.home() / "Memories"
IDLE_SHUTDOWN_SECONDS = 90
WATCHDOG_INTERVAL_SECONDS = 5


def main() -> None:
    if _running_instance_answers(PREFERRED_PORT):
        open_in_browser(f"http://{HOST}:{PREFERRED_PORT}")
        return

    port = PREFERRED_PORT if _port_is_free(PREFERRED_PORT) else _any_free_port()
    work_directory = Path(tempfile.mkdtemp(prefix="snapmemories-"))
    library = Library(OUTPUT_DIRECTORY)
    session = ImportSession(work_directory, library)
    heartbeat = Heartbeat()
    app = create_app(session, library, heartbeat, local_origins(port))
    server = make_server(HOST, port, app, threaded=True)

    def shut_down_when_abandoned() -> None:
        while True:
            time.sleep(WATCHDOG_INTERVAL_SECONDS)
            idle = heartbeat.seconds_since_last_beat() > IDLE_SHUTDOWN_SECONDS
            if idle and not session.is_busy:
                server.shutdown()
                return

    threading.Thread(target=shut_down_when_abandoned, daemon=True).start()
    open_in_browser(f"http://{HOST}:{port}")
    try:
        server.serve_forever()
    finally:
        shutil.rmtree(work_directory, ignore_errors=True)


def _running_instance_answers(port: int) -> bool:
    try:
        response = requests.get(f"http://{HOST}:{port}/api/ping", timeout=1)
        payload: object = response.json()
    except requests.RequestException, ValueError:
        return False
    return isinstance(payload, dict) and payload.get("app") == APP_IDENTIFIER


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((HOST, port))
        except OSError:
            return False
    return True


def _any_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((HOST, 0))
        port: int = probe.getsockname()[1]
    return port


if __name__ == "__main__":
    main()
