from pathlib import Path

from flask.testing import FlaskClient

from snapmemories.heartbeat import Heartbeat
from snapmemories.library import Library
from snapmemories.security import local_origins
from snapmemories.server import create_app
from snapmemories.session import ImportSession

PORT = 7842
ORIGIN = f"http://127.0.0.1:{PORT}"


def client(tmp_path: Path) -> FlaskClient:
    library = Library(tmp_path / "Memories")
    app = create_app(
        ImportSession(tmp_path / "work", library), library, Heartbeat(), local_origins(PORT)
    )
    return app.test_client()


def test_rejects_foreign_host_headers(tmp_path: Path) -> None:
    response = client(tmp_path).get("/api/library", headers={"Host": "evil.test"})

    assert response.status_code == 403


def test_rejects_cross_site_writes(tmp_path: Path) -> None:
    response = client(tmp_path).post(
        "/api/reset", headers={"Host": f"127.0.0.1:{PORT}", "Origin": "https://evil.test"}
    )

    assert response.status_code == 403


def test_accepts_same_origin_writes_and_sets_csp(tmp_path: Path) -> None:
    response = client(tmp_path).post(
        "/api/reset", headers={"Host": f"127.0.0.1:{PORT}", "Origin": ORIGIN}
    )

    assert response.status_code == 200
    assert "script-src 'self'" in response.headers["Content-Security-Policy"]


def test_rejects_uploads_that_are_not_archives(tmp_path: Path) -> None:
    response = client(tmp_path).put(
        "/api/uploads?name=evil.exe",
        data=b"MZ",
        headers={"Host": f"127.0.0.1:{PORT}", "Origin": ORIGIN},
    )

    assert response.status_code == 400
    assert "not a .zip" in response.get_json()["error"]


def test_library_files_are_served_by_id_only(tmp_path: Path) -> None:
    response = client(tmp_path).get(
        "/api/library/..%2F..%2Fsecret/file", headers={"Host": f"127.0.0.1:{PORT}"}
    )

    assert response.status_code == 404
