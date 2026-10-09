from flask import Flask, Response, abort, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException

from snapmemories.desktop import open_folder
from snapmemories.heartbeat import Heartbeat
from snapmemories.library import Library, LibraryRecord
from snapmemories.security import install_security
from snapmemories.session import ImportSession, SessionError
from snapmemories.thumbnails import thumbnail_for

APP_IDENTIFIER = "snapmemories"


def create_app(
    session: ImportSession,
    library: Library,
    heartbeat: Heartbeat,
    allowed_origins: frozenset[str],
) -> Flask:
    app = Flask(__name__)
    app.json.sort_keys = False  # type: ignore[attr-defined]
    install_security(app, allowed_origins)

    @app.errorhandler(SessionError)
    def session_error(error: SessionError) -> tuple[Response, int]:
        return jsonify(error=str(error)), 400

    @app.errorhandler(HTTPException)
    def http_error(error: HTTPException) -> tuple[Response, int]:
        return jsonify(error=error.description), error.code or 500

    @app.get("/")
    def home() -> str:
        return render_template("index.html")

    @app.get("/viewer")
    def viewer() -> str:
        return render_template("viewer.html")

    @app.get("/api/ping")
    def ping() -> Response:
        return jsonify(app=APP_IDENTIFIER)

    @app.post("/api/heartbeat")
    def beat() -> Response:
        heartbeat.beat()
        return jsonify(ok=True)

    @app.put("/api/uploads")
    def upload() -> Response:
        name = request.args.get("name", "")
        files = session.add_upload(name, request.stream, request.content_length)
        return jsonify(files=files)

    @app.post("/api/analyze")
    def analyze() -> Response:
        return jsonify(session.analyze())

    @app.post("/api/import")
    def start_import() -> Response:
        session.start()
        return jsonify(ok=True)

    @app.post("/api/import/cancel")
    def cancel_import() -> Response:
        session.cancel()
        return jsonify(ok=True)

    @app.get("/api/import")
    def import_progress() -> Response:
        return jsonify(session.progress())

    @app.post("/api/reset")
    def reset() -> Response:
        session.reset()
        return jsonify(ok=True)

    @app.post("/api/open-folder")
    def open_output_folder() -> Response:
        open_folder(session.output_directory)
        return jsonify(ok=True)

    @app.get("/api/library")
    def library_items() -> Response:
        records = sorted(
            library.existing_records().values(),
            key=lambda record: record.taken_at,
            reverse=True,
        )
        return jsonify([_public_record(record) for record in records])

    @app.get("/api/library/<item_id>/thumbnail")
    def library_thumbnail(item_id: str) -> Response:
        thumbnail = thumbnail_for(library, _find_record(library, item_id))
        if thumbnail is None:
            abort(404)
        return send_file(thumbnail, mimetype="image/jpeg", max_age=86400)

    @app.get("/api/library/<item_id>/file")
    def library_file(item_id: str) -> Response:
        path = library.resolve(_find_record(library, item_id))
        if path is None:
            abort(404)
        return send_file(path, conditional=True)

    return app


def _find_record(library: Library, item_id: str) -> LibraryRecord:
    record = library.records().get(item_id)
    if record is None:
        abort(404)
    return record


def _public_record(record: LibraryRecord) -> dict[str, object]:
    return {
        "id": record.item_id,
        "kind": record.kind.value,
        "taken_at": record.taken_at,
        "latitude": record.latitude,
        "longitude": record.longitude,
    }
