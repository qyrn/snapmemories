from flask import Flask, Response, abort, request

SAFE_METHODS = frozenset({"GET", "HEAD"})
CONTENT_SECURITY_POLICY = "; ".join(
    (
        "default-src 'none'",
        "script-src 'self'",
        "style-src 'self'",
        "img-src 'self'",
        "media-src 'self'",
        "font-src 'self'",
        "connect-src 'self'",
        "base-uri 'none'",
        "form-action 'none'",
        "frame-ancestors 'none'",
    )
)


def local_origins(port: int) -> frozenset[str]:
    return frozenset({f"http://127.0.0.1:{port}", f"http://localhost:{port}"})


def install_security(app: Flask, allowed_origins: frozenset[str]) -> None:
    allowed_hosts = frozenset(origin.removeprefix("http://") for origin in allowed_origins)

    @app.before_request
    def reject_foreign_requests() -> None:
        if request.host not in allowed_hosts:
            abort(403)
        if request.method in SAFE_METHODS:
            return
        if request.headers.get("Origin") not in allowed_origins:
            abort(403)

    @app.after_request
    def add_security_headers(response: Response) -> Response:
        response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers["X-Frame-Options"] = "DENY"
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response
