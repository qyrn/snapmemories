import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import requests

from snapmemories.models import RemoteMemory

USER_AGENT = "SnapMemories/2.0"
CONNECT_TIMEOUT_SECONDS = 15
READ_TIMEOUT_SECONDS = 60
CHUNK_BYTES = 256 * 1024
RETRY_DELAYS_SECONDS = (2, 5, 10)
EXPIRED_STATUSES = frozenset({401, 403, 404, 410})
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


class ExpiredLinkError(Exception):
    pass


class DownloadError(Exception):
    pass


def new_session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def download_memory(
    memory: RemoteMemory,
    target: Path,
    session: requests.Session,
    on_bytes: Callable[[int], None],
) -> None:
    for attempt, delay in enumerate((0, *RETRY_DELAYS_SECONDS)):
        time.sleep(delay)
        try:
            media_url = memory.media_download_url or _resolve_download_link(
                memory.download_link, session
            )
            _stream_to_file(media_url, target, session, on_bytes)
            return
        except requests.HTTPError as error:
            status = error.response.status_code if error.response is not None else 0
            if status in EXPIRED_STATUSES:
                raise ExpiredLinkError from error
            if status not in RETRYABLE_STATUSES or attempt == len(RETRY_DELAYS_SECONDS):
                raise DownloadError(f"HTTP {status}") from error
        except (requests.ConnectionError, requests.Timeout) as error:
            if attempt == len(RETRY_DELAYS_SECONDS):
                raise DownloadError("Network unreachable") from error


def _resolve_download_link(download_link: str, session: requests.Session) -> str:
    parts = urlsplit(download_link)
    endpoint = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    response = session.post(
        endpoint,
        data=parts.query,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
    )
    response.raise_for_status()
    media_url = response.text.strip()
    if not media_url.startswith("https://"):
        raise DownloadError("Snapchat returned an unexpected answer")
    return media_url


def _stream_to_file(
    url: str, target: Path, session: requests.Session, on_bytes: Callable[[int], None]
) -> None:
    with session.get(
        url, stream=True, timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS)
    ) as response:
        response.raise_for_status()
        with target.open("wb") as output:
            for chunk in response.iter_content(CHUNK_BYTES):
                output.write(chunk)
                on_bytes(len(chunk))
