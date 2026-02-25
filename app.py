"""
SnapMemories — Save your Snapchat Memories locally.
All processing happens on your computer. No data ever leaves your machine.
"""

import os
import io
import json
import time
import base64
import shutil
import struct
import zipfile
import threading
import tempfile
import webbrowser
import traceback
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import piexif
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────

PORT = 7842
MAX_WORKERS = 10
MAX_RETRIES = 3
RETRY_BACKOFF = [2, 4, 8]   # seconds between retries
OUTPUT_BASE = Path.home() / "Memories"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024 * 1024  # 20 GB

# ─────────────────────────────────────────────
#  Global session state (one user, one session)
# ─────────────────────────────────────────────

session_state = {
    "step": 0,              # 0=idle, 1=parsed, 2=downloading, 3=done
    "entries": [],          # parsed memories
    "total_photos": 0,
    "total_videos": 0,
    "estimated_bytes": 0,
    "output_dir": str(OUTPUT_BASE),
    # download progress
    "downloaded": 0,
    "failed": 0,
    "total": 0,
    "speed_bps": 0,
    "elapsed": 0,
    "current_file": "",
    "errors": [],
    "is_running": False,
    "is_done": False,
    "cancel_flag": False,
    "start_time": None,
    # bytes downloaded for speed calculation
    "_bytes_downloaded": 0,
    "_bytes_lock": None,
    # actual bytes written to disk (computed after download completes)
    "used_bytes": 0,
}

_lock = threading.Lock()


def reset_state():
    global session_state
    with _lock:
        session_state.update({
            "step": 0,
            "entries": [],
            "total_photos": 0,
            "total_videos": 0,
            "estimated_bytes": 0,
            "output_dir": str(OUTPUT_BASE),
            "downloaded": 0,
            "failed": 0,
            "total": 0,
            "speed_bps": 0,
            "elapsed": 0,
            "current_file": "",
            "errors": [],
            "is_running": False,
            "is_done": False,
            "cancel_flag": False,
            "start_time": None,
            "_bytes_downloaded": 0,
            "_bytes_lock": threading.Lock(),
            "used_bytes": 0,
        })


reset_state()  # initialize lock properly


# ─────────────────────────────────────────────
#  Parsing helpers
# ─────────────────────────────────────────────

def parse_memories_json(data: bytes) -> list[dict]:
    """Parse memories_history.json and return a clean list of entries."""
    raw = json.loads(data.decode("utf-8"))

    # Locate the memories array — key name varies across Snapchat export versions
    entries_raw = None
    if isinstance(raw, list):
        entries_raw = raw
    elif isinstance(raw, dict):
        for key in ("Saved Media", "SavedMedia", "memories", "Memories", "saved_media"):
            if key in raw and isinstance(raw[key], list):
                entries_raw = raw[key]
                break
        if entries_raw is None:
            # Fallback: first list value found in the dict
            for v in raw.values():
                if isinstance(v, list):
                    entries_raw = v
                    break

    if not entries_raw:
        return []

    entries = []
    for item in entries_raw:
        if not isinstance(item, dict):
            continue

        date_str = item.get("Date", "")
        media_type = item.get("Media Type", "IMAGE").upper()
        download_link = item.get("Download Link", "")
        raw_location = item.get("Location", {})
        location = raw_location if isinstance(raw_location, dict) else {}

        lat = location.get("Latitude") or location.get("latitude")
        lng = location.get("Longitude") or location.get("longitude")

        # Normalise date → datetime
        dt = None
        for fmt in (
            "%Y-%m-%d %H:%M:%S %Z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
        ):
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                break
            except ValueError:
                continue

        if dt is None:
            dt = datetime(2000, 1, 1)

        if not download_link:
            continue

        entries.append({
            "date": dt,
            "media_type": media_type,   # IMAGE or VIDEO
            "download_link": download_link,
            "lat": float(lat) if lat is not None else None,
            "lng": float(lng) if lng is not None else None,
        })

    return entries


def estimate_size(entries: list[dict]) -> int:
    """Rough size estimate: 3 MB/photo, 15 MB/video."""
    photos = sum(1 for e in entries if e["media_type"] == "IMAGE")
    videos = sum(1 for e in entries if e["media_type"] == "VIDEO")
    return photos * 3_000_000 + videos * 15_000_000


# ─────────────────────────────────────────────
#  Two-step download
# ─────────────────────────────────────────────

def resolve_cdn_url(download_link: str, session: requests.Session) -> str:
    """
    Step 1: POST to the Snapchat endpoint to get the real CDN URL.
    The response body IS the URL (plain text or JSON depending on version).
    """
    resp = session.post(download_link, timeout=30)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "")
    if "json" in content_type:
        data = resp.json()
        # Try common key names
        for key in ("url", "download_url", "media_url", "Location"):
            if key in data:
                return data[key]
        raise ValueError(f"Cannot find URL in JSON response: {list(data.keys())}")

    # Plain text response
    url = resp.text.strip()
    if url.startswith("http"):
        return url

    raise ValueError(f"Unexpected CDN response: {url[:200]}")


def download_file(url: str, session: requests.Session) -> bytes:
    """Download a file and return its bytes. Updates global speed counter."""
    resp = session.get(url, timeout=60, stream=True)
    resp.raise_for_status()

    chunks = []
    for chunk in resp.iter_content(chunk_size=65536):
        if chunk:
            chunks.append(chunk)
            with session_state["_bytes_lock"]:
                session_state["_bytes_downloaded"] += len(chunk)

    return b"".join(chunks)


# ─────────────────────────────────────────────
#  File type detection
# ─────────────────────────────────────────────

JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC  = b"\x89PNG"
MP4_MAGIC  = b"ftyp"        # bytes 4-8
ZIP_MAGIC  = b"PK\x03\x04"


def detect_filetype(data: bytes) -> str:
    """Return 'jpg', 'mp4', 'zip', or 'unknown'."""
    if data[:3] == JPEG_MAGIC:
        return "jpg"
    if data[:4] == PNG_MAGIC:
        return "png"
    if data[4:8] == MP4_MAGIC:
        return "mp4"
    if data[:4] == ZIP_MAGIC:
        return "zip"
    return "jpg"  # fallback — treat as JPEG


# ─────────────────────────────────────────────
#  EXIF helpers
# ─────────────────────────────────────────────

def _deg_to_dms_rational(deg: float):
    """Convert decimal degrees to EXIF DMS rational tuples."""
    d = int(abs(deg))
    m_float = (abs(deg) - d) * 60
    m = int(m_float)
    s = (m_float - m) * 60
    # Store as (numerator, denominator)
    return (d, 1), (m, 1), (int(s * 100), 100)


def embed_exif_image(data: bytes, dt: datetime, lat=None, lng=None) -> bytes:
    """Embed EXIF DateTimeOriginal (and GPS if available) into JPEG bytes."""
    try:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}

        dt_str = dt.strftime("%Y:%m:%d %H:%M:%S")
        exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = dt_str.encode()
        exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = dt_str.encode()
        exif_dict["0th"][piexif.ImageIFD.DateTime] = dt_str.encode()

        if lat is not None and lng is not None:
            exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = b"N" if lat >= 0 else b"S"
            exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = _deg_to_dms_rational(lat)
            exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = b"E" if lng >= 0 else b"W"
            exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = _deg_to_dms_rational(lng)

        exif_bytes = piexif.dump(exif_dict)
        img = Image.open(io.BytesIO(data))
        out = io.BytesIO()
        img.save(out, format="JPEG", exif=exif_bytes)
        return out.getvalue()
    except Exception:
        return data  # if EXIF fails, return original data unchanged


def write_video_sidecar(path: Path, dt: datetime, lat=None, lng=None):
    """Write a JSON sidecar for video metadata into a _metadata/ subfolder."""
    meta_dir = path.parent / "_metadata"
    meta_dir.mkdir(exist_ok=True)
    sidecar = meta_dir / (path.name + ".json")
    meta = {
        "date_taken": dt.isoformat(),
        "latitude": lat,
        "longitude": lng,
    }
    sidecar.write_text(json.dumps(meta, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────
#  Overlay merging
# ─────────────────────────────────────────────

def merge_overlay(base_data: bytes, overlay_zip_data: bytes) -> bytes:
    """
    Extract the PNG overlay from a ZIP and composite it onto the base image.
    Returns the resulting JPEG bytes.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(overlay_zip_data)) as zf:
            png_names = [n for n in zf.namelist() if n.lower().endswith(".png")]
            if not png_names:
                return base_data  # no overlay found
            overlay_data = zf.read(png_names[0])

        base_img = Image.open(io.BytesIO(base_data)).convert("RGBA")
        overlay_img = Image.open(io.BytesIO(overlay_data)).convert("RGBA")

        # Resize overlay to match base if needed
        if overlay_img.size != base_img.size:
            overlay_img = overlay_img.resize(base_img.size, Image.LANCZOS)

        merged = Image.alpha_composite(base_img, overlay_img).convert("RGB")
        out = io.BytesIO()
        merged.save(out, format="JPEG", quality=95)
        return out.getvalue()
    except Exception:
        return base_data  # silently fall back to original


# ─────────────────────────────────────────────
#  French month names
# ─────────────────────────────────────────────

FRENCH_MONTHS = [
    "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
    "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre",
]


# ─────────────────────────────────────────────
#  Output path builder
# ─────────────────────────────────────────────

def build_output_path(output_dir: Path, dt: datetime, ext: str, index: int) -> Path:
    """
    Returns: output_dir/YYYY/Month YYYY/YYYY-MM-DD_HH-MM-SS[_N].ext
    Example: Memories/2025/Février 2025/2025-02-14_18-30-00.jpg
    """
    month_label = f"{FRENCH_MONTHS[dt.month - 1]} {dt.year}"
    folder = output_dir / str(dt.year) / month_label
    folder.mkdir(parents=True, exist_ok=True)
    stem = dt.strftime("%Y-%m-%d_%H-%M-%S")
    candidate = folder / f"{stem}.{ext}"
    if not candidate.exists():
        return candidate
    return folder / f"{stem}_{index:04d}.{ext}"


# ─────────────────────────────────────────────
#  Single entry download & process
# ─────────────────────────────────────────────

def process_entry(entry: dict, index: int, output_dir: Path, http: requests.Session) -> dict:
    """
    Download + process one memory entry.
    Returns {"ok": bool, "path": str, "error": str}.
    """
    dt = entry["date"]
    lat = entry["lat"]
    lng = entry["lng"]
    media_type = entry["media_type"]

    for attempt in range(MAX_RETRIES):
        try:
            with _lock:
                session_state["current_file"] = (
                    f"{dt.strftime('%Y-%m-%d')} ({media_type.lower()})"
                )

            # ── Step 1: resolve real CDN URL ──
            cdn_url = resolve_cdn_url(entry["download_link"], http)

            # ── Step 2: detect if the resolved URL itself is an overlay ZIP
            #    (some endpoints return overlay links directly)
            is_overlay = cdn_url.lower().endswith(".zip")

            # ── Step 3: download the media bytes ──
            media_data = download_file(cdn_url, http)

            # ── Step 4: detect actual file type ──
            ftype = detect_filetype(media_data)

            if ftype == "zip":
                # This is an overlay ZIP — we need the base image
                # The entry's download_link usually has a companion base image
                # stored as the same URL without overlay suffix.
                # Strategy: extract base image from ZIP if present, else skip overlay.
                overlay_zip = media_data
                # Try to find a base image sibling URL (Snapchat convention)
                base_url = cdn_url.replace("_overlay", "").replace("overlay_", "")
                if base_url != cdn_url:
                    try:
                        base_data = download_file(base_url, http)
                        if detect_filetype(base_data) in ("jpg", "png"):
                            media_data = merge_overlay(base_data, overlay_zip)
                            ftype = "jpg"
                    except Exception:
                        pass

                if ftype == "zip":
                    # Could not merge — save the ZIP as-is or skip
                    # We'll save the raw file and note it
                    out_path = build_output_path(output_dir, dt, "zip", index)
                    out_path.write_bytes(media_data)
                    return {"ok": True, "path": str(out_path), "error": ""}

            # ── Step 5: embed metadata ──
            if ftype in ("jpg", "png"):
                if ftype == "png":
                    # Convert PNG to JPEG for EXIF support
                    img = Image.open(io.BytesIO(media_data)).convert("RGB")
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=95)
                    media_data = buf.getvalue()
                    ftype = "jpg"
                media_data = embed_exif_image(media_data, dt, lat, lng)
            elif ftype == "mp4":
                # EXIF not supported in MP4 — sidecar written after save
                pass

            # ── Step 6: write to disk ──
            out_path = build_output_path(output_dir, dt, ftype, index)
            out_path.write_bytes(media_data)

            if ftype == "mp4":
                write_video_sidecar(out_path, dt, lat, lng)

            return {"ok": True, "path": str(out_path), "error": ""}

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status in (403, 410) and attempt == 0:
                # Expired link — no point retrying
                return {
                    "ok": False,
                    "path": "",
                    "error": f"EXPIRED:{dt.strftime('%Y-%m-%d')}",
                }
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF[attempt])
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF[attempt])
            else:
                return {
                    "ok": False,
                    "path": "",
                    "error": f"{dt.strftime('%Y-%m-%d')}: {str(e)[:120]}",
                }

    return {"ok": False, "path": "", "error": f"{dt.strftime('%Y-%m-%d')}: max retries exceeded"}


# ─────────────────────────────────────────────
#  Main download orchestrator (runs in thread)
# ─────────────────────────────────────────────

def run_download(entries: list[dict], output_dir: Path, workers: int = MAX_WORKERS):
    """Background thread: download and process all entries."""
    with _lock:
        session_state["is_running"] = True
        session_state["total"] = len(entries)
        session_state["downloaded"] = 0
        session_state["failed"] = 0
        session_state["errors"] = []
        session_state["start_time"] = time.monotonic()
        session_state["_bytes_downloaded"] = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    http = requests.Session()
    http.headers.update({"User-Agent": "SnapMemories/1.0"})

    # Speed calculation thread
    def speed_tracker():
        prev_bytes = 0
        while session_state["is_running"]:
            time.sleep(2)
            curr = session_state["_bytes_downloaded"]
            bps = (curr - prev_bytes) / 2
            prev_bytes = curr
            with _lock:
                session_state["speed_bps"] = bps
                if session_state["start_time"]:
                    session_state["elapsed"] = time.monotonic() - session_state["start_time"]

    speed_thread = threading.Thread(target=speed_tracker, daemon=True)
    speed_thread.start()

    expired_count = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_entry, entry, i, output_dir, http): i
            for i, entry in enumerate(entries)
        }

        for future in as_completed(futures):
            if session_state["cancel_flag"]:
                pool.shutdown(wait=False, cancel_futures=True)
                break

            result = future.result()
            with _lock:
                if result["ok"]:
                    session_state["downloaded"] += 1
                else:
                    session_state["failed"] += 1
                    err = result["error"]
                    if err.startswith("EXPIRED:"):
                        expired_count += 1
                    else:
                        session_state["errors"].append(err)

    # Calculate actual disk usage (exclude _metadata sidecars)
    try:
        used_bytes = sum(
            f.stat().st_size for f in output_dir.rglob("*")
            if f.is_file() and "_metadata" not in f.parts
        )
    except Exception:
        used_bytes = 0

    with _lock:
        session_state["is_running"] = False
        session_state["is_done"] = True
        session_state["step"] = 3
        session_state["used_bytes"] = used_bytes
        if expired_count > 0:
            session_state["errors"].insert(
                0,
                f"EXPIRED_LINKS:{expired_count} link(s) have expired — please request a new Snapchat export.",
            )


# ─────────────────────────────────────────────
#  Disk space check
# ─────────────────────────────────────────────

def check_disk_space(output_dir: Path, estimated_bytes: int) -> dict:
    """Returns {"ok": bool, "free_bytes": int, "needed_bytes": int}."""
    try:
        stat = shutil.disk_usage(output_dir.parent if not output_dir.exists() else output_dir)
        needed = estimated_bytes * 2  # 2× safety margin
        return {
            "ok": stat.free >= needed,
            "free_bytes": stat.free,
            "needed_bytes": needed,
        }
    except Exception:
        return {"ok": True, "free_bytes": -1, "needed_bytes": 0}


# ─────────────────────────────────────────────
#  Media metadata readers (for the viewer)
# ─────────────────────────────────────────────

def _dms_to_decimal(dms_rational) -> float:
    """Convert EXIF DMS rational tuples to decimal degrees."""
    d = dms_rational[0][0] / dms_rational[0][1]
    m = dms_rational[1][0] / dms_rational[1][1]
    s = dms_rational[2][0] / dms_rational[2][1]
    return d + m / 60.0 + s / 3600.0


def read_jpg_metadata(path: Path) -> dict:
    """Read EXIF date and GPS coordinates from a JPEG file."""
    try:
        exif_data = piexif.load(str(path))

        dt = None
        date_bytes = exif_data.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal)
        if date_bytes:
            dt = datetime.strptime(date_bytes.decode("utf-8"), "%Y:%m:%d %H:%M:%S")

        lat = lng = None
        gps = exif_data.get("GPS", {})
        if piexif.GPSIFD.GPSLatitude in gps and piexif.GPSIFD.GPSLongitude in gps:
            lat = _dms_to_decimal(gps[piexif.GPSIFD.GPSLatitude])
            lng = _dms_to_decimal(gps[piexif.GPSIFD.GPSLongitude])
            if gps.get(piexif.GPSIFD.GPSLatitudeRef) in (b"S", "S"):
                lat = -lat
            if gps.get(piexif.GPSIFD.GPSLongitudeRef) in (b"W", "W"):
                lng = -lng

        return {"date": dt, "lat": lat, "lng": lng}
    except Exception:
        return {"date": None, "lat": None, "lng": None}


def read_mp4_metadata(path: Path) -> dict:
    """Read metadata from the _metadata/ sidecar JSON file."""
    sidecar = path.parent / "_metadata" / (path.name + ".json")
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        dt = None
        if meta.get("date_taken"):
            dt = datetime.fromisoformat(meta["date_taken"])
        return {
            "date": dt,
            "lat": meta.get("latitude"),
            "lng": meta.get("longitude"),
        }
    except Exception:
        return {"date": None, "lat": None, "lng": None}


# ─────────────────────────────────────────────
#  Thumbnail generator (for /api/media)
# ─────────────────────────────────────────────

def make_thumbnail(path: Path) -> str | None:
    """Generate a base64-encoded JPEG thumbnail (max 300×300) for a JPEG file."""
    try:
        img = Image.open(path)
        img.thumbnail((300, 300), Image.LANCZOS)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=75)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


# ─────────────────────────────────────────────
#  Flask routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload-zip", methods=["POST"])
def upload_zip():
    """Receive the Snapchat export ZIP and parse memories_history.json."""
    if "file" not in request.files:
        return jsonify({"error": "No file received."}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".zip"):
        return jsonify({"error": "The file must be a .zip"}), 400

    try:
        zip_bytes = file.read()

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            # Find memories_history.json anywhere in the ZIP
            json_names = [
                n for n in zf.namelist()
                if n.endswith("memories_history.json")
            ]
            if not json_names:
                return jsonify({
                    "error": (
                        "File memories_history.json not found in the ZIP. "
                        "Make sure you exported your Snapchat data correctly "
                        "via accounts.snapchat.com."
                    )
                }), 400

            json_data = zf.read(json_names[0])

        entries = parse_memories_json(json_data)
        if not entries:
            return jsonify({"error": "No memories found in the file."}), 400

        photos = sum(1 for e in entries if e["media_type"] == "IMAGE")
        videos = sum(1 for e in entries if e["media_type"] == "VIDEO")
        estimated = estimate_size(entries)

        # Store in session state (convert datetime → ISO string for JSON)
        serialisable = []
        for e in entries:
            serialisable.append({
                **e,
                "date": e["date"].isoformat(),
            })

        with _lock:
            session_state["entries"] = entries
            session_state["total_photos"] = photos
            session_state["total_videos"] = videos
            session_state["estimated_bytes"] = estimated
            session_state["step"] = 1

        disk = check_disk_space(OUTPUT_BASE, estimated)

        return jsonify({
            "photos": photos,
            "videos": videos,
            "total": len(entries),
            "estimated_bytes": estimated,
            "disk": disk,
            "output_dir": str(OUTPUT_BASE),
        })

    except zipfile.BadZipFile:
        return jsonify({"error": "The ZIP file appears to be corrupted."}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@app.route("/api/start-download", methods=["POST"])
def start_download():
    """Launch the download process in a background thread."""
    with _lock:
        if session_state["is_running"]:
            return jsonify({"error": "Download already in progress."}), 400
        if not session_state["entries"]:
            return jsonify({"error": "No entries to download."}), 400

        entries = session_state["entries"]
        output_dir = Path(session_state["output_dir"])
        session_state["step"] = 2
        session_state["cancel_flag"] = False

    workers = int(request.json.get("workers", MAX_WORKERS)) if request.is_json else MAX_WORKERS
    workers = max(1, min(workers, 20))

    thread = threading.Thread(
        target=run_download,
        args=(entries, output_dir, workers),
        daemon=True,
    )
    thread.start()

    return jsonify({"ok": True})


@app.route("/api/progress")
def progress():
    """Polling endpoint — return current download state."""
    with _lock:
        total = session_state["total"]
        done = session_state["downloaded"] + session_state["failed"]
        elapsed = session_state["elapsed"] or 0

        # ETA
        if done > 0 and elapsed > 0 and total > done:
            rate = done / elapsed
            eta = (total - done) / rate if rate > 0 else 0
        else:
            eta = 0

        has_expired = any(
            e.startswith("EXPIRED_LINKS:") for e in session_state["errors"]
        )

        return jsonify({
            "step": session_state["step"],
            "downloaded": session_state["downloaded"],
            "failed": session_state["failed"],
            "total": total,
            "is_running": session_state["is_running"],
            "is_done": session_state["is_done"],
            "speed_bps": round(session_state["speed_bps"]),
            "elapsed": round(elapsed),
            "eta": round(eta),
            "current_file": session_state["current_file"],
            "errors": session_state["errors"][:50],  # cap for JSON size
            "has_expired_links": has_expired,
            "output_dir": session_state["output_dir"],
            "used_bytes": session_state["used_bytes"],
        })


@app.route("/api/cancel", methods=["POST"])
def cancel():
    with _lock:
        session_state["cancel_flag"] = True
    return jsonify({"ok": True})


@app.route("/api/open-folder", methods=["POST"])
def open_folder():
    """Open the output folder in Windows Explorer."""
    folder = session_state["output_dir"]
    try:
        os.startfile(folder)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/reset", methods=["POST"])
def reset():
    reset_state()
    return jsonify({"ok": True})


@app.route("/viewer")
def viewer():
    return render_template("viewer.html")


@app.route("/api/media")
def api_media():
    """Scan the Memories folder and return a JSON list of all media with metadata."""
    output_dir = Path(session_state["output_dir"])
    if not output_dir.exists():
        return jsonify([])

    media_list = []
    for path in sorted(output_dir.rglob("*")):
        # Skip _metadata folders and non-files
        if not path.is_file():
            continue
        if "_metadata" in path.parts:
            continue

        suffix = path.suffix.lower()
        rel = str(path.relative_to(output_dir)).replace("\\", "/")

        if suffix == ".jpg":
            meta = read_jpg_metadata(path)
            media_list.append({
                "path": rel,
                "type": "photo",
                "date": meta["date"].isoformat() if meta["date"] else None,
                "lat": meta["lat"],
                "lng": meta["lng"],
                "filename": path.name,
                "thumb": make_thumbnail(path),
            })
        elif suffix == ".mp4":
            meta = read_mp4_metadata(path)
            media_list.append({
                "path": rel,
                "type": "video",
                "date": meta["date"].isoformat() if meta["date"] else None,
                "lat": meta["lat"],
                "lng": meta["lng"],
                "filename": path.name,
                "thumb": None,
            })

    # Newest first
    media_list.sort(key=lambda x: x["date"] or "0000", reverse=True)
    return jsonify(media_list)


@app.route("/media/<path:filepath>")
def serve_media(filepath):
    """Serve a media file from the Memories output folder."""
    output_dir = Path(session_state["output_dir"]).resolve()
    try:
        safe_path = Path(filepath.replace("/", os.sep))
        full_path = (output_dir / safe_path).resolve()
        full_path.relative_to(output_dir)  # raises ValueError if path escapes
    except (ValueError, Exception):
        return "", 403
    if not full_path.is_file():
        return "", 404
    return send_file(full_path)


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

def open_browser():
    """Wait for Flask to start, then open the browser."""
    time.sleep(1.2)
    webbrowser.open(f"http://localhost:{PORT}")


if __name__ == "__main__":
    t = threading.Thread(target=open_browser, daemon=True)
    t.start()
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
