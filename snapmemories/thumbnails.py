import hashlib
import threading
from pathlib import Path

from PIL import Image, ImageOps

from snapmemories.library import Library, LibraryRecord
from snapmemories.models import MediaKind

THUMBNAIL_EDGE = 480
THUMBNAIL_QUALITY = 80

_generation_lock = threading.Lock()


def thumbnail_for(library: Library, record: LibraryRecord) -> Path | None:
    if record.kind != MediaKind.PHOTO:
        return None
    source = library.resolve(record)
    if source is None:
        return None
    name = hashlib.sha256(record.item_id.encode("utf-8")).hexdigest()[:32]
    target = library.thumbnail_directory / f"{name}.jpg"
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return target

    with _generation_lock:
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        try:
            with Image.open(source) as image:
                preview = ImageOps.exif_transpose(image).convert("RGB")
                preview.thumbnail((THUMBNAIL_EDGE, THUMBNAIL_EDGE), Image.Resampling.LANCZOS)
                preview.save(partial, format="JPEG", quality=THUMBNAIL_QUALITY)
            partial.replace(target)
        except OSError, ValueError, SyntaxError:
            partial.unlink(missing_ok=True)
            return None
    return target
