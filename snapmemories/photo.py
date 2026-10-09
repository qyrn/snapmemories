import io
from datetime import datetime

from PIL import ExifTags, Image, ImageOps

from snapmemories.jpeg_exif import with_capture_metadata
from snapmemories.media_format import MediaFormat, sniff_format
from snapmemories.models import GeoPoint

JPEG_QUALITY = 95
NORMAL_ORIENTATION = 1


def render_photo(
    main: bytes, overlay: bytes | None, taken_at: datetime, location: GeoPoint | None
) -> bytes:
    if overlay is not None:
        jpeg = compose_overlay(main, overlay)
    elif sniff_format(main[:16]) == MediaFormat.JPEG:
        jpeg = main
    else:
        jpeg = convert_to_jpeg(main)
    return with_capture_metadata(jpeg, taken_at, location)


def compose_overlay(main: bytes, overlay: bytes) -> bytes:
    with Image.open(io.BytesIO(main)) as source, Image.open(io.BytesIO(overlay)) as layer:
        icc_profile = source.info.get("icc_profile")
        exif = source.getexif()
        base = ImageOps.exif_transpose(source).convert("RGBA")
        sticker = layer.convert("RGBA")
        if sticker.size != base.size:
            sticker = sticker.resize(base.size, Image.Resampling.LANCZOS)
        merged = Image.alpha_composite(base, sticker).convert("RGB")
    if ExifTags.Base.Orientation in exif:
        exif[ExifTags.Base.Orientation] = NORMAL_ORIENTATION
    return _encode_jpeg(merged, exif, icc_profile)


def convert_to_jpeg(data: bytes) -> bytes:
    with Image.open(io.BytesIO(data)) as source:
        icc_profile = source.info.get("icc_profile")
        exif = source.getexif()
        image = ImageOps.exif_transpose(source)
        if image.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", image.size, (0, 0, 0))
            background.paste(image.convert("RGBA"), mask=image.convert("RGBA").getchannel("A"))
            image = background
        else:
            image = image.convert("RGB")
    if ExifTags.Base.Orientation in exif:
        exif[ExifTags.Base.Orientation] = NORMAL_ORIENTATION
    return _encode_jpeg(image, exif, icc_profile)


def _encode_jpeg(image: Image.Image, exif: Image.Exif, icc_profile: object) -> bytes:
    output = io.BytesIO()
    options: dict[str, object] = {"quality": JPEG_QUALITY, "subsampling": 0, "exif": exif}
    if isinstance(icc_profile, bytes):
        options["icc_profile"] = icc_profile
    image.save(output, format="JPEG", **options)
    return output.getvalue()
