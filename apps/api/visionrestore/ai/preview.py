from pathlib import Path

from PIL import Image

from visionrestore.core.config import get_settings


def create_ai_preview(image_path: str | Path) -> tuple[bytes, dict]:
    settings = get_settings()
    max_edge = max(768, min(int(settings.multimodal_image_max_edge), 1280))
    quality = max(50, min(int(settings.multimodal_image_quality), 95))
    with Image.open(image_path) as im:
        rgb = im.convert("RGB")
        original_size = rgb.size
        width, height = rgb.size
        scale = min(1.0, max_edge / max(width, height))
        if scale < 1.0:
            rgb = rgb.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
        clean = Image.new("RGB", rgb.size)
        clean.putdata(list(rgb.getdata()))
        import io
        bio = io.BytesIO()
        clean.save(bio, format="JPEG", quality=quality, optimize=True)
    return bio.getvalue(), {
        "format": "jpeg",
        "quality": quality,
        "max_edge": max_edge,
        "original_size": original_size,
        "preview_size": clean.size,
        "metadata_removed": bool(settings.multimodal_remove_metadata),
    }
