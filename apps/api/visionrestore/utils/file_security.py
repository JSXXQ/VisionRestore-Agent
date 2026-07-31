import hashlib
import mimetypes
from pathlib import Path
from uuid import uuid4
from PIL import Image, ImageFile, UnidentifiedImageError
from visionrestore.core.config import get_settings
from visionrestore.schemas.common import now_iso
from visionrestore.schemas.image import FileRecord

Image.MAX_IMAGE_PIXELS = get_settings().max_image_pixels
ImageFile.LOAD_TRUNCATED_IMAGES = False

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ALLOWED_MIME_PREFIXES = ("image/png", "image/jpeg", "image/bmp", "image/tiff")

class UploadValidationError(ValueError):
    pass

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def safe_image_upload(original_name: str, data: bytes, content_type: str | None) -> FileRecord:
    settings = get_settings()
    suffix = Path(original_name or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise UploadValidationError(f"不支持的文件扩展名: {suffix or '(empty)'}")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise UploadValidationError(f"文件超过 {settings.max_upload_mb} MB")
    guessed = mimetypes.guess_type(f"x{suffix}")[0] or ""
    mime = content_type or guessed
    if mime and not any(mime.startswith(prefix) for prefix in ALLOWED_MIME_PREFIXES):
        raise UploadValidationError(f"不支持的 MIME 类型: {mime}")
    try:
        with Image.open(__import__("io").BytesIO(data)) as im:
            im.verify()
        with Image.open(__import__("io").BytesIO(data)) as im:
            width, height = im.size
            if width <= 0 or height <= 0:
                raise UploadValidationError("图像尺寸非法")
            if width * height > settings.max_image_pixels:
                raise UploadValidationError("图像像素数超过限制")
            im.convert("RGB")
    except UnidentifiedImageError as exc:
        raise UploadValidationError("文件无法作为图像解码") from exc
    file_id = str(uuid4())
    filename = f"{file_id}{suffix}"
    rel = f"uploads/{filename}"
    out = settings.data_dir / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return FileRecord(
        file_id=file_id,
        relative_path=rel.replace("\\", "/"),
        sha256=sha256_bytes(data),
        mime_type=mime or guessed,
        width=width,
        height=height,
        size_bytes=len(data),
        uploaded_at=now_iso(),
    )

def resolve_registered_path(relative_path: str) -> Path:
    settings = get_settings()
    root = settings.data_dir.resolve()
    path = (settings.data_dir / relative_path).resolve()
    if root not in path.parents and path != root:
        raise ValueError("非法文件路径")
    return path
