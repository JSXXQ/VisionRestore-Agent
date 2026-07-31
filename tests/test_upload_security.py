import pytest
from PIL import Image
import io
from visionrestore.utils.file_security import UploadValidationError, safe_image_upload

def png_bytes():
    bio = io.BytesIO()
    Image.new("RGB", (16, 16), (10, 10, 10)).save(bio, format="PNG")
    return bio.getvalue()

def test_upload_rejects_extension():
    with pytest.raises(UploadValidationError):
        safe_image_upload("../bad.exe", b"not-image", "application/octet-stream")

def test_upload_accepts_png():
    rec = safe_image_upload("x.png", png_bytes(), "image/png")
    assert rec.width == 16
    assert "/" in rec.relative_path
