import numpy as np
from PIL import Image
from visionrestore.services.image_analyzer import ImageAnalyzer

def test_low_light_analysis(tmp_path):
    path = tmp_path / "low.png"
    Image.fromarray(np.full((64, 64, 3), 20, dtype=np.uint8)).save(path)
    result = ImageAnalyzer().analyze(str(path))
    assert result.width == 64
    assert result.dark_pixel_ratio > 0.9
    assert result.mean_luminance < 30
