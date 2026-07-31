import time
from PIL import Image, ImageEnhance
from .base import ModelAdapter, EnhancementResult

class MockModelAdapter(ModelAdapter):
    model_id = "mock_model"
    display_name = "MockModelAdapter - tests only"
    description = "仅用于自动化测试，不会在正式运行中自动启用。"
    repository_url = "local-tests"
    license_name = "internal test helper"
    supported_devices = ["cpu"]
    supported_precisions = ["fp32"]

    def check_installation(self):
        return True, "Mock 仅用于测试"

    def enhance(self, image_path: str, output_path: str, device: str, precision: str, parameters: dict) -> EnhancementResult:
        start = time.perf_counter()
        with Image.open(image_path) as im:
            out = ImageEnhance.Brightness(im.convert("RGB")).enhance(float(parameters.get("brightness", 1.6)))
            out = ImageEnhance.Contrast(out).enhance(float(parameters.get("contrast", 1.1)))
            out.save(output_path)
        return EnhancementResult(
            output_path=output_path,
            parameters={"mock": True, **parameters},
            runtime_ms=int((time.perf_counter() - start) * 1000),
            logs=["MockModelAdapter produced a labeled test-only output."],
        )
