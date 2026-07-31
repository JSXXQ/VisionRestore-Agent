import time

from PIL import Image, ImageEnhance

from visionrestore.core.model_config import file_sha256

from .base import EnhancementResult, ModelAdapter


class MockModelAdapter(ModelAdapter):
    model_id = "mock_model"
    display_name = "MockModelAdapter - tests only"
    description = "Only for automated tests. It is never enabled in formal mode unless ALLOW_MOCK_MODEL is true."
    repository_url = "local-tests"
    license_name = "internal test helper"
    supported_devices = ["cpu"]
    supported_precisions = ["fp32"]
    default_checkpoint = "mock_checkpoint"

    def check_installation(self):
        return True, "Mock is enabled explicitly for tests."

    def discover_weights(self) -> list[dict]:
        return [{
            "checkpoint_id": self.default_checkpoint,
            "display_name": "Mock checkpoint",
            "path": None,
            "file_name": None,
            "size_bytes": None,
            "sha256": None,
            "config_path": None,
            "config_exists": True,
            "license_name": self.license_name,
            "domain": "test-only",
            "auto_route": False,
            "default": True,
            "exists": True,
            "status": "found",
            "health_check": "not_run",
            "last_error": None,
        }]

    def enhance(self, image_path: str, output_path: str, device: str, precision: str, parameters: dict) -> EnhancementResult:
        start = time.perf_counter()
        params = {"brightness": 1.6, "contrast": 1.1, **(parameters or {})}
        with Image.open(image_path) as im:
            out = ImageEnhance.Brightness(im.convert("RGB")).enhance(float(params["brightness"]))
            out = ImageEnhance.Contrast(out).enhance(float(params["contrast"]))
            out.save(output_path)
        return EnhancementResult(
            output_path=output_path,
            parameters={"mock": True, **params},
            runtime_ms=int((time.perf_counter() - start) * 1000),
            peak_memory_mb=0,
            logs=["MockModelAdapter produced a labeled test-only output."],
            is_mock=True,
            adapter_class=self.__class__.__name__,
            model_id=self.model_id,
            checkpoint_id=self.default_checkpoint,
            checkpoint_path=None,
            checkpoint_sha256=None,
            device="cpu",
            precision="fp32",
            input_sha256=file_sha256(image_path),
            output_sha256=file_sha256(output_path),
        )
