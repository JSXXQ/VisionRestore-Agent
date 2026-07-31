import time
from .base import ModelAdapter, EnhancementResult

class ZeroDCEAdapter(ModelAdapter):
    model_id = "zero_dce"
    display_name = "Zero-DCE"
    description = "轻量级无参考低照度增强模型，适合快速增强。"
    repository_url = "https://github.com/Li-Chongyi/Zero-DCE"
    license_name = "Author repository license required; see THIRD_PARTY_NOTICES.md"
    supported_devices = ["cpu", "cuda"]
    supported_precisions = ["fp32", "fp16"]
    source_markers = ["Zero-DCE_code/model.py"]
    weight_files = ["Epoch99.pth"]

    def enhance(self, image_path: str, output_path: str, device: str, precision: str, parameters: dict) -> EnhancementResult:
        self.load()
        start = time.perf_counter()
        # Real inference is intentionally delegated to the official source tree and weights.
        # This adapter fails honestly until the integration script has verified those assets.
        raise RuntimeError("Zero-DCE 官方源码/权重已检测，但自动封装推理尚需按 docs/model_integration.md 完成路径验证。不会使用 OpenCV/Gamma 伪造模型输出。")
