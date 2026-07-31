import time
from .base import ModelAdapter, EnhancementResult

class SCIAdapter(ModelAdapter):
    model_id = "sci"
    display_name = "SCI"
    description = "Self-Calibrated Illumination，低算力或速度优先场景。"
    repository_url = "https://github.com/vis-opt-group/SCI"
    license_name = "Author repository license required; see THIRD_PARTY_NOTICES.md"
    supported_devices = ["cpu", "cuda"]
    supported_precisions = ["fp32"]
    source_markers = ["model.py"]
    weight_files = ["medium.pt"]

    def enhance(self, image_path: str, output_path: str, device: str, precision: str, parameters: dict) -> EnhancementResult:
        self.load()
        raise RuntimeError("SCI 官方源码/权重已检测，但自动封装推理尚需按 docs/model_integration.md 完成路径验证。不会使用传统增强伪造模型输出。")
