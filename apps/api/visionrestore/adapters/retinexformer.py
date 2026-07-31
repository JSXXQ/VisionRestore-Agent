from .base import ModelAdapter, EnhancementResult

class RetinexformerAdapter(ModelAdapter):
    model_id = "retinexformer"
    display_name = "Retinexformer"
    description = "质量优先的低照度图像恢复模型。"
    repository_url = "https://github.com/caiyuanhao1998/Retinexformer"
    license_name = "Author repository license required; see THIRD_PARTY_NOTICES.md"
    supported_devices = ["cuda"]
    supported_precisions = ["fp32", "fp16"]
    source_markers = ["README.md"]
    weight_files = ["retinexformer.pth"]

    def enhance(self, image_path: str, output_path: str, device: str, precision: str, parameters: dict) -> EnhancementResult:
        self.load()
        raise RuntimeError("Retinexformer 适配器已保留，但本阶段未完成真实权重推理封装。")
