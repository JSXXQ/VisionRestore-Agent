from .base import ModelAdapter, EnhancementResult

class SNRAwareAdapter(ModelAdapter):
    model_id = "snr_aware"
    display_name = "SNR-Aware Low-Light Enhancement"
    description = "针对极暗且强噪声图像的恢复模型。"
    repository_url = "https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance"
    license_name = "Author repository license required; see THIRD_PARTY_NOTICES.md"
    supported_devices = ["cuda"]
    supported_precisions = ["fp32", "fp16"]
    source_markers = ["README.md"]
    weight_files = ["snr_aware.pth"]

    def enhance(self, image_path: str, output_path: str, device: str, precision: str, parameters: dict) -> EnhancementResult:
        self.load()
        raise RuntimeError("SNR-Aware 适配器已保留，但本阶段未完成真实权重推理封装。")
