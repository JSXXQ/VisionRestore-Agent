from abc import ABC, abstractmethod
from pathlib import Path
from pydantic import BaseModel
from visionrestore.core.config import get_settings
from visionrestore.schemas.model import ModelStatus

class EnhancementResult(BaseModel):
    output_path: str
    parameters: dict
    runtime_ms: int
    peak_memory_mb: float = 0
    logs: list[str] = []

class ModelAdapter(ABC):
    model_id: str
    display_name: str
    description: str
    repository_url: str
    license_name: str
    supported_devices: list[str] = ["cpu", "cuda"]
    supported_precisions: list[str] = ["fp32"]
    weight_files: list[str] = []
    source_markers: list[str] = []

    def __init__(self):
        self.settings = get_settings()
        self.loaded = False

    @property
    def source_dir(self) -> Path:
        return self.settings.third_party_dir / self.model_id

    @property
    def weight_dir(self) -> Path:
        return self.settings.weights_dir / self.model_id

    def check_installation(self) -> tuple[bool, str]:
        missing_source = [m for m in self.source_markers if not (self.source_dir / m).exists()]
        missing_weights = [w for w in self.weight_files if not (self.weight_dir / w).exists()]
        if missing_source:
            return False, f"缺少官方源码文件: {', '.join(missing_source)}"
        if missing_weights:
            return False, f"缺少预训练权重: {', '.join(missing_weights)}"
        return True, "官方源码和预训练权重已检测到"

    def get_status(self) -> ModelStatus:
        installed, msg = self.check_installation()
        return ModelStatus(
            model_id=self.model_id,
            display_name=self.display_name,
            description=self.description,
            repository_url=self.repository_url,
            license_name=self.license_name,
            supported_devices=self.supported_devices,
            supported_precisions=self.supported_precisions,
            installed=installed,
            available=installed,
            loaded=self.loaded,
            weight_path=str(self.weight_dir) if self.weight_files else None,
            source_path=str(self.source_dir),
            status_message=msg,
            install_hint=self.install_hint(),
            capabilities=self.get_capabilities(),
        )

    def install_hint(self) -> str:
        return f"运行 scripts/download_models.ps1 -Model {self.model_id} 后，再按 docs/model_integration.md 放置权重。"

    def get_default_parameters(self) -> dict:
        return {"tile_size": 512, "tile_overlap": 32, "batch_size": 1}

    def validate_parameters(self, parameters: dict) -> dict:
        merged = self.get_default_parameters() | (parameters or {})
        merged["tile_size"] = max(128, int(merged.get("tile_size", 512)))
        merged["tile_overlap"] = max(0, min(int(merged.get("tile_overlap", 32)), merged["tile_size"] // 2))
        return merged

    def estimate_memory(self, width: int, height: int, precision: str = "fp32") -> float:
        bytes_per = 2 if precision == "fp16" else 4
        return float(width * height * 3 * bytes_per * 8 / 1024 / 1024)

    def health_check(self) -> dict:
        status = self.get_status()
        return {"model_id": self.model_id, "available": status.available, "message": status.status_message}

    def get_capabilities(self) -> dict:
        return {"rgb_static_image": True, "video": False, "event_camera": False, "tile_inference": True}

    def load(self, *args, **kwargs):
        installed, msg = self.check_installation()
        if not installed:
            raise RuntimeError(msg)
        self.loaded = True

    def unload(self):
        self.loaded = False

    @abstractmethod
    def enhance(self, image_path: str, output_path: str, device: str, precision: str, parameters: dict) -> EnhancementResult:
        raise NotImplementedError
