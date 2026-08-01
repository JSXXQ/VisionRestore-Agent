from pathlib import Path

from visionrestore.adapters.base import EnhancementResult, ModelAdapter
from visionrestore.backends.subprocess_backend import SubprocessBackend, SubprocessModelRuntime
from visionrestore.core.model_config import file_sha256, file_size, get_model_runtime_config
from visionrestore.schemas.model import ModelStatus
from visionrestore.schemas.worker import WorkerRequest


WORKER_MODEL_SPECS = {
    "darkir": {
        "display_name": "DarkIR",
        "description": "低照度、噪声和模糊联合恢复专家。",
        "task_type": "enhancement",
        "license_name": "Unknown",
        "supports_tiling": True,
    },
    "hvi_cidnet": {
        "display_name": "HVI-CIDNet",
        "description": "颜色和亮度恢复专家。",
        "task_type": "enhancement",
        "license_name": "Unknown",
        "supports_tiling": False,
    },
    "flol": {
        "display_name": "FLOL",
        "description": "快速真实低照度增强专家。",
        "task_type": "enhancement",
        "license_name": "Unknown",
        "supports_tiling": False,
    },
    "lpdm": {
        "display_name": "LPDM",
        "description": "低照度增强后处理去噪器。",
        "task_type": "denoising",
        "license_name": "Unknown",
        "supports_tiling": False,
    },
    "mambair": {
        "display_name": "MambaIR",
        "description": "真实图像去噪和超分后处理工具。",
        "task_type": "denoising,super_resolution",
        "license_name": "Unknown",
        "supports_tiling": True,
    },
}


class WorkerModelAdapter(ModelAdapter):
    repository_url = ""
    supported_devices = ["cuda", "cpu"]
    supported_precisions = ["fp32", "fp16"]
    default_checkpoint = ""

    def __init__(self, model_id: str):
        self.model_id = model_id
        spec = WORKER_MODEL_SPECS[model_id]
        self.display_name = spec["display_name"]
        self.description = spec["description"]
        self.license_name = spec["license_name"]
        self.task_type = spec["task_type"]
        self.supports_tiling = bool(spec["supports_tiling"])
        super().__init__()
        self.runtime_config = get_model_runtime_config(model_id)

    @property
    def source_path(self) -> str:
        return self.runtime_config.source_path

    def discover_weights(self) -> list[dict]:
        weights = []
        for profile in self.runtime_config.weight_profiles:
            path = profile.path
            config_path = profile.config_path
            exists = bool(path and Path(path).exists())
            config_exists = True if not config_path else Path(config_path).exists()
            source_exists = bool(self.source_path and Path(self.source_path).exists())
            if not source_exists:
                status = "source_missing"
            elif not exists:
                status = "weight_missing"
            elif not config_exists:
                status = "config_missing"
            else:
                status = "found"
            weights.append({
                "checkpoint_id": profile.checkpoint_id,
                "display_name": profile.display_name or profile.checkpoint_id,
                "path": path,
                "file_name": Path(path).name if path else None,
                "size_bytes": file_size(path) if path else None,
                "sha256": file_sha256(path) if path and exists else None,
                "config_path": config_path,
                "config_exists": config_exists,
                "domain": profile.domain,
                "auto_route": profile.auto_route,
                "default": profile.default,
                "exists": exists,
                "status": status,
                "health_check": "not_run",
                "last_error": None,
                "metadata": profile.metadata,
            })
        return weights

    def _runtime_health(self) -> dict:
        runtime = SubprocessModelRuntime(
            model_id=self.model_id,
            source_path=self.runtime_config.source_path,
            python_executable=self.runtime_config.python_executable,
            worker_script=self.runtime_config.worker_script,
            environment_name=self.runtime_config.environment_name,
            timeout_seconds=self.runtime_config.timeout_seconds,
        )
        return SubprocessBackend(runtime).health_check()

    def check_installation(self) -> tuple[bool, str]:
        runtime_health = self._runtime_health()
        if not self.source_path or not Path(self.source_path).exists():
            return False, f"源码目录不存在: {self.source_path or '(empty)'}"
        if "python_executable" in runtime_health.get("missing", []):
            return False, f"Python环境不可用: {self.runtime_config.python_executable}"
        if "worker_script" in runtime_health.get("missing", []):
            return False, f"worker脚本不存在: {self.runtime_config.worker_script or '(empty)'}"
        found = [w for w in self.discover_weights() if w["status"] == "found"]
        if not found:
            return False, "未找到完整可验证的本地权重/配置"
        return False, "已找到基础资源，但尚未通过真实小图健康检查，不能标记ready"

    def get_status(self) -> ModelStatus:
        installed, msg = self.check_installation()
        runtime_health = self._runtime_health()
        weights = self.discover_weights()
        return ModelStatus(
            model_id=self.model_id,
            display_name=self.display_name,
            description=self.description,
            repository_url=self.local_config.get("repository_url", self.repository_url),
            license_name=self.local_config.get("license_name", self.license_name),
            supported_devices=["cuda", "cpu"] if self.runtime_config.supports_cpu else ["cuda"],
            supported_precisions=["fp16", "fp32"] if self.runtime_config.supports_fp16 else ["fp32"],
            installed=installed,
            available=False,
            loaded=False,
            weight_path=None,
            source_path=self.source_path,
            status_message=msg,
            install_hint="配置 models.local.yaml 中的源码、Python环境、worker脚本、权重和配置；健康检查通过前不会参与正式任务。",
            capabilities={
                "weights": weights,
                "task_type": self.task_type,
                "execution_backend": self.runtime_config.execution_backend,
                "python_executable": self.runtime_config.python_executable,
                "environment_name": self.runtime_config.environment_name,
                "worker_script": self.runtime_config.worker_script,
                "runtime_health": runtime_health,
                "supports_tiling": self.runtime_config.supports_tiling,
                "supports_cpu": self.runtime_config.supports_cpu,
                "max_concurrency": self.runtime_config.max_concurrency,
                "ready_requires_real_small_image_inference": True,
                "auto_route": False,
                "rgb_static_image": True,
                "video": False,
                "event_camera": False,
            },
        )

    def health_check(self) -> dict:
        status = self.get_status()
        if not status.capabilities["runtime_health"].get("available"):
            return {
                "model_id": self.model_id,
                "available": False,
                "message": status.status_message,
                "runtime_health": status.capabilities["runtime_health"],
            }
        checkpoint = next((w["checkpoint_id"] for w in status.capabilities["weights"] if w["status"] == "found"), "")
        runtime = SubprocessModelRuntime(
            model_id=self.model_id,
            source_path=self.runtime_config.source_path,
            python_executable=self.runtime_config.python_executable,
            worker_script=self.runtime_config.worker_script,
            environment_name=self.runtime_config.environment_name,
            timeout_seconds=self.runtime_config.timeout_seconds,
        )
        response = SubprocessBackend(runtime).run(WorkerRequest(
            request_id=f"health-{self.model_id}",
            operation="health_check",
            output_path="health.json",
            checkpoint_id=checkpoint,
            device=self.runtime_config.device,
            precision=self.runtime_config.precision,
        ))
        return {
            "model_id": self.model_id,
            "checkpoint_id": checkpoint,
            "available": bool(response.success and not response.is_mock),
            "message": "真实worker健康检查通过" if response.success and not response.is_mock else response.error or "worker健康检查失败",
            "runtime_ms": response.runtime_ms,
            "peak_memory_mb": response.peak_memory_mb,
            "metadata": response.metadata,
        }

    def enhance(self, image_path: str, output_path: str, device: str, precision: str, parameters: dict) -> EnhancementResult:
        raise RuntimeError(f"{self.display_name} 尚未完成真实增强worker接入，不能执行正式推理")
