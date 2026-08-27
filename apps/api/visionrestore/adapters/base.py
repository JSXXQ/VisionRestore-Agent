import json
import os
import subprocess
from abc import ABC
from pathlib import Path
from pydantic import BaseModel
from visionrestore.core.config import PROJECT_ROOT, get_settings
from visionrestore.core.model_config import external_python, file_sha256, file_size, get_model_config
from visionrestore.schemas.model import ModelStatus

class EnhancementResult(BaseModel):
    output_path: str
    parameters: dict
    runtime_ms: int
    peak_memory_mb: float = 0
    logs: list[str] = []
    is_mock: bool = False
    adapter_class: str | None = None
    model_id: str | None = None
    checkpoint_id: str | None = None
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    device: str | None = None
    precision: str | None = None
    input_sha256: str | None = None
    output_sha256: str | None = None

class ModelAdapter(ABC):
    model_id: str
    display_name: str
    description: str
    repository_url: str = ""
    license_name: str = ""
    supported_devices: list[str] = ["cuda"]
    supported_precisions: list[str] = ["fp32", "fp16"]
    size_multiple: int = 1
    supports_tiling: bool = False
    default_checkpoint: str = ""

    def __init__(self):
        self.settings = get_settings()
        self.local_config = get_model_config().get("models", {}).get(self.model_id, {})
        self.loaded = False

    @property
    def source_path(self) -> str:
        return self.local_config.get("source_path", "")

    def discover_weights(self) -> list[dict]:
        weights = []
        for checkpoint_id, cfg in (self.local_config.get("weights", {}) or {}).items():
            path = cfg.get("path")
            config_path = cfg.get("config_path")
            exists = bool(path and Path(path).exists())
            config_exists = True if not config_path else Path(config_path).exists()
            status = "found" if exists and config_exists and Path(self.source_path).exists() else "unavailable"
            if not Path(self.source_path).exists(): status = "source_missing"
            elif not exists: status = "weight_missing"
            elif not config_exists: status = "config_missing"
            weights.append({
                "checkpoint_id": checkpoint_id,
                "display_name": cfg.get("display_name", checkpoint_id),
                "path": path,
                "file_name": Path(path).name if path else None,
                "size_bytes": file_size(path) if path else None,
                "sha256": file_sha256(path) if path and exists else None,
                "config_path": config_path,
                "config_exists": config_exists,
                "license_name": self.license_name,
                "domain": cfg.get("domain", ""),
                "auto_route": bool(cfg.get("auto_route", self.local_config.get("auto_route", True))),
                "default": bool(cfg.get("default", checkpoint_id == self.default_checkpoint)),
                "exists": exists,
                "status": status,
                "health_check": "not_run",
                "last_error": None,
            })
        return weights

    def check_installation(self) -> tuple[bool, str]:
        if not self.source_path or not Path(self.source_path).exists():
            return False, f"源码目录不存在: {self.source_path or '(empty)'}"
        found = [w for w in self.discover_weights() if w["status"] == "found"]
        if not found:
            return False, "未找到可用于推理的本地权重"
        return True, f"发现 {len(found)} 个本地权重；健康检查需运行真实小图推理"

    def install_hint(self) -> str:
        return "使用 config/models.local.yaml 指向已有本地源码和权重；本项目不会重复下载模型。"

    def get_status(self) -> ModelStatus:
        installed, msg = self.check_installation()
        cfg_url = self.local_config.get("repository_url") or self.repository_url
        cfg_license = self.local_config.get("license_name") or self.license_name
        weights = self.discover_weights()
        return ModelStatus(
            model_id=self.model_id,
            display_name=self.display_name,
            description=self.description,
            repository_url=cfg_url,
            license_name=cfg_license,
            supported_devices=self.supported_devices,
            supported_precisions=self.supported_precisions,
            installed=installed,
            available=installed,
            loaded=self.loaded,
            weight_path=None,
            source_path=self.source_path,
            status_message=msg,
            install_hint="使用 config/models.local.yaml 中的本地源码和权重路径；本项目不会重复下载。",
            capabilities={
                "weights": weights,
                "size_multiple": self.size_multiple,
                "supports_tiling": self.supports_tiling,
                "auto_route": bool(self.local_config.get("auto_route", True)),
                "manual_only_note": self.local_config.get("manual_only_note"),
                "rgb_static_image": True,
                "video": False,
                "event_camera": False,
            },
        )

    def validate_weight(self, checkpoint_id: str) -> dict:
        for item in self.discover_weights():
            if item["checkpoint_id"] == checkpoint_id:
                if item["status"] != "found":
                    raise RuntimeError(f"权重不可用: {checkpoint_id}: {item['status']}")
                return item
        raise RuntimeError(f"未知 checkpoint: {checkpoint_id}")

    def load(self, *args, **kwargs):
        ok, msg = self.check_installation()
        if not ok:
            raise RuntimeError(msg)
        self.loaded = True

    def unload(self):
        self.loaded = False

    def preprocess(self, *args, **kwargs): return None
    def infer(self, *args, **kwargs): return None
    def postprocess(self, *args, **kwargs): return None

    def estimate_memory(self, width: int, height: int, precision: str = "fp32") -> float:
        bytes_per = 2 if precision == "fp16" else 4
        return float(width * height * 3 * bytes_per * 10 / 1024 / 1024)

    def get_default_parameters(self) -> dict:
        return {"tile_size": 512, "tile_overlap": 64, "batch_size": 1, "device": "cuda", "precision": "fp32"}

    def validate_parameters(self, parameters: dict) -> dict:
        merged = self.get_default_parameters() | (parameters or {})
        merged["tile_size"] = max(128, int(merged.get("tile_size", 512)))
        merged["tile_overlap"] = max(0, min(int(merged.get("tile_overlap", 64)), merged["tile_size"] // 2))
        return merged

    def health_check(self) -> dict:
        try:
            checkpoint_id = self.default_checkpoint or next(w["checkpoint_id"] for w in self.discover_weights() if w["status"] == "found")
            health_dir = self.settings.cache_dir / "health"
            health_dir.mkdir(parents=True, exist_ok=True)
            input_path = health_dir / f"{self.model_id}_input.png"
            output_path = health_dir / f"{self.model_id}_{checkpoint_id}.png"
            if not input_path.exists():
                from PIL import Image
                Image.new("RGB", (32, 32), (18, 18, 22)).save(input_path)
            result = self.enhance(str(input_path), str(output_path), "cuda", "fp32", {"checkpoint_id": checkpoint_id})
            return {"model_id": self.model_id, "checkpoint_id": checkpoint_id, "available": True, "message": "真实小图推理通过", "runtime_ms": result.runtime_ms, "peak_memory_mb": result.peak_memory_mb}
        except Exception as exc:
            return {"model_id": self.model_id, "available": False, "message": str(exc)}

    def get_capabilities(self) -> dict:
        return self.get_status().capabilities

    def enhance(self, image_path: str, output_path: str, device: str, precision: str, parameters: dict) -> EnhancementResult:
        checkpoint_id = (parameters or {}).get("checkpoint_id") or self.default_checkpoint
        weight = self.validate_weight(checkpoint_id)
        params = self.validate_parameters(parameters)
        run_device = params.get("device", device or "cuda")
        run_precision = params.get("precision", precision or "fp32")
        input_digest = file_sha256(image_path)
        checkpoint_digest = weight.get("sha256") or file_sha256(weight["path"])
        logs = [
            f"Loading real model adapter: {self.__class__.__name__}",
            f"Loading checkpoint: {checkpoint_id}",
            f"Checkpoint SHA256: {checkpoint_digest}",
            f"Running inference on {run_device}: {self.model_id}:{checkpoint_id}",
        ]
        cmd = [
            external_python(),
            str(PROJECT_ROOT / "scripts" / "model_infer_runner.py"),
            "--model", self.model_id,
            "--checkpoint-id", checkpoint_id,
            "--source-path", self.source_path,
            "--weight-path", weight["path"],
            "--input", image_path,
            "--output", output_path,
            "--device", run_device,
            "--precision", run_precision,
        ]
        process_env = os.environ.copy()
        local_packages = PROJECT_ROOT / "data" / "python_packages"
        if local_packages.exists():
            existing_pythonpath = process_env.get("PYTHONPATH", "")
            process_env["PYTHONPATH"] = str(local_packages) + (
                os.pathsep + existing_pythonpath if existing_pythonpath else ""
            )
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            text=True,
            capture_output=True,
            timeout=int(params.get("timeout_seconds", 240)),
            env=process_env,
        )
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        payload = json.loads(lines[-1]) if lines else {"success": False, "error": proc.stderr.strip()}
        if proc.returncode != 0 or not payload.get("success"):
            raise RuntimeError(payload.get("error") or proc.stderr.strip() or "模型推理失败")
        output_digest = file_sha256(output_path)
        logs.extend([
            f"Inference completed: runtime_ms={int(payload.get('runtime_ms', 0))}, peak_memory_mb={float(payload.get('peak_memory_mb', 0)):.2f}",
            f"Output SHA256: {output_digest}",
        ])
        if proc.stderr.strip():
            logs.append(proc.stderr.strip())
        return EnhancementResult(
            output_path=output_path,
            parameters={**params, "checkpoint_id": checkpoint_id, "runner": payload},
            runtime_ms=int(payload.get("runtime_ms", 0)),
            peak_memory_mb=float(payload.get("peak_memory_mb", 0)),
            logs=logs,
            is_mock=False,
            adapter_class=self.__class__.__name__,
            model_id=self.model_id,
            checkpoint_id=checkpoint_id,
            checkpoint_path=weight["path"],
            checkpoint_sha256=checkpoint_digest,
            device=run_device,
            precision=run_precision,
            input_sha256=input_digest,
            output_sha256=output_digest,
        )

