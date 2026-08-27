from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from visionrestore.adapters.registry import ModelRegistry
from visionrestore.core.model_config import get_model_runtime_config


class RealESRGANStatusService:
    """Reports Real-ESRGAN readiness without fabricating runtime success."""

    REQUIRED_PYTHON_PACKAGES = ["torch", "cv2", "realesrgan", "basicsr"]
    EXPECTED_WEIGHT_FILES = {
        "realesrgan_x2plus": "RealESRGAN_x2plus.pth",
        "realesrgan_x4plus": "RealESRGAN_x4plus.pth",
        "realesr_general_x4v3": "realesr-general-x4v3.pth",
    }

    def __init__(self, registry: ModelRegistry | None = None):
        self.registry = registry or ModelRegistry()
        self.runtime = get_model_runtime_config("realesrgan")

    def status(self) -> dict[str, Any]:
        source_path = Path(self.runtime.source_path) if self.runtime.source_path else None
        python_path = Path(self.runtime.python_executable) if self.runtime.python_executable else None
        worker_path = Path(self.runtime.worker_script) if self.runtime.worker_script else None
        package_status = self._python_package_status()
        ncnn_status = self._ncnn_status()
        try:
            adapter = self.registry.get("realesrgan")
            model_status = adapter.get_status()
            weights = model_status.capabilities.get("weights", [])
            health = model_status.capabilities.get("last_health", {}) or {}
        except Exception as exc:
            model_status = None
            weights = []
            health = {"available": False, "message": str(exc)}

        installed_weights = [item for item in weights if item.get("status") == "found"]
        missing_expected = [
            {"checkpoint_id": checkpoint_id, "file_name": file_name}
            for checkpoint_id, file_name in self.EXPECTED_WEIGHT_FILES.items()
            if not any(item.get("checkpoint_id") == checkpoint_id and item.get("status") == "found" for item in weights)
        ]
        x2_ready = any(item.get("checkpoint_id") == "realesrgan_x2plus" and item.get("status") == "found" for item in weights)
        x4_ready = any(item.get("checkpoint_id") == "realesrgan_x4plus" and item.get("status") == "found" for item in weights)
        packages_ready = all(item["available"] for item in package_status.values())
        health_ready = bool(health.get("available"))
        pytorch_ready = bool(
            source_path and source_path.exists()
            and python_path and python_path.exists()
            and worker_path and worker_path.exists()
            and installed_weights
            and packages_ready
            and health_ready
        )
        return {
            "model_id": "realesrgan",
            "display_name": "Real-ESRGAN",
            "role": "formal_super_resolution_postprocess",
            "auto_enhancement_candidate": False,
            "mambair_realsr_enabled": False,
            "default_backend": "pytorch",
            "default_checkpoint": "realesrgan_x2plus",
            "default_scale": 2,
            "available_scales": [scale for scale, ready in [(2, x2_ready), (4, x4_ready)] if ready],
            "pytorch_backend": {
                "available": pytorch_ready,
                "source_path": str(source_path) if source_path else "",
                "source_exists": bool(source_path and source_path.exists()),
                "python_executable": str(python_path) if python_path else "",
                "python_exists": bool(python_path and python_path.exists()),
                "worker_script": str(worker_path) if worker_path else "",
                "worker_exists": bool(worker_path and worker_path.exists()),
                "packages": package_status,
                "health_check": health,
                "status_message": model_status.status_message if model_status else health.get("message", "status unavailable"),
            },
            "ncnn_vulkan_backend": ncnn_status,
            "weights": weights,
            "installed_weight_count": len(installed_weights),
            "missing_expected_weights": missing_expected,
            "policy": {
                "default_scale": 2,
                "automatic_x4": False,
                "requires_user_confirmation": True,
                "formal_mode_allows_mock": False,
            },
        }

    def _python_package_status(self) -> dict[str, dict[str, Any]]:
        if not self.runtime.python_executable or not Path(self.runtime.python_executable).exists():
            return {name: {"available": False, "error": "python_executable_missing"} for name in self.REQUIRED_PYTHON_PACKAGES}
        code = """
import importlib.util, json
mods = {name: importlib.util.find_spec(name) is not None for name in %r}
print(json.dumps(mods))
""" % self.REQUIRED_PYTHON_PACKAGES
        try:
            env = os.environ.copy()
            local_paths = []
            if self.runtime.source_path:
                source_path = Path(self.runtime.source_path)
                local_paths.extend([str(source_path), str(source_path.parent / "basicsr_runtime")])
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = os.pathsep.join([*local_paths, existing_pythonpath]).rstrip(os.pathsep)
            proc = subprocess.run(
                [self.runtime.python_executable, "-c", code],
                text=True,
                capture_output=True,
                timeout=20,
                env=env,
            )
            if proc.returncode != 0:
                return {name: {"available": False, "error": (proc.stderr or "package_check_failed").strip()} for name in self.REQUIRED_PYTHON_PACKAGES}
            import json
            found = json.loads((proc.stdout or "{}").strip().splitlines()[-1])
            return {name: {"available": bool(found.get(name)), "error": None if found.get(name) else "not_installed"} for name in self.REQUIRED_PYTHON_PACKAGES}
        except Exception as exc:
            return {name: {"available": False, "error": str(exc)} for name in self.REQUIRED_PYTHON_PACKAGES}

    def _ncnn_status(self) -> dict[str, Any]:
        exe = os.environ.get("REALESRGAN_NCNN_EXECUTABLE", "").strip()
        model_dir = os.environ.get("REALESRGAN_NCNN_MODEL_DIR", "").strip()
        exe_path = Path(exe) if exe else None
        model_path = Path(model_dir) if model_dir else None
        return {
            "available": bool(exe_path and exe_path.exists()),
            "configured": bool(exe),
            "executable": str(exe_path) if exe_path else "",
            "executable_exists": bool(exe_path and exe_path.exists()),
            "model_dir": str(model_path) if model_path else "",
            "model_dir_exists": bool(model_path and model_path.exists()),
            "status_message": "NCNN-Vulkan executable found" if exe_path and exe_path.exists() else "NCNN-Vulkan executable is not configured; PyTorch backend remains primary.",
        }
