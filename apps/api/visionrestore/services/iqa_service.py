from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from visionrestore.core.model_config import get_model_config
from visionrestore.storage.database import Database
from visionrestore.utils.file_security import resolve_registered_path


class IQAService:
    METRICS = ["musiq", "clipiqa"]
    DISABLED_METRICS = {
        "topiq_nr": "disabled: CFANet registration/dependency chain is unstable in the current environment",
        "niqe": "disabled: unreliable on small health images and not required for current scoring",
        "brisque": "disabled: CUDA/NVRTC instability in the current environment",
    }

    def __init__(self, database: Database | None = None):
        self.db = database or Database()
        cfg = get_model_config()
        self.python_executable = cfg.get("iqa_python_executable") or cfg.get("external_python") or "python"
        self.device = str(cfg.get("iqa_device") or "cuda")

    def status(self) -> dict[str, Any]:
        python_path = Path(self.python_executable)
        package = self._package_available()
        return {
            "available": bool(python_path.exists() and package.get("pyiqa", {}).get("available")),
            "backend": "pyiqa",
            "python_executable": self.python_executable,
            "python_exists": python_path.exists(),
            "device": self.device,
            "metrics": self.METRICS,
            "enabled_metrics": self.METRICS,
            "disabled_metrics": self.DISABLED_METRICS,
            "packages": package,
            "formal_mode_allows_mock": False,
            "fallback": "local_technical_score_only" if not package.get("pyiqa", {}).get("available") else "none",
            "status_message": "pyiqa is ready; default IQA metrics are MUSIQ and CLIP-IQA" if package.get("pyiqa", {}).get("available") else "pyiqa is not installed in the configured environment",
        }

    def evaluate_candidate(self, *, task, candidate_id: str) -> dict[str, Any]:
        candidate = next((item for item in task.candidates if item.candidate_id == candidate_id or item.output_file_id == candidate_id), None)
        if not candidate:
            return {"available": False, "status": "candidate_not_found", "candidate_id": candidate_id}
        path = self._candidate_path(candidate)
        if not path or not path.exists():
            return {"available": False, "status": "output_missing", "candidate_id": candidate_id, "output_path": str(path) if path else ""}
        status = self.status()
        if not status["available"]:
            result = {"available": False, "status": "unavailable", "candidate_id": candidate_id, "output_path": str(path), "reason": status["status_message"], "status_detail": status}
            self.db.put_entity("candidate_metric", f"{task.task_id}:{candidate_id}:iqa", result, task_id=task.task_id)
            return result
        result = self.evaluate_image(path)
        payload = {"candidate_id": candidate_id, "output_path": str(path), **result}
        self.db.put_entity("candidate_metric", f"{task.task_id}:{candidate_id}:iqa", payload, task_id=task.task_id)
        return payload

    def evaluate_image(self, image_path: Path) -> dict[str, Any]:
        code = r'''
import json, sys, traceback
from pathlib import Path
image_path = Path(sys.argv[1])
device = sys.argv[2]
metric_names = json.loads(sys.argv[3])
try:
    import pyiqa
    results = {}
    errors = {}
    for name in metric_names:
        try:
            metric = pyiqa.create_metric(name, device=device)
            value = metric(str(image_path))
            try:
                raw = float(value.detach().cpu().item())
            except Exception:
                raw = float(value)
            results[name] = raw
        except Exception as exc:
            errors[name] = f"{exc.__class__.__name__}: {exc}"
    print(json.dumps({"success": True, "raw": results, "errors": errors}, ensure_ascii=False))
except Exception as exc:
    print(json.dumps({"success": False, "error": f"{exc.__class__.__name__}: {exc}", "traceback": traceback.format_exc()}, ensure_ascii=False))
'''
        proc = subprocess.run([self.python_executable, "-c", code, str(image_path), self.device, json.dumps(self.METRICS)], text=True, capture_output=True, timeout=180)
        if proc.returncode != 0:
            return {"available": False, "status": "worker_failed", "error": (proc.stderr or "pyiqa subprocess failed").strip(), "raw": {}, "errors": {}}
        try:
            payload = json.loads((proc.stdout or "{}").strip().splitlines()[-1])
        except Exception as exc:
            return {"available": False, "status": "invalid_worker_output", "error": str(exc), "raw": {}, "errors": {}}
        if not payload.get("success"):
            return {"available": False, "status": "failed", "error": payload.get("error"), "raw": {}, "errors": {}}
        raw = payload.get("raw") or {}
        errors = payload.get("errors") or {}
        normalized = self._normalize(raw)
        return {"available": bool(raw), "status": "completed" if raw else "all_metrics_failed", "backend": "pyiqa", "raw": raw, "normalized": normalized, "score": round(sum(normalized.values()) / len(normalized), 4) if normalized else None, "errors": errors}

    def _package_available(self) -> dict[str, dict[str, Any]]:
        python_path = Path(self.python_executable)
        if not python_path.exists():
            return {"pyiqa": {"available": False, "error": "python_executable_missing"}}
        code = "import importlib.util, json; print(json.dumps({'pyiqa': importlib.util.find_spec('pyiqa') is not None}))"
        try:
            proc = subprocess.run([self.python_executable, "-c", code], text=True, capture_output=True, timeout=20)
            found = json.loads((proc.stdout or "{}").strip().splitlines()[-1]) if proc.returncode == 0 else {}
            return {"pyiqa": {"available": bool(found.get("pyiqa")), "error": None if found.get("pyiqa") else "not_installed"}}
        except Exception as exc:
            return {"pyiqa": {"available": False, "error": str(exc)}}

    def _candidate_path(self, candidate) -> Path | None:
        if candidate.output_path:
            return Path(candidate.output_path)
        if candidate.output_file_id:
            record = self.db.get_file(candidate.output_file_id)
            if record:
                return resolve_registered_path(record["relative_path"])
        return None

    def _normalize(self, raw: dict[str, float]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for name, value in raw.items():
            v = float(value)
            if name in {"niqe", "brisque"}:
                normalized[name] = max(0.0, min(1.0, 1.0 - v / 100.0))
            elif name == "clipiqa":
                normalized[name] = max(0.0, min(1.0, v))
            elif name == "musiq":
                normalized[name] = max(0.0, min(1.0, v / 100.0))
            elif name == "topiq_nr":
                normalized[name] = max(0.0, min(1.0, v))
        return normalized
