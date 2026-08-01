import importlib.util
from pathlib import Path

from worker_common import load_request, write_response


def main():
    request, response_path = load_request()
    params = request.get("parameters") or {}
    required_modules = ["mamba_ssm", "causal_conv1d", "timm", "einops"]
    missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
    source_path = Path(params.get("source_path") or "")
    weight_path = Path(params.get("weight_path") or "")
    file_missing = []
    if not source_path.exists():
        file_missing.append(f"source_path:{source_path}")
    if not weight_path.exists():
        file_missing.append(f"weight_path:{weight_path}")
    error_parts = []
    if missing:
        error_parts.append("Missing Python packages: " + ", ".join(missing))
    if file_missing:
        error_parts.append("Missing local files: " + ", ".join(file_missing))
    write_response(response_path, {
        "success": False,
        "model_id": "mambair",
        "checkpoint_id": request.get("checkpoint_id", ""),
        "is_mock": False,
        "output_path": None,
        "runtime_ms": 0,
        "peak_memory_mb": 0,
        "warnings": [],
        "metadata": {"operation": request.get("operation"), "missing_modules": missing, "missing_files": file_missing, "worker_status": "dependency_blocked"},
        "error": "; ".join(error_parts) or "MambaIR dependency check passed, but real inference worker is not enabled yet.",
    })


if __name__ == "__main__":
    main()