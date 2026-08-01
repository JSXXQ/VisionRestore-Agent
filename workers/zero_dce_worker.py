import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from PIL import Image

MODEL_ID = "zero_dce"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    response_path = Path(args.response)
    params = request.get("parameters") or {}
    checkpoint_id = request.get("checkpoint_id") or params.get("checkpoint_id", "")
    input_paths = request.get("input_paths") or []
    output_path = Path(request.get("output_path") or response_path.with_suffix(".png"))
    if request.get("operation") == "health_check" and not input_paths:
        health_input = response_path.parent / "health_input.png"
        Image.new("RGB", (32, 32), (18, 18, 22)).save(health_input)
        input_paths = [str(health_input)]
        output_path = response_path.parent / "health_output.png"
    started = time.perf_counter()
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "model_infer_runner.py"),
        "--model", MODEL_ID,
        "--checkpoint-id", checkpoint_id,
        "--source-path", params.get("source_path", ""),
        "--weight-path", params.get("weight_path", ""),
        "--input", input_paths[0] if input_paths else "",
        "--output", str(output_path),
        "--device", request.get("device", "cuda:0"),
        "--precision", request.get("precision", "fp16"),
    ]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True, capture_output=True, timeout=int(params.get("timeout_seconds", 240)))
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    payload = json.loads(lines[-1]) if lines else {"success": False, "error": proc.stderr.strip()}
    success = proc.returncode == 0 and payload.get("success") is True
    response = {
        "success": success,
        "model_id": MODEL_ID,
        "checkpoint_id": checkpoint_id,
        "is_mock": False,
        "output_path": str(output_path) if success else None,
        "runtime_ms": int(payload.get("runtime_ms") or (time.perf_counter() - started) * 1000),
        "peak_memory_mb": float(payload.get("peak_memory_mb", 0)),
        "warnings": [proc.stderr.strip()] if proc.stderr.strip() else [],
        "metadata": {"runner": payload},
        "error": None if success else payload.get("error") or proc.stderr.strip() or "worker inference failed",
    }
    response_path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

