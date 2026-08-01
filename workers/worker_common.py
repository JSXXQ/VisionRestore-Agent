import argparse
import json
from pathlib import Path


def load_request():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    return request, Path(args.response)


def write_response(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def unsupported_worker(model_id, message):
    request, response_path = load_request()
    write_response(response_path, {
        "success": False,
        "model_id": model_id,
        "checkpoint_id": request.get("checkpoint_id", ""),
        "is_mock": False,
        "output_path": None,
        "runtime_ms": 0,
        "peak_memory_mb": 0,
        "warnings": [],
        "metadata": {"operation": request.get("operation"), "worker_status": "not_implemented"},
        "error": message,
    })
