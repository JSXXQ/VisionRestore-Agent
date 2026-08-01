import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image

_DLL_DIRECTORY_HANDLES = []


def _prepare_conda_dll_search_path():
    env_root = Path(sys.executable).resolve().parent
    for rel in ["bin", "Library/bin", "DLLs"]:
        dll_dir = env_root / rel
        if dll_dir.exists():
            os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(dll_dir)))


_prepare_conda_dll_search_path()

import torch
import torch.nn.functional as F
import yaml


def _load_image(path: Path, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    tensor = data.view(img.height, img.width, 3).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    return tensor.to(device)


def _save_image(tensor: torch.Tensor, path: Path) -> None:
    tensor = tensor.detach().float().cpu().clamp(0, 1).squeeze(0)
    arr = tensor.mul(255.0).byte().permute(1, 2, 0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path)


def _load_network_config(config_path: Path | None) -> dict:
    if config_path and config_path.exists():
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        network = payload.get("network") or {}
    else:
        network = {}
    return {
        "img_channel": int(network.get("img_channels", 3)),
        "width": int(network.get("width", 32)),
        "middle_blk_num_enc": int(network.get("middle_blk_num_enc", 2)),
        "middle_blk_num_dec": int(network.get("middle_blk_num_dec", 2)),
        "enc_blk_nums": list(network.get("enc_blk_nums", [1, 2, 3])),
        "dec_blk_nums": list(network.get("dec_blk_nums", [3, 1, 1])),
        "dilations": list(network.get("dilations", [1, 4, 9])),
        "extra_depth_wise": bool(network.get("extra_depth_wise", True)),
    }


def _load_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ["params", "state_dict", "model", "net"]:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def _strip_module_prefix(state_dict: dict) -> dict:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def _infer(request: dict, response_path: Path) -> dict:
    params = request.get("parameters") or {}
    source_path = Path(params.get("source_path") or ".")
    archs_path = source_path / "archs"
    if str(archs_path) not in sys.path:
        sys.path.insert(0, str(archs_path))
    darkir_file = archs_path / "DarkIR.py"
    spec = importlib.util.spec_from_file_location("darkir_arch", darkir_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"DarkIR source file not found: {darkir_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    DarkIR = module.DarkIR

    checkpoint_id = request.get("checkpoint_id") or params.get("checkpoint_id", "")
    weight_path = Path(params.get("weight_path") or "")
    if not weight_path.exists():
        raise RuntimeError(f"DarkIR weight not found: {weight_path}")
    config_path = Path(params["config_path"]) if params.get("config_path") else None

    device_name = request.get("device", "cuda:0")
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    input_paths = request.get("input_paths") or []
    output_path = Path(request.get("output_path") or response_path.parent / "output.png")
    if request.get("operation") == "health_check" and not input_paths:
        health_input = response_path.parent / "health_input.png"
        Image.new("RGB", (32, 32), (18, 18, 22)).save(health_input)
        input_paths = [str(health_input)]
        output_path = response_path.parent / "health_output.png"
    if not input_paths:
        raise RuntimeError("DarkIR worker requires one input image")

    started = time.perf_counter()
    model = DarkIR(**_load_network_config(config_path)).to(device)
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(_strip_module_prefix(_load_state_dict(checkpoint)))
    model.eval()

    img = _load_image(Path(input_paths[0]), device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        output = model(img, side_loss=False)
    output = torch.clamp(output, 0.0, 1.0)
    _save_image(output, output_path)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        torch.cuda.empty_cache()
    else:
        peak_memory_mb = 0
    return {
        "success": True,
        "model_id": "darkir",
        "checkpoint_id": checkpoint_id,
        "is_mock": False,
        "output_path": str(output_path),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "peak_memory_mb": float(peak_memory_mb),
        "warnings": [],
        "metadata": {"device": str(device), "weight_path": str(weight_path), "config_path": str(config_path) if config_path else None},
        "error": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args()
    response_path = Path(args.response)
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    try:
        response = _infer(request, response_path)
    except Exception as exc:
        response = {
            "success": False,
            "model_id": "darkir",
            "checkpoint_id": request.get("checkpoint_id", ""),
            "is_mock": False,
            "output_path": None,
            "runtime_ms": 0,
            "peak_memory_mb": 0,
            "warnings": [],
            "metadata": {"operation": request.get("operation")},
            "error": str(exc),
        }
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
