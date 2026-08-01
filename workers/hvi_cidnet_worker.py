import argparse
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


def _pad_tensor(tensor: torch.Tensor, multiple: int = 8) -> torch.Tensor:
    _, _, height, width = tensor.shape
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor
    mode = "reflect" if height > pad_h and width > pad_w else "constant"
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode)


def _load_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ["params", "state_dict", "model", "net"]:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def _infer(request: dict, response_path: Path) -> dict:
    params = request.get("parameters") or {}
    source_path = Path(params.get("source_path") or ".")
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))
    from net.CIDNet import CIDNet

    checkpoint_id = request.get("checkpoint_id") or params.get("checkpoint_id", "")
    weight_path = Path(params.get("weight_path") or "")
    if not weight_path.exists():
        raise RuntimeError(f"HVI-CIDNet weight not found: {weight_path}")

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
        raise RuntimeError("HVI-CIDNet worker requires one input image")

    started = time.perf_counter()
    model = CIDNet().to(device)
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(_load_state_dict(checkpoint))
    model.eval()

    gamma = float(params.get("gamma", 1.0))
    alpha = params.get("alpha")
    use_gated2 = bool(params.get("gated2", False))
    use_gated = bool(params.get("gated", False))
    if use_gated:
        model.trans.gated = True
    if use_gated2:
        model.trans.gated2 = True
        if alpha is not None:
            model.trans.alpha = float(alpha)

    img = _load_image(Path(input_paths[0]), device)
    _, _, height, width = img.shape
    tensor = _pad_tensor(img, 8)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        output = model(tensor ** gamma)
    output = output[:, :, :height, :width]
    _save_image(output, output_path)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        torch.cuda.empty_cache()
    else:
        peak_memory_mb = 0
    return {
        "success": True,
        "model_id": "hvi_cidnet",
        "checkpoint_id": checkpoint_id,
        "is_mock": False,
        "output_path": str(output_path),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "peak_memory_mb": float(peak_memory_mb),
        "warnings": [],
        "metadata": {"device": str(device), "weight_path": str(weight_path), "gamma": gamma},
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
            "model_id": "hvi_cidnet",
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