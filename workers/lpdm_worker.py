import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from einops import rearrange
from omegaconf import OmegaConf
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
import torchvision.transforms as T


def _tensor_to_pil(t, mode="RGB"):
    if t.ndim == 4:
        t = t[0]
    t = torch.clamp((t + 1.0) / 2.0, min=0.0, max=1.0)
    img = 255.0 * rearrange(t.detach().cpu().numpy(), "c h w -> h w c")
    return Image.fromarray(img.astype(np.uint8), mode=mode).convert("RGB")


def _pil_to_tensor_in_range(path: Path) -> torch.Tensor:
    return T.ToTensor()(Image.open(path).convert("RGB")) * 2.0 - 1.0


def _pad_to_multiple(im, mul=16):
    h, w = im.shape[2], im.shape[3]
    pad_h = (mul - h % mul) % mul
    pad_w = (mul - w % mul) % mul
    if pad_h == 0 and pad_w == 0:
        return im
    mode = "reflect" if h > pad_h and w > pad_w else "constant"
    return F.pad(im, (0, pad_w, 0, pad_h), mode=mode)


def _load_model_from_config(source_path: Path, config_path: Path, ckpt_path: Path, device: torch.device):
    for extra in [source_path, source_path / "external" / "taming-transformers", source_path / "external" / "clip"]:
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    from ldm.util import instantiate_from_config

    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def _infer(request: dict, response_path: Path) -> dict:
    params = request.get("parameters") or {}
    source_path = Path(params.get("source_path") or ".")
    weight_path = Path(params.get("weight_path") or "")
    config_path = Path(params.get("config_path") or "")
    if not source_path.exists():
        raise RuntimeError(f"LPDM source path not found: {source_path}")
    if not weight_path.exists():
        raise RuntimeError(f"LPDM weight not found: {weight_path}")
    if not config_path.exists():
        raise RuntimeError(f"LPDM config not found: {config_path}")

    device_name = request.get("device", "cuda:0")
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    input_paths = request.get("input_paths") or []
    output_path = Path(request.get("output_path") or response_path.parent / "output.png")
    if request.get("operation") == "health_check" and not input_paths:
        pred = response_path.parent / "health_pred.png"
        cond = response_path.parent / "health_cond.png"
        Image.new("RGB", (64, 64), (34, 34, 38)).save(pred)
        Image.new("RGB", (64, 64), (18, 18, 22)).save(cond)
        input_paths = [str(pred), str(cond)]
        output_path = response_path.parent / "health_output.png"
    if not input_paths:
        raise RuntimeError("LPDM worker requires an enhanced image path")

    pred_path = Path(input_paths[0])
    cond_path = Path(input_paths[1]) if len(input_paths) > 1 else Path(params.get("cond_path") or input_paths[0])
    if not cond_path.exists():
        cond_path = pred_path
    phi = int(params.get("phi", 300))
    s = int(params.get("s", 30))

    started = time.perf_counter()
    model = _load_model_from_config(source_path, config_path, weight_path, device)
    p = _pil_to_tensor_in_range(pred_path).unsqueeze(0).to(device)
    c = _pil_to_tensor_in_range(cond_path).unsqueeze(0).to(device)
    height, width = p.shape[-2], p.shape[-1]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        t = torch.tensor([phi], dtype=torch.long, device=device)
        try:
            noise_pred = model.model(torch.cat([p, c], dim=1), t).detach()
            x0 = model.predict_start_from_noise(p, torch.tensor([s], device=device).long(), noise_pred).detach()
        except Exception:
            padded_p = _pad_to_multiple(p)
            padded_c = _pad_to_multiple(c)
            noise_pred = model.model(torch.cat([padded_p, padded_c], dim=1), t).detach()
            x0 = model.predict_start_from_noise(padded_p, torch.tensor([s], device=device).long(), noise_pred).detach()
            x0 = x0[..., :height, :width]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _tensor_to_pil(torch.clamp(x0, -1.0, 1.0)).save(output_path)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        torch.cuda.empty_cache()
    else:
        peak_memory_mb = 0
    return {
        "success": True,
        "model_id": "lpdm",
        "checkpoint_id": request.get("checkpoint_id") or params.get("checkpoint_id", ""),
        "is_mock": False,
        "output_path": str(output_path),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "peak_memory_mb": float(peak_memory_mb),
        "warnings": [],
        "metadata": {"device": str(device), "weight_path": str(weight_path), "config_path": str(config_path), "phi": phi, "s": s},
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
            "model_id": "lpdm",
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