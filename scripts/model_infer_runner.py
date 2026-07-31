import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def image_to_tensor(torch, image_path: str, device: str, precision: str, multiple: int):
    img = Image.open(image_path).convert("RGB")
    original_size = img.size
    arr = np.asarray(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    if precision == "fp16":
        tensor = tensor.half()
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple if multiple > 1 else 0
    pad_w = (multiple - w % multiple) % multiple if multiple > 1 else 0
    if pad_h or pad_w:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, original_size, (pad_w, pad_h)


def tensor_to_image(torch, tensor, output_path: str, original_size):
    w, h = original_size
    tensor = tensor[..., :h, :w]
    tensor = torch.clamp(tensor.detach().float().cpu(), 0, 1)[0]
    arr = tensor.permute(1, 2, 0).numpy()
    img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
    if img.size != original_size:
        raise RuntimeError(f"output size mismatch before save: {img.size} != {original_size}")
    img.save(output_path)
    saved = Image.open(output_path)
    if saved.size != original_size:
        raise RuntimeError(f"saved output size mismatch: {saved.size} != {original_size}")


def load_state_dict(torch, path: str):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ["params", "params_ema", "state_dict", "model"]:
            if key in ckpt and hasattr(ckpt[key], "keys"):
                ckpt = ckpt[key]
                break
    return {k.replace("module.", "", 1): v for k, v in ckpt.items()}


def run_retinexformer(args, torch):
    arch_path = Path(args.source_path) / "basicsr" / "models" / "archs" / "RetinexFormer_arch.py"
    mod = load_module("vr_retinexformer_arch", arch_path)
    model = mod.RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])
    state = load_state_dict(torch, args.weight_path)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch missing={missing} unexpected={unexpected}")
    return model, 4


def run_sci(args, torch):
    sys.path.insert(0, args.source_path)
    mod = load_module("vr_sci_model", Path(args.source_path) / "model.py")
    model = mod.Finetunemodel(args.weight_path)
    return model, 1


def run_zero_dce(args, torch):
    sys.path.insert(0, args.source_path)
    mod = load_module("vr_zero_dce_model", Path(args.source_path) / "model.py")
    model = mod.enhance_net_nopool()
    state = load_state_dict(torch, args.weight_path)
    model.load_state_dict(state, strict=True)
    return model, 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["retinexformer", "sci", "zero_dce"])
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--weight-path", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    args = parser.parse_args()

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.precision == "fp16" and device == "cpu":
        args.precision = "fp32"

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()

    if args.model == "retinexformer":
        model, multiple = run_retinexformer(args, torch)
    elif args.model == "sci":
        model, multiple = run_sci(args, torch)
    else:
        model, multiple = run_zero_dce(args, torch)

    model = model.to(device).eval()
    if args.precision == "fp16":
        model = model.half()
    tensor, original_size, padding = image_to_tensor(torch, args.input, device, args.precision, multiple)
    with torch.no_grad():
        if args.model == "sci":
            _, output = model(tensor)
        elif args.model == "zero_dce":
            _, output, _ = model(tensor)
        else:
            output = model(tensor)
    if device == "cuda":
        torch.cuda.synchronize()
    tensor_to_image(torch, output, args.output, original_size)
    runtime_ms = int((time.perf_counter() - start) * 1000)
    peak = float(torch.cuda.max_memory_allocated() / 1024 / 1024) if device == "cuda" else 0.0
    print(json.dumps({
        "success": True,
        "model_id": args.model,
        "checkpoint_id": args.checkpoint_id,
        "output_path": args.output,
        "runtime_ms": runtime_ms,
        "peak_memory_mb": peak,
        "device": device,
        "precision": args.precision,
        "original_width": original_size[0],
        "original_height": original_size[1],
        "padding": {"right": padding[0], "bottom": padding[1]},
        "size_multiple": multiple,
    }, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False))
        raise SystemExit(2)
