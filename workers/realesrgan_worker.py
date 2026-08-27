import argparse
import json
import sys
import time
from pathlib import Path





def _install_realesrgan_version_compat() -> None:
    """Some local Real-ESRGAN source snapshots do not include realesrgan.version."""
    import types
    import sys as _sys
    module = types.ModuleType("realesrgan.version")
    module.__version__ = "0.3.0-local"
    module.__gitsha__ = "local"
    _sys.modules.setdefault("realesrgan.version", module)


def _install_torchvision_compat() -> None:
    """BasicSR 1.4.x imports a torchvision module removed in newer torchvision."""
    import types
    import sys as _sys
    try:
        import torchvision.transforms.functional as functional
    except Exception:
        return
    module = types.ModuleType("torchvision.transforms.functional_tensor")
    module.rgb_to_grayscale = functional.rgb_to_grayscale
    _sys.modules.setdefault("torchvision.transforms.functional_tensor", module)


def _install_local_dependency_paths(source_path: Path) -> None:
    """Load project-local model dependencies before importing Real-ESRGAN."""
    dependency_root = source_path.parent / "basicsr_runtime"
    if dependency_root.is_dir():
        sys.path.insert(0, str(dependency_root))


def _write_response(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _image_size(path: Path):
    from PIL import Image
    with Image.open(path) as image:
        return [int(image.width), int(image.height)]


def _checkpoint_spec(checkpoint_id: str) -> dict:
    if checkpoint_id == "realesrgan_x2plus":
        return {"model_name": "RealESRGAN_x2plus", "architecture": "RRDBNet", "native_scale": 2}
    if checkpoint_id == "realesrgan_x4plus":
        return {"model_name": "RealESRGAN_x4plus", "architecture": "RRDBNet", "native_scale": 4}
    if checkpoint_id == "realesr_general_x4v3":
        return {"model_name": "realesr-general-x4v3", "architecture": "SRVGGNetCompact", "native_scale": 4}
    raise ValueError(f"Unsupported Real-ESRGAN checkpoint: {checkpoint_id}")


def _build_model(spec: dict):
    if spec["architecture"] == "RRDBNet":
        from basicsr.archs.rrdbnet_arch import RRDBNet
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=spec["native_scale"])
    if spec["architecture"] == "SRVGGNetCompact":
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        return SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu")
    raise ValueError(f"Unsupported architecture: {spec['architecture']}")


def _run(request: dict) -> dict:
    started = time.perf_counter()
    params = request.get("parameters") or {}
    source_path = Path(params.get("source_path") or "")
    weight_path = Path(params.get("weight_path") or "")
    checkpoint_id = request.get("checkpoint_id") or params.get("checkpoint_id") or "realesrgan_x2plus"
    output_path = Path(request["output_path"])
    spec = _checkpoint_spec(checkpoint_id)

    if not source_path.exists():
        raise FileNotFoundError(f"Real-ESRGAN source path does not exist: {source_path}")
    if not weight_path.exists():
        raise FileNotFoundError(f"Real-ESRGAN checkpoint does not exist: {weight_path}")
    _install_local_dependency_paths(source_path)
    sys.path.insert(0, str(source_path))

    _install_realesrgan_version_compat()
    _install_torchvision_compat()
    import cv2
    import torch
    from realesrgan import RealESRGANer

    device = request.get("device") or params.get("device") or "cuda:0"
    precision = request.get("precision") or params.get("precision") or "fp16"
    outscale = float(params.get("outscale") or params.get("scale") or spec["native_scale"])
    tile_size = int(params.get("tile_size") or params.get("tile") or 0)
    tile_pad = int(params.get("tile_pad") or 10)
    pre_pad = int(params.get("pre_pad") or 0)
    denoise_strength = float(params.get("denoise_strength") or 1.0)
    half = precision == "fp16" and str(device).startswith("cuda") and torch.cuda.is_available()

    input_paths = request.get("input_paths") or []
    if request.get("operation") == "health_check":
        health_input = output_path.parent / "realesrgan_health_input.png"
        import numpy as np
        from PIL import Image
        Image.fromarray(np.full((16, 16, 3), 32, dtype=np.uint8)).save(health_input)
        input_path = health_input
    elif input_paths:
        input_path = Path(input_paths[0])
    else:
        raise ValueError("Real-ESRGAN request requires an input image")

    if not input_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {input_path}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = _build_model(spec)
    upsampler = RealESRGANer(
        scale=spec["native_scale"],
        model_path=str(weight_path),
        dni_weight=None,
        model=model,
        tile=tile_size,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=half,
        device=torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"),
    )
    img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Input image cannot be decoded by OpenCV: {input_path}")
    output, _ = upsampler.enhance(img, outscale=outscale)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), output)
    if not ok:
        raise RuntimeError(f"Failed to write Real-ESRGAN output: {output_path}")

    input_size = _image_size(input_path)
    output_size = _image_size(output_path)
    expected_size = [int(round(input_size[0] * outscale)), int(round(input_size[1] * outscale))]
    if output_size != expected_size:
        raise RuntimeError(f"Real-ESRGAN output size mismatch: {output_size} != {expected_size}")
    peak_memory_mb = 0.0
    if torch.cuda.is_available():
        peak_memory_mb = float(torch.cuda.max_memory_allocated() / 1024 / 1024)
    runtime_ms = int((time.perf_counter() - started) * 1000)
    return {
        "success": True,
        "model_id": "realesrgan",
        "checkpoint_id": checkpoint_id,
        "is_mock": False,
        "output_path": str(output_path),
        "runtime_ms": runtime_ms,
        "peak_memory_mb": peak_memory_mb,
        "warnings": [],
        "metadata": {
            "backend": "pytorch",
            "model_id": spec["model_name"],
            "architecture": spec["architecture"],
            "native_scale": spec["native_scale"],
            "outscale": outscale,
            "input_size": input_size,
            "output_size": output_size,
            "expected_size": expected_size,
            "tile_size": tile_size,
            "tile_pad": tile_pad,
            "pre_pad": pre_pad,
            "denoise_strength": denoise_strength,
            "device": device,
            "precision": precision,
            "half": half,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args()
    request_path = Path(args.request)
    response_path = Path(args.response)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    try:
        payload = _run(request)
    except Exception as exc:
        payload = {
            "success": False,
            "model_id": "realesrgan",
            "checkpoint_id": request.get("checkpoint_id", ""),
            "is_mock": False,
            "runtime_ms": 0,
            "peak_memory_mb": 0,
            "warnings": [],
            "metadata": {"backend": "pytorch"},
            "error": str(exc),
        }
    _write_response(response_path, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
