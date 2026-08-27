import os
import sys
import time
import traceback
import types
from pathlib import Path

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

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from worker_common import load_request, write_response


def _add_source_path(source_path: Path):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))


def _force_local_basicsr(source_path: Path):
    for name in list(sys.modules):
        if name == "basicsr" or name.startswith("basicsr."):
            del sys.modules[name]
    basicsr_root = source_path / "basicsr"
    if not basicsr_root.exists():
        return
    module = types.ModuleType("basicsr")
    module.__path__ = [str(basicsr_root)]
    module.__file__ = str(basicsr_root / "__init__.py")
    sys.modules["basicsr"] = module


def _select_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_name)
    return torch.device("cpu")


def _patch_torchvision_functional_tensor():
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    from torchvision.transforms.functional import rgb_to_grayscale

    module = types.ModuleType("torchvision.transforms.functional_tensor")
    module.rgb_to_grayscale = rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = module


def _cuda_stats_device(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return 0 if device.index is None else int(device.index)


def _load_model(source_path: Path, config_path: Path, weight_path: Path, device: torch.device):
    _add_source_path(source_path)
    _force_local_basicsr(source_path)
    _patch_torchvision_functional_tensor()
    cwd = Path.cwd()
    os.chdir(source_path)
    try:
        from basicsr.models import create_model
        from basicsr.utils.options import parse

        opt = parse(str(config_path), is_train=False)
        opt["dist"] = False
        opt["num_gpu"] = 1 if device.type == "cuda" else 0
        opt.setdefault("path", {})["pretrain_network_g"] = str(weight_path)
        model = create_model(opt)
        model.net_g = model.net_g.to(device)
        model.net_g.eval()
        return model
    finally:
        os.chdir(cwd)


def _read_image_tensor(path: Path, device: torch.device):
    from basicsr.utils import img2tensor

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"NAFNet input image not readable: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = img2tensor(img, bgr2rgb=False, float32=True).unsqueeze(0).to(device)
    return tensor


def _pad_to_multiple(tensor: torch.Tensor, multiple: int = 16):
    height, width = tensor.shape[-2], tensor.shape[-1]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if not pad_h and not pad_w:
        return tensor, height, width
    mode = "reflect" if height > pad_h and width > pad_w else "replicate"
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode), height, width


def _save_result(result: torch.Tensor, output_path: Path, height: int, width: int):
    from basicsr.utils import imwrite, tensor2img

    result = result[..., :height, :width]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite(tensor2img([result]), str(output_path))


def _infer(request: dict, response_path: Path) -> dict:
    params = request.get("parameters") or {}
    source_path = Path(params.get("source_path") or ".")
    weight_path = Path(params.get("weight_path") or "")
    config_path = Path(params.get("config_path") or "")
    checkpoint_id = request.get("checkpoint_id") or params.get("checkpoint_id", "")

    if not source_path.exists():
        raise RuntimeError(f"NAFNet source path not found: {source_path}")
    if not weight_path.exists():
        raise RuntimeError(f"NAFNet weight not found: {weight_path}")
    if not config_path.exists():
        raise RuntimeError(f"NAFNet config not found: {config_path}")

    device = _select_device(request.get("device") or "cuda:0")
    output_path = Path(request.get("output_path") or response_path.parent / "output.png")
    input_paths = request.get("input_paths") or []
    if request.get("operation") == "health_check" and not input_paths:
        health_input = response_path.parent / "nafnet_health_input.png"
        sample = np.zeros((64, 64, 3), dtype=np.uint8)
        sample[..., 0] = 18
        sample[..., 1] = 22
        sample[..., 2] = 28
        cv2.imwrite(str(health_input), sample)
        input_paths = [str(health_input)]
        output_path = response_path.parent / "nafnet_health_output.png"
    if not input_paths:
        raise RuntimeError("NAFNet worker requires one input image path")

    started = time.perf_counter()
    stats_device = _cuda_stats_device(device)
    if stats_device is not None:
        try:
            torch.cuda.reset_peak_memory_stats(stats_device)
        except RuntimeError:
            stats_device = None

    model = _load_model(source_path, config_path, weight_path, device)
    tensor = _read_image_tensor(Path(input_paths[0]), device)
    tensor, height, width = _pad_to_multiple(tensor)

    with torch.no_grad():
        model.feed_data({"lq": tensor})
        model.test()
        result = model.get_current_visuals()["result"]

    _save_result(result, output_path, height, width)
    if stats_device is not None:
        try:
            torch.cuda.synchronize(stats_device)
            peak_memory_mb = torch.cuda.max_memory_allocated(stats_device) / 1024 / 1024
        except RuntimeError:
            peak_memory_mb = 0.0
        torch.cuda.empty_cache()
    else:
        peak_memory_mb = 0.0

    return {
        "success": True,
        "model_id": "nafnet",
        "checkpoint_id": checkpoint_id,
        "is_mock": False,
        "output_path": str(output_path),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "peak_memory_mb": float(peak_memory_mb),
        "warnings": [],
        "metadata": {
            "device": str(device),
            "weight_path": str(weight_path),
            "config_path": str(config_path),
            "padded_to_multiple": 16,
        },
        "error": None,
    }


def main():
    request, response_path = load_request()
    try:
        response = _infer(request, response_path)
    except Exception as exc:
        response = {
            "success": False,
            "model_id": "nafnet",
            "checkpoint_id": request.get("checkpoint_id", ""),
            "is_mock": False,
            "output_path": None,
            "runtime_ms": 0,
            "peak_memory_mb": 0,
            "warnings": [],
            "metadata": {"operation": request.get("operation"), "traceback": traceback.format_exc()},
            "error": str(exc),
        }
    write_response(response_path, response)


if __name__ == "__main__":
    main()
