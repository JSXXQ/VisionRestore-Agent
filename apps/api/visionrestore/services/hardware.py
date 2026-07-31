import platform
import shutil
import subprocess
import sys

class HardwareInspector:
    def inspect(self) -> dict:
        info = {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "cpu": platform.processor() or platform.machine(),
            "cuda_available": False,
            "gpu_name": None,
            "gpu_memory_mb": None,
            "torch": None,
            "nvidia_smi": bool(shutil.which("nvidia-smi")),
        }
        try:
            import torch
            info["torch"] = torch.__version__
            info["cuda_available"] = bool(torch.cuda.is_available())
            if info["cuda_available"]:
                idx = torch.cuda.current_device()
                info["gpu_name"] = torch.cuda.get_device_name(idx)
                info["gpu_memory_mb"] = int(torch.cuda.get_device_properties(idx).total_memory / 1024 / 1024)
        except Exception as exc:
            info["torch_error"] = str(exc)
        if info["nvidia_smi"] and not info.get("gpu_memory_mb"):
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=3,
                ).strip()
                if out:
                    name, mem = [part.strip() for part in out.splitlines()[0].split(",", 1)]
                    info["gpu_name"] = name
                    info["gpu_memory_mb"] = int(mem)
            except Exception as exc:
                info["nvidia_smi_error"] = str(exc)
        return info
