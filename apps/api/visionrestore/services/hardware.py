import json
import platform
import shutil
import subprocess
import sys
from visionrestore.core.model_config import external_python

class HardwareInspector:
    def inspect(self) -> dict:
        info = {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "inference_python": external_python(),
            "cpu": platform.processor() or platform.machine(),
            "cuda_available": False,
            "gpu_name": None,
            "gpu_memory_mb": None,
            "torch": None,
            "nvidia_smi": bool(shutil.which("nvidia-smi")),
        }
        try:
            code = """
import json, torch
print(json.dumps({
 'torch': torch.__version__,
 'cuda_available': bool(torch.cuda.is_available()),
 'cuda_version': torch.version.cuda,
 'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
 'gpu_memory_mb': int(torch.cuda.get_device_properties(0).total_memory/1024/1024) if torch.cuda.is_available() else None,
}))
"""
            out = subprocess.check_output([external_python(), "-c", code], text=True, stderr=subprocess.STDOUT, timeout=20).strip().splitlines()[-1]
            data = json.loads(out)
            info.update(data)
        except Exception as exc:
            info["inference_python_error"] = str(exc)
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
                    info["cuda_available"] = True
            except Exception as exc:
                info["nvidia_smi_error"] = str(exc)
        return info
