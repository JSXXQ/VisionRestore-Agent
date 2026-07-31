import time
from .base import InferenceBackend

class PyTorchBackend(InferenceBackend):
    def __init__(self, device: str = "auto", precision: str = "fp32"):
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("PyTorch 未安装，无法运行真实深度学习模型") from exc
        self.torch = torch
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        if precision == "fp16" and self.device == "cpu":
            self.precision = "fp32"
            self.precision_fallback_reason = "CPU 不支持本项目的 fp16 推理，已回退到 fp32"
        else:
            self.precision = precision
            self.precision_fallback_reason = None
        self.model = None

    def load_model(self, model):
        self.model = model.to(self.device)
        self.model.eval()
        if self.precision == "fp16":
            self.model.half()
        return self.model

    def run(self, tensor):
        if self.model is None:
            raise RuntimeError("模型尚未加载")
        tensor = tensor.to(self.device)
        if self.precision == "fp16":
            tensor = tensor.half()
        with self.torch.no_grad():
            return self.model(tensor)

    def synchronize(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def get_device(self) -> str:
        return self.device

    def get_precision(self) -> str:
        return self.precision

    def get_peak_memory(self) -> float:
        if self.device == "cuda":
            return float(self.torch.cuda.max_memory_allocated() / 1024 / 1024)
        return 0.0

    def release(self):
        self.model = None
        if self.device == "cuda":
            self.torch.cuda.empty_cache()
