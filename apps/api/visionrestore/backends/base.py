from abc import ABC, abstractmethod

class InferenceBackend(ABC):
    @abstractmethod
    def load_model(self, *args, **kwargs): ...
    @abstractmethod
    def run(self, *args, **kwargs): ...
    def synchronize(self): return None
    @abstractmethod
    def get_device(self) -> str: ...
    @abstractmethod
    def get_precision(self) -> str: ...
    def get_peak_memory(self) -> float: return 0.0
    def release(self): return None
