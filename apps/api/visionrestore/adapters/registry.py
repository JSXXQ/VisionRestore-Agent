from visionrestore.core.config import get_settings
from .zero_dce import ZeroDCEAdapter
from .sci import SCIAdapter
from .retinexformer import RetinexformerAdapter
from .snr_aware import SNRAwareAdapter
from .mock import MockModelAdapter

class ModelRegistry:
    def __init__(self):
        self.adapters = {
            "zero_dce": ZeroDCEAdapter(),
            "sci": SCIAdapter(),
            "retinexformer": RetinexformerAdapter(),
            "snr_aware": SNRAwareAdapter(),
        }
        if get_settings().allow_mock_models:
            self.adapters["mock_model"] = MockModelAdapter()

    def list(self):
        return [a.get_status().model_dump() for a in self.adapters.values()]

    def get(self, model_id: str):
        if model_id not in self.adapters:
            raise KeyError(model_id)
        return self.adapters[model_id]

    def available(self):
        return [a for a in self.adapters.values() if a.get_status().available]
