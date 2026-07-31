from .zero_dce import ZeroDCEAdapter
from .sci import SCIAdapter
from .retinexformer import RetinexformerAdapter
from .mock import MockModelAdapter
from visionrestore.core.config import get_settings

class ModelRegistry:
    def __init__(self):
        self.adapters = {
            "retinexformer": RetinexformerAdapter(),
            "sci": SCIAdapter(),
            "zero_dce": ZeroDCEAdapter(),
        }
        if get_settings().allow_mock_models:
            self.adapters["mock_model"] = MockModelAdapter()

    def list(self):
        return [a.get_status().model_dump() for a in self.adapters.values()]

    def get(self, model_id: str):
        if model_id not in self.adapters:
            raise KeyError(model_id)
        return self.adapters[model_id]

    def weights(self, model_id: str):
        return self.get(model_id).discover_weights()

    def refresh(self, model_id: str):
        adapter = self.get(model_id)
        return adapter.get_status().model_dump()

    def available(self):
        return [a for a in self.adapters.values() if a.get_status().available]
