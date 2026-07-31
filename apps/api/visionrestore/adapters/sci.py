from .base import ModelAdapter

class SCIAdapter(ModelAdapter):
    model_id = "sci"
    display_name = "SCI"
    description = "轻量快速主模型，支持 easy、medium、difficult 权重。"
    repository_url = "https://github.com/vis-opt-group/SCI"
    license_name = "MIT License in local source"
    supported_devices = ["cuda"]
    supported_precisions = ["fp32"]
    size_multiple = 1
    supports_tiling = True
    default_checkpoint = "medium"
