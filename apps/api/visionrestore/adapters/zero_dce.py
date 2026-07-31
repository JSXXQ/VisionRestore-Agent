from .base import ModelAdapter

class ZeroDCEAdapter(ModelAdapter):
    model_id = "zero_dce"
    display_name = "Zero-DCE"
    description = "经典轻量基线与最后兜底模型；不作为自动模式第一优先主模型。"
    repository_url = "https://github.com/Li-Chongyi/Zero-DCE"
    license_name = "Non-commercial research notice in project README; verify before commercial use"
    supported_devices = ["cuda"]
    supported_precisions = ["fp32"]
    size_multiple = 1
    supports_tiling = False
    default_checkpoint = "epoch99"
