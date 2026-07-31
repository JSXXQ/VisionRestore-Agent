from .base import ModelAdapter

class RetinexformerAdapter(ModelAdapter):
    model_id = "retinexformer"
    display_name = "Retinexformer"
    description = "高质量主模型，支持 LOL-v2-real、SDSD-indoor、SDSD-outdoor、NTIRE 多领域权重。"
    repository_url = "https://github.com/caiyuanhao1998/Retinexformer"
    license_name = "MIT License in local source"
    supported_devices = ["cuda"]
    supported_precisions = ["fp32", "fp16"]
    size_multiple = 4
    supports_tiling = True
    default_checkpoint = "lol_v2_real"

