class TensorRTBackend:
    backend_id = "tensorrt"
    implemented = False

    def status(self):
        return {"available": False, "reason": "本阶段仅保留接口，未转换和验证的模型不会显示为支持 TensorRT。"}
