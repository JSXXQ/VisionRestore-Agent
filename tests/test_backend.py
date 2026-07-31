from visionrestore.backends.onnx_backend import ONNXRuntimeBackend
from visionrestore.backends.tensorrt_backend import TensorRTBackend

def test_reserved_backends_are_not_claimed_available():
    assert ONNXRuntimeBackend().status()["available"] is False
    assert TensorRTBackend().status()["available"] is False
