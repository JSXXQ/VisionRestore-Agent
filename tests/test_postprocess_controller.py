from types import SimpleNamespace

from visionrestore.schemas.postprocess import PostprocessDecision
from visionrestore.services.postprocess_controller import PostprocessController


class FakeStatus:
    def __init__(self, available, message="status"):
        self.available = available
        self.status_message = message


class FakeAdapter:
    def __init__(self, available):
        self.available = available

    def get_status(self):
        return FakeStatus(self.available, "ready" if self.available else "not ready")


class FakeRegistry:
    def __init__(self, available=False):
        self.available = available

    def get(self, model_id):
        return FakeAdapter(self.available)


def _task():
    return SimpleNamespace(task_id="task-1")


def test_postprocess_controller_skip_denoise_moves_to_sr_confirmation():
    result = PostprocessController(FakeRegistry()).decide(task=_task(), decision=PostprocessDecision(operation="denoise", decision="skip"))
    assert result.accepted is True
    assert result.executed is False
    assert result.next_status == "awaiting_sr_confirmation"


def test_postprocess_controller_accept_unready_model_does_not_execute():
    result = PostprocessController(FakeRegistry(False)).decide(task=_task(), decision=PostprocessDecision(operation="denoise", decision="accept", model_id="lpdm"))
    assert result.accepted is True
    assert result.executed is False
    assert result.next_status == "awaiting_denoise_confirmation"
    assert "不伪造" in result.message


def test_postprocess_controller_rejects_unknown_operation():
    result = PostprocessController(FakeRegistry()).decide(task=_task(), decision=PostprocessDecision(operation="bad", decision="accept"))
    assert result.accepted is False
