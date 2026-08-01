from pathlib import Path

import numpy as np
from PIL import Image

from visionrestore.adapters.base import EnhancementResult
from visionrestore.schemas.candidate import CandidateExecutionPlan, CandidatePlanItem
from visionrestore.services.multi_candidate_executor import MultiCandidateExecutor


class FakeAdapter:
    def __init__(self, model_id, fail=False, is_mock=False):
        self.model_id = model_id
        self.fail = fail
        self.is_mock = is_mock
        self.inputs = []

    def enhance(self, image_path, output_path, device, precision, parameters):
        self.inputs.append(image_path)
        if self.fail:
            raise RuntimeError("planned failure")
        Image.fromarray(np.full((16, 16, 3), 80, dtype=np.uint8)).save(output_path)
        return EnhancementResult(
            output_path=output_path,
            parameters=parameters,
            runtime_ms=10,
            peak_memory_mb=1,
            is_mock=self.is_mock,
            adapter_class="FakeAdapter",
            model_id=self.model_id,
            checkpoint_id=parameters.get("checkpoint_id"),
            device=device,
            precision=precision,
        )


class FakeRegistry:
    def __init__(self, adapters):
        self.adapters = adapters

    def get(self, model_id):
        return self.adapters[model_id]


class FakeEvaluator:
    def evaluate(self, input_path, output_path, priority, runtime_ms, peak_memory_mb):
        return {"score": 77, "runtime_ms": runtime_ms, "peak_memory_mb": peak_memory_mb}


def _plan(*models):
    return CandidateExecutionPlan(
        mode="auto",
        priority="balanced",
        max_candidates=len(models),
        input_image_id="img",
        candidates=[CandidatePlanItem(candidate_id=f"candidate_{i:02d}", model_id=model, checkpoint_id="ckpt", reason="test") for i, model in enumerate(models, start=1)],
    )


def _input(tmp_path):
    path = tmp_path / "input.png"
    Image.fromarray(np.full((16, 16, 3), 20, dtype=np.uint8)).save(path)
    return path


def test_multi_candidate_executor_uses_same_original_input(tmp_path):
    a = FakeAdapter("a")
    b = FakeAdapter("b")
    executor = MultiCandidateExecutor(FakeRegistry({"a": a, "b": b}), FakeEvaluator())
    input_path = _input(tmp_path)
    results = executor.execute(plan=_plan("a", "b"), input_path=str(input_path), task_dir=tmp_path / "task")
    assert [item.status for item in results] == ["completed", "completed"]
    assert a.inputs == [str(input_path)]
    assert b.inputs == [str(input_path)]
    assert all(item.metrics["input_policy"] == "original_input_only" for item in results)


def test_multi_candidate_executor_keeps_running_after_candidate_failure(tmp_path):
    bad = FakeAdapter("bad", fail=True)
    good = FakeAdapter("good")
    executor = MultiCandidateExecutor(FakeRegistry({"bad": bad, "good": good}), FakeEvaluator())
    results = executor.execute(plan=_plan("bad", "good"), input_path=str(_input(tmp_path)), task_dir=tmp_path / "task")
    assert results[0].status == "failed"
    assert results[1].status == "completed"


def test_multi_candidate_executor_rejects_mock_result(tmp_path):
    mock = FakeAdapter("mock", is_mock=True)
    executor = MultiCandidateExecutor(FakeRegistry({"mock": mock}), FakeEvaluator())
    results = executor.execute(plan=_plan("mock"), input_path=str(_input(tmp_path)), task_dir=tmp_path / "task")
    assert results[0].status == "failed"
    assert "Mock" in results[0].error
