from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from visionrestore.schemas.postprocess import PostprocessDecision
from visionrestore.schemas.task import CandidateResult
from visionrestore.services.postprocess_controller import PostprocessController


class FakeStatus:
    def __init__(self, available, message="status", model_id="nafnet"):
        self.available = available
        self.status_message = message
        checkpoint_ids = {
            "nafnet": ["sidd_width32", "sidd_width64"],
            "realesrgan": ["realesrgan_x2plus", "realesrgan_x4plus", "realesr_general_x4v3"],
        }.get(model_id, ["lpdm_lol"])
        self.capabilities = {
            "weights": [
                {
                    "checkpoint_id": checkpoint_id,
                    "status": "found",
                    "exists": True,
                    "auto_route": False,
                    "default": index == 0,
                }
                for index, checkpoint_id in enumerate(checkpoint_ids)
            ]
        }

    def model_dump(self):
        return {
            "available": self.available,
            "status_message": self.status_message,
            "capabilities": self.capabilities,
        }


class FakeAdapter:
    def __init__(self, available, model_id="nafnet"):
        self.available = available
        self.model_id = model_id

    def get_status(self):
        return FakeStatus(self.available, "ready" if self.available else "not ready", self.model_id)


class FakeRegistry:
    def __init__(self, available=False):
        self.available = available

    def get(self, model_id):
        return FakeAdapter(self.available, model_id)


class ExecutingAdapter:
    def __init__(self, model_id="nafnet"):
        self.model_id = model_id

    def get_status(self):
        return FakeStatus(True, "ready", self.model_id)

    def enhance(self, *, image_path, output_path, device, precision, parameters):
        Image.open(image_path).save(output_path)
        return SimpleNamespace(
            output_path=output_path,
            runtime_ms=25,
            peak_memory_mb=12,
            parameters=parameters,
            is_mock=False,
            adapter_class="ExecutingAdapter",
            checkpoint_path="checkpoint.pth",
            checkpoint_sha256="checkpoint-sha",
            device=device,
            precision=precision,
            input_sha256="input-sha",
            output_sha256="output-sha",
            logs=[],
        )


class ExecutingRegistry:
    def get(self, model_id):
        return ExecutingAdapter(model_id)


class FailingAdapter:
    def __init__(self, model_id="nafnet"):
        self.model_id = model_id

    def get_status(self):
        return FakeStatus(True, "ready", self.model_id)

    def enhance(self, **_kwargs):
        raise RuntimeError("dependency missing")


class FailingRegistry:
    def get(self, model_id):
        return FailingAdapter(model_id)


class NoopDatabase:
    def get_file(self, _file_id):
        return None

    def put_file(self, _file_id, _payload):
        return None


class FixedQualityEvaluator:
    def evaluate(self, *_args, **_kwargs):
        return {
            "score": 70,
            "components": {
                "shadow_recovery": 0.7,
                "highlight_protection": 0.8,
                "color_stability": 0.8,
                "noise_control": 0.7,
                "artifact_control": 0.8,
                "sharpness": 0.8,
                "structure": 0.8,
            },
            "mean_luminance_after": 70,
            "overexposed_pixel_ratio_after": 0.01,
            "color_cast_index_after": 0.02,
            "structure_keep_estimate": 0.8,
            "runtime_ms": 25,
            "peak_memory_mb": 12,
        }


class FixedFinalEvaluator:
    def score(self, *_args, **_kwargs):
        return SimpleNamespace(
            score=70,
            eliminated=False,
            layers={"image_quality": 0.7, "restoration": 0.7, "constraint": 1, "stability": 0.8},
            reasons=["fixed lower score"],
            evidence={"source": "test"},
        )


def _task(*, sr_recommended=True):
    return SimpleNamespace(
        task_id="task-1",
        status="awaiting_denoise_confirmation",
        progress=0.9,
        completed_at=None,
        postprocess_recommendation={"super_resolution_recommended": sr_recommended},
    )


def test_postprocess_controller_skip_denoise_moves_to_sr_confirmation():
    result = PostprocessController(FakeRegistry()).decide(task=_task(), decision=PostprocessDecision(operation="denoise", decision="skip"))
    assert result.accepted is True
    assert result.executed is False
    assert result.next_status == "awaiting_sr_confirmation"


def test_postprocess_controller_skip_denoise_finishes_when_sr_is_not_recommended():
    task = _task(sr_recommended=False)
    result = PostprocessController(FakeRegistry()).decide(
        task=task,
        decision=PostprocessDecision(operation="denoise", decision="skip"),
    )

    assert result.next_status == "completed"
    assert task.status == "completed"
    assert task.completed_at is not None


def test_unavailable_denoiser_keeps_best_and_advances_instead_of_looping():
    task = _task(sr_recommended=True)
    result = PostprocessController(FakeRegistry(False)).decide(
        task=task,
        decision=PostprocessDecision(
            operation="denoise",
            decision="choose_model",
            model_id="nafnet",
        ),
    )

    assert result.executed is False
    assert result.next_status == "awaiting_sr_confirmation"
    assert task.status == "awaiting_sr_confirmation"


def test_postprocess_controller_rejects_runtime_unstable_lpdm_and_recommends_nafnet():
    result = PostprocessController(FakeRegistry(False)).decide(task=_task(), decision=PostprocessDecision(operation="denoise", decision="accept", model_id="lpdm"))
    assert result.accepted is False
    assert result.executed is False
    assert result.next_status == "awaiting_denoise_confirmation"
    assert result.metadata["recommended_model"] == "nafnet"
    assert "暂停执行" in result.message


def test_postprocess_controller_rejects_unknown_operation():
    result = PostprocessController(FakeRegistry()).decide(task=_task(), decision=PostprocessDecision(operation="bad", decision="accept"))
    assert result.accepted is False


def test_postprocess_controller_rolls_back_to_main_enhancement_candidate():
    main = CandidateResult(
        model_id="retinexformer",
        checkpoint_id="lol_v2_real",
        output_file_id="out-1",
        output_url="/api/v1/files/out-1",
        status="completed",
        score=80,
    )
    post = CandidateResult(
        model_id="lpdm",
        checkpoint_id="lpdm_lol",
        output_file_id="post-1",
        output_url="/api/v1/files/post-1",
        status="completed",
        score=82,
        parameters={"role": "postprocess", "operation": "denoise", "input_file_id": "out-1"},
    )
    task = SimpleNamespace(
        task_id="task-rollback",
        status="completed",
        progress=1.0,
        completed_at=None,
        final_recommendation=None,
        candidates=[main, post],
        best_result=post,
    )

    result = PostprocessController(FakeRegistry()).rollback(task=task)

    assert result.rollback_performed is True
    assert result.executed is True
    assert task.best_result.output_file_id == "out-1"
    assert "撤回后处理" in task.final_recommendation


def test_denoise_uses_unified_final_score_and_rolls_back_when_score_drops(tmp_path, monkeypatch):
    input_path = tmp_path / "selected.png"
    Image.new("RGB", (16, 16), (60, 60, 60)).save(input_path)
    previous = CandidateResult(
        candidate_id="candidate_01",
        model_id="retinexformer",
        checkpoint_id="lol_v2_real",
        output_file_id="selected",
        status="completed",
        score=80,
        final_score=80,
    )
    task = SimpleNamespace(
        task_id="task-auto-rollback",
        image_id="original",
        status="awaiting_denoise_confirmation",
        progress=0.94,
        completed_at=None,
        priority="quality",
        best_result=previous,
        candidates=[previous],
        postprocess_history=[],
        postprocess_recommendation={"super_resolution_recommended": False},
        region_constraints=[],
        final_recommendation=None,
    )
    controller = PostprocessController(
        ExecutingRegistry(),
        evaluator=FixedQualityEvaluator(),
        database=NoopDatabase(),
        candidate_evaluator=FixedFinalEvaluator(),
    )
    controller.settings = SimpleNamespace(data_dir=Path(tmp_path), output_dir=Path(tmp_path))
    monkeypatch.setattr(controller, "_candidate_output_path", lambda *_args, **_kwargs: input_path)
    monkeypatch.setattr(controller, "_original_input_path", lambda *_args, **_kwargs: input_path)
    monkeypatch.setattr(controller, "_register_output_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "visionrestore.services.postprocess_controller.IQAService.status",
        lambda _self: {"available": False, "status_message": "disabled in test"},
    )

    result = controller.decide(
        task=task,
        decision=PostprocessDecision(
            operation="denoise",
            decision="choose_model",
            model_id="nafnet",
        ),
    )

    assert result.executed is True
    assert result.rollback_performed is True
    assert result.next_status == "completed"
    assert task.best_result.output_file_id == "selected"
    assert task.candidates[-1].final_score == 70
    assert task.candidates[-1].metrics["score_layers"]["restoration"] == 0.7
    assert task.candidates[-1].checkpoint_id == "sidd_width32"
    assert task.candidates[-1].parameters["checkpoint_selection"]["mode"] == "local_checkpoint_match"
    assert task.checkpoint_candidates[-1]["model_id"] == "nafnet"


def test_realesrgan_checkpoint_selection_respects_requested_scale(tmp_path):
    input_path = tmp_path / "selected.png"
    Image.new("RGB", (32, 32), (60, 60, 60)).save(input_path)
    task = SimpleNamespace(
        priority="balanced",
        user_goal="自然稳定的超分结果",
        hardware_info={"cuda_available": True, "gpu_memory_mb": 12288},
        ai_analysis=None,
    )
    controller = PostprocessController(ExecutingRegistry(), database=NoopDatabase())
    status = FakeStatus(True, "ready", "realesrgan")

    x2 = controller._select_postprocess_checkpoint(
        task=task,
        model_id="realesrgan",
        status=status,
        input_path=input_path,
        operation="super_resolution",
        explicit_checkpoint=None,
        scale=2,
    )
    x4 = controller._select_postprocess_checkpoint(
        task=task,
        model_id="realesrgan",
        status=status,
        input_path=input_path,
        operation="super_resolution",
        explicit_checkpoint=None,
        scale=4,
    )

    assert x2.checkpoint_id == "realesrgan_x2plus"
    assert x4.checkpoint_id in {"realesrgan_x4plus", "realesr_general_x4v3"}
    assert all(
        item["checkpoint_id"] != "realesrgan_x2plus"
        for item in x4.ranked_candidates
        if item.get("eligible", True)
    )


def test_failed_denoise_execution_keeps_best_and_finishes_without_retry_loop(tmp_path, monkeypatch):
    input_path = tmp_path / "selected.png"
    Image.new("RGB", (16, 16), (60, 60, 60)).save(input_path)
    previous = CandidateResult(
        candidate_id="candidate_01",
        model_id="retinexformer",
        checkpoint_id="lol_v2_real",
        output_file_id="selected",
        status="completed",
        score=80,
        final_score=80,
    )
    task = SimpleNamespace(
        task_id="task-denoise-failure",
        image_id="original",
        status="awaiting_denoise_confirmation",
        progress=0.94,
        completed_at=None,
        priority="quality",
        best_result=previous,
        candidates=[previous],
        postprocess_history=[],
        postprocess_recommendation={"super_resolution_recommended": False},
        region_constraints=[],
        final_recommendation=None,
    )
    controller = PostprocessController(FailingRegistry(), database=NoopDatabase())
    controller.settings = SimpleNamespace(data_dir=Path(tmp_path), output_dir=Path(tmp_path))
    monkeypatch.setattr(controller, "_candidate_output_path", lambda *_args, **_kwargs: input_path)

    result = controller.decide(
        task=task,
        decision=PostprocessDecision(
            operation="denoise",
            decision="choose_model",
            model_id="nafnet",
        ),
    )

    assert result.executed is False
    assert result.next_status == "completed"
    assert task.status == "completed"
    assert task.best_result.output_file_id == "selected"
    assert task.postprocess_history[-1]["error"] == "dependency missing"
