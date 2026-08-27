from pathlib import Path
from types import SimpleNamespace

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from visionrestore.agent.intent_parser import IntentParser
from visionrestore.graph.builder import build_visionrestore_graph
from visionrestore.graph.runtime import VisionRestoreGraphRuntime
from visionrestore.schemas.ai import MultimodalAnalysisResult
from visionrestore.schemas.candidate import CandidateExecutionPlan, CandidatePlanItem
from visionrestore.schemas.common import now_iso
from visionrestore.schemas.image import ImageAnalysisResult
from visionrestore.schemas.postprocess import PostprocessRecommendation, PostprocessResult
from visionrestore.schemas.selection import ResultSelection
from visionrestore.schemas.task import CandidateResult, TaskRecord


class FakeDatabase:
    def __init__(self):
        self.entities = {}
        self.tasks = {}

    def put_entity(self, entity_type, record_id, payload, task_id=None):
        self.entities[(entity_type, record_id)] = payload

    def get_entity(self, entity_type, record_id):
        return self.entities.get((entity_type, record_id))

    def get_task(self, task_id):
        return self.tasks.get(task_id)


class FakeAnalyzer:
    def analyze(self, _image_path):
        return ImageAnalysisResult(
            width=64,
            height=64,
            channels=3,
            format="png",
            bit_depth=8,
            mean_luminance=20,
            median_luminance=20,
            luminance_p05=10,
            luminance_p25=15,
            luminance_p75=30,
            luminance_p95=40,
            dark_pixel_ratio=0.8,
            bright_pixel_ratio=0,
            overexposed_pixel_ratio=0,
            rgb_means=[20, 20, 20],
            color_cast_index=0,
            color_cast_label="none",
            dynamic_range=30,
            rms_contrast=5,
            laplacian_sharpness=30,
            noise_estimate=8,
            local_luminance_non_uniformity=0.1,
            suggest_tile_inference=False,
            estimated_memory_mb=1,
        )


class FakePlanner:
    def plan(self, *, image_id, intent, analysis, hardware, model_statuses, mode, semantic_analysis, retrieved_context=None):
        candidates = [
            CandidatePlanItem(
                candidate_id="candidate_a",
                model_id="sci",
                checkpoint_id="medium",
                planning_score=91,
                reason=["fast candidate"],
            ),
            CandidatePlanItem(
                candidate_id="candidate_b",
                model_id="zero_dce",
                checkpoint_id="epoch99",
                planning_score=82,
                reason=["complementary candidate"],
            ),
        ]
        return CandidateExecutionPlan(
            task_mode="multi_candidate",
            mode=mode,
            priority=intent.priority,
            max_candidates=len(candidates),
            input_image_id=image_id,
            candidates=candidates,
        )


class FakeExecutor:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.inputs = []

    def execute(self, *, plan, input_path, priority, task_dir, cancel_check, on_event, region_constraints):
        candidate = plan.candidates[0]
        self.inputs.append(input_path)
        output_path = self.output_dir / f"{candidate.candidate_id}.png"
        output_path.write_bytes(b"candidate")
        score = 80 if candidate.candidate_id == "candidate_a" else 70
        return [
            CandidateResult(
                candidate_id=candidate.candidate_id,
                model_id=candidate.model_id,
                checkpoint_id=candidate.checkpoint_id,
                status="completed",
                planning_score=candidate.planning_score,
                score=score,
                output_path=str(output_path),
                metrics={"score": score, "noise_estimate_after": 9, "width": 64, "height": 64},
                input_path=input_path,
            )
        ]


class FakeSelector:
    def select(self, candidates, priority):
        ranked = sorted(candidates, key=lambda item: item["metrics"]["score"], reverse=True)
        successful = [
            {
                "candidate_id": item["candidate_id"],
                "model_id": item["model_id"],
                "checkpoint_id": item["checkpoint_id"],
                "score": item["metrics"]["score"],
                "layers": {"technical": item["metrics"]["score"]},
                "reasons": ["real output score"],
                "valid": True,
                "eliminated": False,
            }
            for item in ranked
        ]
        return ResultSelection(selected=successful[0], successful=successful, failed=[], message="selected")


class FakeResidual:
    def __init__(self, recommend_denoise):
        self.recommend_denoise = recommend_denoise

    def analyze(self, **_kwargs):
        return PostprocessRecommendation(
            denoise_recommended=self.recommend_denoise,
            denoise_confidence=0.9 if self.recommend_denoise else 0,
            denoise_reason=["residual noise"] if self.recommend_denoise else [],
            preferred_denoiser="nafnet",
        )


class FakePostprocess:
    def decide(self, *, task, decision):
        return PostprocessResult(
            task_id=task.task_id,
            operation=decision.operation,
            decision=decision.decision,
            accepted=decision.decision != "skip",
            executed=False,
            next_status="completed",
            message="postprocess decision recorded",
        )


def _build_runtime(tmp_path, monkeypatch, *, recommend_denoise=False):
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"input")
    monkeypatch.setattr("visionrestore.graph.runtime.resolve_registered_path", lambda _relative: input_path)

    database = FakeDatabase()
    executor = FakeExecutor(tmp_path)
    dependencies = SimpleNamespace(
        settings=SimpleNamespace(report_dir=tmp_path / "reports"),
        database=database,
        registry=SimpleNamespace(list=lambda: [{"model_id": "sci"}, {"model_id": "zero_dce"}]),
        analyzer=FakeAnalyzer(),
        intent_parser=IntentParser(),
        hardware=SimpleNamespace(inspect=lambda: {"cuda_available": False, "gpu_memory_mb": 0}),
        context_retrieval=SimpleNamespace(retrieve=lambda **_kwargs: []),
        semantic_advisory=SimpleNamespace(
            analyze=lambda **_kwargs: MultimodalAnalysisResult(
                provider="disabled",
                validation_passed=True,
                rejection_reason="NO_EXTERNAL_MULTIMODAL_ADVICE",
            )
        ),
        planner=FakePlanner(),
        selector=FakeSelector(),
        residual=FakeResidual(recommend_denoise),
        region_constraints=SimpleNamespace(build_constraints=lambda **_kwargs: []),
        lineage=SimpleNamespace(create_task_layout=lambda _task_id: tmp_path),
        report=SimpleNamespace(write_reports=lambda _payload, path: path.parent.mkdir(parents=True, exist_ok=True)),
        postprocess=FakePostprocess(),
        executor=executor,
    )
    saved = {}

    def save_task(task):
        payload = task.model_dump(mode="json")
        saved[task.task_id] = payload
        database.tasks[task.task_id] = payload

    runtime = VisionRestoreGraphRuntime(save_task=save_task, dependencies=dependencies)
    task = TaskRecord(
        task_id="graph-task",
        status="queued",
        image_id="image-1",
        user_goal="自然增强并保护高光",
        mode="auto",
        priority="balanced",
        created_at=now_iso(),
    )
    return runtime, task, saved, executor


def test_langgraph_multi_candidate_flow_uses_original_input_and_finishes(tmp_path, monkeypatch):
    runtime, task, saved, executor = _build_runtime(tmp_path, monkeypatch)
    graph = build_visionrestore_graph(runtime, checkpointer=InMemorySaver())
    result = graph.invoke(
        runtime.initial_state(task, {"relative_path": "registered/input.png"}),
        config={"configurable": {"thread_id": task.task_id}, "max_concurrency": 1},
    )

    assert result["terminal_status"] == "completed"
    assert saved[task.task_id]["workflow_engine"] == "langgraph"
    assert saved[task.task_id]["best_result"]["candidate_id"] == "candidate_a"
    assert len(saved[task.task_id]["candidates"]) == 2
    assert saved[task.task_id]["candidates"][0]["planning_score"] is not None
    assert saved[task.task_id]["candidates"][0]["final_score"] is not None
    assert len(set(executor.inputs)) == 1
    assert executor.inputs == [str(tmp_path / "input.png"), str(tmp_path / "input.png")]


def test_langgraph_postprocess_interrupt_resume_is_idempotent(tmp_path, monkeypatch):
    runtime, task, saved, executor = _build_runtime(tmp_path, monkeypatch, recommend_denoise=True)
    graph = build_visionrestore_graph(runtime, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": task.task_id}, "max_concurrency": 1}
    paused = graph.invoke(runtime.initial_state(task, {"relative_path": "registered/input.png"}), config=config)

    assert paused.get("__interrupt__")
    assert saved[task.task_id]["status"] == "awaiting_denoise_confirmation"
    assert saved[task.task_id]["pending_confirmation"]["operation"] == "denoise"
    executed_before_resume = len(executor.inputs)

    resumed = graph.invoke(
        Command(resume={"operation": "denoise", "decision": "skip", "model_id": "nafnet"}),
        config=config,
    )

    assert resumed["terminal_status"] == "completed"
    assert len(executor.inputs) == executed_before_resume
    assert saved[task.task_id]["pending_confirmation"] is None
    assert resumed["last_postprocess_result"]["decision"] == "skip"


def test_empty_retry_plan_keeps_valid_first_pass_results(tmp_path, monkeypatch):
    runtime, _task, _saved, _executor = _build_runtime(tmp_path, monkeypatch)

    route = runtime.dispatch_candidates(
        {
            "pending_candidates": [],
            "candidate_results": [{"candidate_id": "first_pass", "status": "completed"}],
        }
    )

    assert route == "rank_candidates"
