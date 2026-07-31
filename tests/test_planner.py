from types import SimpleNamespace
from visionrestore.agent.planner import DeterministicPlanner

def test_quality_prefers_retinexformer_even_if_unavailable_reason_mentions_it():
    analysis = SimpleNamespace(dark_pixel_ratio=.2, noise_estimate=3, estimated_memory_mb=10, suggest_tile_inference=False)
    plan = DeterministicPlanner().plan(image_id="x", user_goal="", mode="auto", priority="quality", analysis=analysis, hardware={"cuda_available": False}, models=[])
    assert plan.selected_model == "retinexformer"
    assert "不可用" in plan.selection_reason
