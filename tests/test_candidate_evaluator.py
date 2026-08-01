from visionrestore.services.candidate_evaluator import CandidateEvaluator, CandidateRanker


def _candidate(candidate_id, score=80, runtime_ms=1000, status="completed", **extra):
    metrics = {
        "score": score,
        "components": {
            "shadow_recovery": score / 100,
            "highlight_protection": 0.8,
            "color_stability": 0.8,
            "noise_control": 0.75,
            "sharpness": 0.8,
            "structure": 0.85,
        },
        "runtime_ms": runtime_ms,
        "peak_memory_mb": 2000,
        "mean_luminance_after": 80,
        "overexposed_pixel_ratio_after": 0.02,
        "color_cast_index_after": 0.05,
        "structure_keep_estimate": 0.8,
    }
    metrics.update(extra)
    return {"candidate_id": candidate_id, "model_id": "retinexformer", "checkpoint_id": "lol_v2_real", "status": status, "metrics": metrics}


def test_candidate_evaluator_eliminates_invalid_output():
    score = CandidateEvaluator().score(_candidate("bad", mean_luminance_after=0), priority="quality")
    assert score.eliminated is True
    assert score.score == 0
    assert any("纯黑" in reason for reason in score.reasons)


def test_candidate_ranker_selects_best_successful_candidate():
    ranking = CandidateRanker().rank([_candidate("a", 60), _candidate("b", 90)], priority="balanced")
    assert ranking["best"].candidate_id == "b"
    assert ranking["second_best"].candidate_id == "a"
    assert ranking["eliminated"] == []


def test_candidate_ranker_flags_close_competition():
    ranking = CandidateRanker().rank([_candidate("a", 80), _candidate("b", 81)], priority="quality")
    assert ranking["close_competition"] is True
    assert "人工对比" in ranking["note"]
