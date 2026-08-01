from visionrestore.services.result_selector import ResultSelector


def _candidate(candidate_id, score=80, status="completed", **extra):
    metrics = {
        "score": score,
        "components": {"shadow_recovery": score / 100, "highlight_protection": .8, "color_stability": .8, "noise_control": .8, "sharpness": .8, "structure": .8},
        "runtime_ms": 1000,
        "peak_memory_mb": 1000,
        "mean_luminance_after": 80,
        "overexposed_pixel_ratio_after": .01,
        "color_cast_index_after": .02,
        "structure_keep_estimate": .8,
    }
    metrics.update(extra)
    return {"candidate_id": candidate_id, "model_id": candidate_id, "checkpoint_id": "ck", "status": status, "metrics": metrics}


def test_result_selector_keeps_best_second_and_failed():
    selection = ResultSelector().select([
        _candidate("a", 60),
        _candidate("b", 90),
        _candidate("bad", 0, status="failed"),
    ])
    assert selection.selected["model_id"] == "b"
    assert selection.second_best["model_id"] == "a"
    assert selection.failed[0]["model_id"] == "bad"


def test_result_selector_reports_close_competition():
    selection = ResultSelector().select([_candidate("a", 80), _candidate("b", 81)], priority="quality")
    assert selection.close_competition is True
    assert "人工对比" in selection.message


def test_result_selector_handles_all_failed():
    selection = ResultSelector().select([_candidate("bad", 0, status="failed")])
    assert selection.selected is None
    assert "没有候选" in selection.message
