import numpy as np
from PIL import Image
from visionrestore.services.evaluator import QualityEvaluator

def test_quality_evaluator_scores(tmp_path):
    a = tmp_path / "a.png"; b = tmp_path / "b.png"
    Image.fromarray(np.full((32,32,3), 20, dtype=np.uint8)).save(a)
    Image.fromarray(np.full((32,32,3), 90, dtype=np.uint8)).save(b)
    metrics = QualityEvaluator().evaluate(str(a), str(b))
    assert metrics["score"] > 0


def test_target_brightness_penalizes_overbright_output(tmp_path):
    source = tmp_path / "source.png"
    natural = tmp_path / "natural.png"
    overbright = tmp_path / "overbright.png"
    Image.fromarray(np.full((32, 32, 3), 15, dtype=np.uint8)).save(source)
    Image.fromarray(np.full((32, 32, 3), 95, dtype=np.uint8)).save(natural)
    Image.fromarray(np.full((32, 32, 3), 180, dtype=np.uint8)).save(overbright)

    natural_metrics = QualityEvaluator().evaluate(str(source), str(natural))
    overbright_metrics = QualityEvaluator().evaluate(str(source), str(overbright))

    assert natural_metrics["components"]["brightness_target_fit"] > overbright_metrics["components"]["brightness_target_fit"]
    assert natural_metrics["components"]["shadow_recovery"] > overbright_metrics["components"]["shadow_recovery"]
