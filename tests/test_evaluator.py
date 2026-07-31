import numpy as np
from PIL import Image
from visionrestore.services.evaluator import QualityEvaluator

def test_quality_evaluator_scores(tmp_path):
    a = tmp_path / "a.png"; b = tmp_path / "b.png"
    Image.fromarray(np.full((32,32,3), 20, dtype=np.uint8)).save(a)
    Image.fromarray(np.full((32,32,3), 90, dtype=np.uint8)).save(b)
    metrics = QualityEvaluator().evaluate(str(a), str(b))
    assert metrics["score"] > 0
