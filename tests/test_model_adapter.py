from visionrestore.adapters.registry import ModelRegistry

def test_model_status_is_honest():
    models = ModelRegistry().list()
    ids = {m["model_id"] for m in models}
    assert {"zero_dce", "sci", "retinexformer", "snr_aware"}.issubset(ids)
    assert all("status_message" in m for m in models)
