from visionrestore.adapters.registry import ModelRegistry


def test_model_status_is_honest():
    models = ModelRegistry().list()
    ids = {m["model_id"] for m in models}
    assert {"zero_dce", "sci", "retinexformer"}.issubset(ids)
    assert "snr_aware" not in ids
    assert all("status_message" in m for m in models)
    assert any(m["capabilities"].get("weights") for m in models if m["model_id"] == "retinexformer")
