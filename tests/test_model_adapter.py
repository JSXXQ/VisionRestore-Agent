from visionrestore.adapters.registry import ModelRegistry


def test_model_status_is_honest():
    models = ModelRegistry().list()
    ids = {m["model_id"] for m in models}
    assert {"zero_dce", "sci", "retinexformer"}.issubset(ids)
    assert {"darkir", "hvi_cidnet", "flol", "lpdm", "mambair"}.issubset(ids)
    assert "snr_aware" not in ids
    assert all("status_message" in m for m in models)
    assert any(m["capabilities"].get("weights") for m in models if m["model_id"] == "retinexformer")


def test_new_worker_models_are_visible_but_not_ready_without_health_check():
    models = {m["model_id"]: m for m in ModelRegistry().list()}
    for model_id in ["darkir", "hvi_cidnet", "flol", "lpdm", "mambair"]:
        item = models[model_id]
        assert item["available"] is False
        assert item["capabilities"]["execution_backend"] == "subprocess"
        assert item["capabilities"]["ready_requires_real_small_image_inference"] is True
        assert item["capabilities"]["auto_route"] is False


def test_registry_groups_enhancement_and_postprocess_models():
    registry = ModelRegistry()
    assert registry.enhancement_model_ids() == ["retinexformer", "darkir", "hvi_cidnet", "flol", "sci", "zero_dce"]
    assert registry.postprocess_model_ids() == ["lpdm", "mambair"]
