from visionrestore.adapters.registry import ModelRegistry


def test_model_status_is_honest():
    models = ModelRegistry().list()
    ids = {m["model_id"] for m in models}
    assert {"zero_dce", "sci", "retinexformer"}.issubset(ids)
    assert {"darkir", "hvi_cidnet", "flol", "lpdm", "nafnet", "realesrgan"}.issubset(ids)
    assert "snr_aware" not in ids
    assert all("status_message" in m for m in models)
    assert any(m["capabilities"].get("weights") for m in models if m["model_id"] == "retinexformer")


def test_new_worker_models_require_real_health_check_before_routing():
    models = {m["model_id"]: m for m in ModelRegistry().list()}
    for model_id in ["darkir", "hvi_cidnet", "flol", "lpdm", "nafnet", "realesrgan"]:
        item = models[model_id]
        assert item["capabilities"]["execution_backend"] == "subprocess"
        assert item["capabilities"]["ready_requires_real_small_image_inference"] is True
        if item["available"]:
            assert item["capabilities"]["last_health"]["available"] is True
        else:
            assert item["capabilities"]["auto_route"] is False


def test_worker_adapter_resolves_relative_paths_to_project_root():
    from visionrestore.adapters.worker_model import WorkerModelAdapter
    from visionrestore.core.config import PROJECT_ROOT

    adapter = WorkerModelAdapter("flol")
    assert adapter._project_path("data/tasks/example/input.png") == (PROJECT_ROOT / "data/tasks/example/input.png").resolve()


def test_model_config_resolves_project_relative_paths():
    from visionrestore.core.config import PROJECT_ROOT
    from visionrestore.core.model_config import resolve_project_path

    assert resolve_project_path("weights/retinexformer/lol_v2_real.pth") == str(
        (PROJECT_ROOT / "weights/retinexformer/lol_v2_real.pth").resolve()
    )
    absolute = str((PROJECT_ROOT / "third_party/retinexformer").resolve())
    assert resolve_project_path(absolute) == absolute


def test_registry_groups_enhancement_and_postprocess_models():
    registry = ModelRegistry()
    assert registry.enhancement_model_ids() == ["retinexformer", "darkir", "hvi_cidnet", "flol", "sci", "zero_dce"]
    assert registry.postprocess_model_ids() == ["lpdm", "nafnet", "realesrgan"]
