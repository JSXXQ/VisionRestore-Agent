from types import SimpleNamespace

from visionrestore.context import ContextRetrievalService


def test_context_retrieval_returns_allowlisted_model_context():
    service = ContextRetrievalService()
    metrics = SimpleNamespace(
        model_dump=lambda: {
            "noise_estimate": 18,
            "laplacian_sharpness": 60,
            "color_cast_index": 0.18,
            "width": 3024,
            "height": 4032,
        }
    )
    results = service.retrieve(
        user_request="quality first, keep natural color and reduce noise",
        image_metrics=metrics,
        available_models=[
            {"model_id": "darkir", "available": True},
            {"model_id": "hvi_cidnet", "available": True},
            {"model_id": "flol", "available": True},
        ],
        hardware_summary={"gpu_memory_mb": 3072},
        max_items=5,
    )

    assert results
    titles = " ".join(item.title.lower() for item in results)
    assert "darkir" in titles or "hvi-cidnet" in titles or "flol" in titles
    assert all(item.item_id for item in results)
    assert any(item.matched_terms for item in results)
    assert all("api_key" not in item.content.lower() for item in results)
    assert all("bearer " not in item.content.lower() for item in results)


def test_prompt_context_marks_external_send_policy():
    service = ContextRetrievalService()
    context = service.build_prompt_context(
        user_request="outdoor night road, quality first",
        image_metrics={"noise_estimate": 15},
        available_models=[{"model_id": "darkir", "available": True}],
        hardware_summary={"gpu_memory_mb": 8192},
    )

    assert context["enabled"] is True
    assert "advisory only" in context["policy"]
    assert "override" in context["policy"]


def test_model_reference_returns_one_standard_per_available_family():
    service = ContextRetrievalService()
    references = service.model_reference(
        [
            {"model_id": "retinexformer", "available": True},
            {"model_id": "darkir", "available": True},
            {"model_id": "hvi_cidnet", "available": False},
        ]
    )

    matched = {item.matched_terms[0] for item in references}
    assert matched == {"retinexformer", "darkir"}
    assert all(item.source == "model_roles" for item in references)
    assert all(item.score == 0 for item in references)
