from visionrestore.context import PlanningKnowledgeAdapter
from visionrestore.schemas.candidate import CandidatePlanItem


def _candidate() -> CandidatePlanItem:
    return CandidatePlanItem(
        candidate_id="candidate_01",
        model_id="darkir",
        checkpoint_id="real_lsrw",
        local_score=70,
        ai_semantic_bonus=2,
        hardware_adjustment=-1,
        planning_score=71,
    )


def test_planning_knowledge_ignores_identity_only_and_non_scoring_sources():
    candidate = _candidate()
    contexts = [
        {
            "item_id": "model-role-darkir",
            "source": "model_roles",
            "title": "DarkIR role",
            "content": "role",
            "tags": ["darkir"],
            "matched_terms": ["darkir"],
        },
        {
            "item_id": "workflow-boundary",
            "source": "workflow_guardrails",
            "title": "Boundary",
            "content": "guardrail",
            "tags": ["darkir", "noise"],
            "matched_terms": ["darkir", "noise"],
        },
    ]

    PlanningKnowledgeAdapter().apply([candidate], contexts)

    assert candidate.knowledge_adjustment == 0
    assert candidate.knowledge_evidence == []
    assert candidate.planning_score == 71


def test_planning_knowledge_never_adds_an_independent_score():
    candidate = _candidate()
    contexts = [
        {
            "item_id": "model-role-darkir",
            "source": "model_roles",
            "title": "DarkIR role",
            "content": "role",
            "tags": ["darkir", "noise", "blur", "night", "quality"],
            "matched_terms": ["darkir", "noise", "blur", "night", "quality"],
        },
        {
            "item_id": "darkir-eval-a",
            "source": "eval_history",
            "title": "DarkIR eval A",
            "content": "eval",
            "tags": ["darkir", "quality", "speed", "history"],
            "matched_terms": ["darkir", "quality", "speed", "history"],
        },
        {
            "item_id": "darkir-eval-b",
            "source": "eval_history",
            "title": "DarkIR eval B",
            "content": "eval",
            "tags": ["darkir", "quality", "noise", "history"],
            "matched_terms": ["darkir", "quality", "noise", "history"],
        },
    ]

    PlanningKnowledgeAdapter().apply([candidate], contexts)

    assert candidate.knowledge_adjustment == 0
    assert candidate.knowledge_evidence == []
    assert candidate.planning_score == 71
