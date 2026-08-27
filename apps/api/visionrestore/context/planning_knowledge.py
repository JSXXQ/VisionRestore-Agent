from __future__ import annotations

from collections.abc import Iterable

from visionrestore.schemas.ai import RetrievedContext
from visionrestore.schemas.candidate import CandidatePlanItem


class PlanningKnowledgeAdapter:
    """Backward-compatible no-op for persisted V2.2 integrations.

    Active planning uses Knowledge as the LLM model-definition reference.
    It intentionally does not produce an independent score adjustment.
    """

    def apply(
        self,
        candidates: list[CandidatePlanItem],
        retrieved_context: Iterable[RetrievedContext | dict] | None,
    ) -> list[CandidatePlanItem]:
        del retrieved_context
        for candidate in candidates:
            candidate.knowledge_adjustment = 0
            candidate.knowledge_evidence = []
        return candidates
