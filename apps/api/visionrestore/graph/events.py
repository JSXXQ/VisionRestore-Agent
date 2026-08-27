from __future__ import annotations

from typing import Any
from uuid import uuid4

from visionrestore.schemas.common import now_iso


def make_event(
    *,
    task_id: str,
    run_id: str,
    step_id: str,
    event_type: str,
    message: str,
    status: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict:
    return {
        "event_id": str(uuid4()),
        "task_id": task_id,
        "run_id": run_id,
        "step_id": step_id,
        "event_type": event_type,
        "status": status,
        "message": message,
        "metadata": metadata or {},
        "timestamp": now_iso(),
    }


def make_message(*, role: str, content: str, source: str, step_id: str, metadata: dict | None = None) -> dict:
    return {
        "message_id": str(uuid4()),
        "role": role,
        "content": content,
        "source": source,
        "step_id": step_id,
        "metadata": metadata or {},
        "timestamp": now_iso(),
    }
