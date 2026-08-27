import pytest

from visionrestore.storage.database import Database


def test_database_settings_roundtrip():
    db = Database()
    db.set_settings({"x": {"y": 1}})
    assert db.get_settings()["x"]["y"] == 1


def test_database_records_v2_schema_version():
    db = Database()
    assert "v2_foundation_2026_08_01" in db.list_schema_versions()


def test_database_v2_entity_roundtrip():
    db = Database()
    db.put_entity("candidate_plan", "plan-1", {"items": [1]}, task_id="task-1")
    assert db.get_entity("candidate_plan", "plan-1")["items"] == [1]
    assert db.list_entities("candidate_plan", task_id="task-1")[0]["items"] == [1]


def test_database_agent_context_entity_roundtrip():
    db = Database()
    db.put_entity("retrieved_context", "task-1:latest", [{"title": "guardrail"}], task_id="task-1")
    db.put_entity("region_constraint", "task-1:latest", [{"target": "streetlight"}], task_id="task-1")
    assert db.get_entity("retrieved_context", "task-1:latest")[0]["title"] == "guardrail"
    assert db.get_entity("region_constraint", "task-1:latest")[0]["target"] == "streetlight"


def test_database_rejects_unknown_entity_type():
    db = Database()
    with pytest.raises(KeyError):
        db.put_entity("unknown", "x", {})
