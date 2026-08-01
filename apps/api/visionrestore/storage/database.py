import json
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import create_engine, MetaData, Table, Column, String, Text, Integer
from sqlalchemy import select, insert, update, delete
from visionrestore.core.config import get_settings

metadata = MetaData()
files = Table(
    "files", metadata,
    Column("file_id", String, primary_key=True),
    Column("payload", Text, nullable=False),
    Column("created_at", String, nullable=False),
)
tasks = Table(
    "tasks", metadata,
    Column("task_id", String, primary_key=True),
    Column("payload", Text, nullable=False),
    Column("created_at", String, nullable=False),
)
settings_table = Table(
    "settings", metadata,
    Column("key", String, primary_key=True),
    Column("payload", Text, nullable=False),
    Column("updated_at", String, nullable=False),
)
schema_migrations = Table(
    "schema_migrations", metadata,
    Column("version", String, primary_key=True),
    Column("applied_at", String, nullable=False),
)


def _entity_table(name: str) -> Table:
    return Table(
        name, metadata,
        Column("record_id", String, primary_key=True),
        Column("task_id", String, nullable=True, index=True),
        Column("payload", Text, nullable=False),
        Column("created_at", String, nullable=False),
    )


candidate_plans = _entity_table("candidate_plans")
candidate_results = _entity_table("candidate_results")
candidate_metrics = _entity_table("candidate_metrics")
candidate_rankings = _entity_table("candidate_rankings")
postprocess_recommendations = _entity_table("postprocess_recommendations")
postprocess_decisions = _entity_table("postprocess_decisions")
postprocess_results = _entity_table("postprocess_results")
artifact_lineages = _entity_table("artifact_lineages")
model_health_records = _entity_table("model_health_records")
ai_analysis_records = _entity_table("ai_analysis_records")

ENTITY_TABLES = {
    "candidate_plan": candidate_plans,
    "candidate_result": candidate_results,
    "candidate_metric": candidate_metrics,
    "candidate_ranking": candidate_rankings,
    "postprocess_recommendation": postprocess_recommendations,
    "postprocess_decision": postprocess_decisions,
    "postprocess_result": postprocess_results,
    "artifact_lineage": artifact_lineages,
    "model_health_record": model_health_records,
    "ai_analysis_record": ai_analysis_records,
}

SCHEMA_VERSION = "v2_foundation_2026_08_01"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self):
        settings = get_settings()
        Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(settings.database_url, future=True)
        metadata.create_all(self.engine)
        self._record_schema_version()

    def _record_schema_version(self):
        with self.engine.begin() as conn:
            row = conn.execute(select(schema_migrations.c.version).where(schema_migrations.c.version == SCHEMA_VERSION)).first()
            if not row:
                conn.execute(insert(schema_migrations).values(version=SCHEMA_VERSION, applied_at=utc_now()))

    def put_file(self, file_id: str, payload: dict):
        with self.engine.begin() as conn:
            conn.execute(insert(files).values(file_id=file_id, payload=json.dumps(payload), created_at=utc_now()))

    def get_file(self, file_id: str) -> dict | None:
        with self.engine.begin() as conn:
            row = conn.execute(select(files.c.payload).where(files.c.file_id == file_id)).first()
        return json.loads(row[0]) if row else None

    def put_task(self, task_id: str, payload: dict):
        data = json.dumps(payload)
        with self.engine.begin() as conn:
            row = conn.execute(select(tasks.c.task_id).where(tasks.c.task_id == task_id)).first()
            if row:
                conn.execute(update(tasks).where(tasks.c.task_id == task_id).values(payload=data))
            else:
                conn.execute(insert(tasks).values(task_id=task_id, payload=data, created_at=utc_now()))

    def get_task(self, task_id: str) -> dict | None:
        with self.engine.begin() as conn:
            row = conn.execute(select(tasks.c.payload).where(tasks.c.task_id == task_id)).first()
        return json.loads(row[0]) if row else None

    def list_tasks(self) -> list[dict]:
        with self.engine.begin() as conn:
            rows = conn.execute(select(tasks.c.payload).order_by(tasks.c.created_at.desc())).all()
        return [json.loads(row[0]) for row in rows]

    def delete_task(self, task_id: str) -> bool:
        with self.engine.begin() as conn:
            result = conn.execute(delete(tasks).where(tasks.c.task_id == task_id))
        return result.rowcount > 0

    def get_settings(self) -> dict:
        with self.engine.begin() as conn:
            rows = conn.execute(select(settings_table.c.key, settings_table.c.payload)).all()
        return {key: json.loads(payload) for key, payload in rows}

    def set_settings(self, payload: dict):
        with self.engine.begin() as conn:
            for key, value in payload.items():
                row = conn.execute(select(settings_table.c.key).where(settings_table.c.key == key)).first()
                data = json.dumps(value)
                if row:
                    conn.execute(update(settings_table).where(settings_table.c.key == key).values(payload=data, updated_at=utc_now()))
                else:
                    conn.execute(insert(settings_table).values(key=key, payload=data, updated_at=utc_now()))

    def put_entity(self, entity_type: str, record_id: str, payload: dict, task_id: str | None = None):
        table = self._entity_table(entity_type)
        data = json.dumps(payload)
        with self.engine.begin() as conn:
            row = conn.execute(select(table.c.record_id).where(table.c.record_id == record_id)).first()
            if row:
                conn.execute(update(table).where(table.c.record_id == record_id).values(task_id=task_id, payload=data))
            else:
                conn.execute(insert(table).values(record_id=record_id, task_id=task_id, payload=data, created_at=utc_now()))

    def get_entity(self, entity_type: str, record_id: str) -> dict | None:
        table = self._entity_table(entity_type)
        with self.engine.begin() as conn:
            row = conn.execute(select(table.c.payload).where(table.c.record_id == record_id)).first()
        return json.loads(row[0]) if row else None

    def list_entities(self, entity_type: str, task_id: str | None = None) -> list[dict]:
        table = self._entity_table(entity_type)
        statement = select(table.c.payload).order_by(table.c.created_at.asc())
        if task_id is not None:
            statement = statement.where(table.c.task_id == task_id)
        with self.engine.begin() as conn:
            rows = conn.execute(statement).all()
        return [json.loads(row[0]) for row in rows]

    def list_schema_versions(self) -> list[str]:
        with self.engine.begin() as conn:
            rows = conn.execute(select(schema_migrations.c.version).order_by(schema_migrations.c.applied_at.asc())).all()
        return [row[0] for row in rows]

    def _entity_table(self, entity_type: str) -> Table:
        if entity_type not in ENTITY_TABLES:
            raise KeyError(f"unknown entity type: {entity_type}")
        return ENTITY_TABLES[entity_type]
