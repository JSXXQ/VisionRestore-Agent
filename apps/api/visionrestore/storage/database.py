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

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

class Database:
    def __init__(self):
        settings = get_settings()
        Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(settings.database_url, future=True)
        metadata.create_all(self.engine)

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
