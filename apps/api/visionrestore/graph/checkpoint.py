from __future__ import annotations

import sqlite3
from functools import lru_cache

from langgraph.checkpoint.sqlite import SqliteSaver

from visionrestore.core.config import get_settings


@lru_cache
def get_graph_checkpointer() -> SqliteSaver:
    path = get_settings().data_dir / "langgraph" / "checkpoints.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), check_same_thread=False)
    saver = SqliteSaver(connection)
    saver.setup()
    return saver
