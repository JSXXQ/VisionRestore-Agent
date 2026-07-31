from pathlib import Path
from functools import lru_cache
import hashlib
import yaml
from visionrestore.core.config import PROJECT_ROOT

CONFIG_DIR = PROJECT_ROOT / "config"
LOCAL_MODELS = CONFIG_DIR / "models.local.yaml"
ROUTING_RULES = CONFIG_DIR / "routing_rules.yaml"


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


@lru_cache
def get_model_config() -> dict:
    return _read_yaml(LOCAL_MODELS)


@lru_cache
def get_routing_rules() -> dict:
    return _read_yaml(ROUTING_RULES)


def external_python() -> str:
    return get_model_config().get("external_python", "python")


def file_sha256(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_size(path: str | Path) -> int | None:
    p = Path(path)
    return p.stat().st_size if p.exists() and p.is_file() else None
