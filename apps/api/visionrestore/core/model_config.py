from pathlib import Path
from functools import lru_cache
import hashlib
import yaml
from pydantic import BaseModel, Field
from visionrestore.core.config import PROJECT_ROOT

CONFIG_DIR = PROJECT_ROOT / "config"
LOCAL_MODELS = CONFIG_DIR / "models.local.yaml"
ROUTING_RULES = CONFIG_DIR / "routing_rules.yaml"
SCORING_RULES = CONFIG_DIR / "scoring_rules.yaml"
POSTPROCESS_RULES = CONFIG_DIR / "postprocess_rules.yaml"
PLANNING_RULES = CONFIG_DIR / "planning_rules.yaml"

PROJECT_PATH_FIELDS = frozenset(
    {
        "background_image_path",
        "ui_reference_image_path",
        "source_path",
        "worker_script",
        "path",
        "config_path",
    }
)


class WeightProfile(BaseModel):
    checkpoint_id: str
    path: str = ""
    config_path: str | None = None
    display_name: str = ""
    domain: str = ""
    auto_route: bool = True
    default: bool = False
    metadata: dict = Field(default_factory=dict)


class ModelRuntimeConfig(BaseModel):
    model_id: str
    source_path: str = ""
    python_executable: str = "python"
    environment_name: str = ""
    execution_backend: str = "subprocess"
    worker_script: str = ""
    device: str = "cuda:0"
    precision: str = "fp16"
    timeout_seconds: int = 240
    supports_fp16: bool = True
    supports_tiling: bool = False
    supports_cpu: bool = False
    max_concurrency: int = 1
    weight_profiles: list[WeightProfile] = Field(default_factory=list)


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve_project_path(value: str) -> str:
    """Resolve a project-relative model path without changing absolute paths."""
    if not value or value.startswith("<"):
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def _resolve_config_paths(value, field_name: str | None = None):
    if isinstance(value, dict):
        return {key: _resolve_config_paths(item, key) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_config_paths(item, field_name) for item in value]
    if field_name in PROJECT_PATH_FIELDS and isinstance(value, str):
        return resolve_project_path(value)
    return value


@lru_cache
def get_model_config() -> dict:
    return _resolve_config_paths(_read_yaml(LOCAL_MODELS))


@lru_cache
def get_routing_rules() -> dict:
    return _read_yaml(ROUTING_RULES)


@lru_cache
def get_scoring_rules() -> dict:
    return _read_yaml(SCORING_RULES)


@lru_cache
def get_planning_rules() -> dict:
    return _read_yaml(PLANNING_RULES)



@lru_cache
def get_postprocess_rules() -> dict:
    return _read_yaml(POSTPROCESS_RULES)


def external_python() -> str:
    return get_model_config().get("external_python", "python")


def get_model_runtime_config(model_id: str) -> ModelRuntimeConfig:
    root = get_model_config()
    cfg = (root.get("models", {}) or {}).get(model_id, {}) or {}
    weights = cfg.get("weight_profiles") or cfg.get("weights") or {}
    profiles: list[WeightProfile] = []
    if isinstance(weights, dict):
        for checkpoint_id, item in weights.items():
            item = item or {}
            profiles.append(WeightProfile(checkpoint_id=checkpoint_id, **item))
    elif isinstance(weights, list):
        profiles = [WeightProfile.model_validate(item) for item in weights]
    return ModelRuntimeConfig(
        model_id=model_id,
        source_path=cfg.get("source_path", ""),
        python_executable=cfg.get("python_executable") or root.get("external_python", "python"),
        environment_name=cfg.get("environment_name", ""),
        execution_backend=cfg.get("execution_backend", "subprocess"),
        worker_script=cfg.get("worker_script", ""),
        device=cfg.get("device", "cuda:0"),
        precision=cfg.get("precision", "fp16"),
        timeout_seconds=int(cfg.get("timeout_seconds", 240)),
        supports_fp16=bool(cfg.get("supports_fp16", True)),
        supports_tiling=bool(cfg.get("supports_tiling", False)),
        supports_cpu=bool(cfg.get("supports_cpu", False)),
        max_concurrency=int(cfg.get("max_concurrency", 1)),
        weight_profiles=profiles,
    )


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
