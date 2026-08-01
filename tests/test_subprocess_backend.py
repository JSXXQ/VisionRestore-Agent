import json
import sys

import pytest

from visionrestore.backends.subprocess_backend import SubprocessBackend, SubprocessModelRuntime
from visionrestore.schemas.worker import WorkerRequest


def _write_worker(path):
    path.write_text(
        "\n".join(
            [
                "import argparse, json",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--request')",
                "parser.add_argument('--response')",
                "args = parser.parse_args()",
                "req = json.load(open(args.request, encoding='utf-8'))",
                "json.dump({",
                "  'success': True,",
                "  'model_id': 'fake_model',",
                "  'checkpoint_id': req.get('checkpoint_id', ''),",
                "  'is_mock': False,",
                "  'output_path': req.get('output_path'),",
                "  'runtime_ms': 7,",
                "  'peak_memory_mb': 12.5,",
                "  'warnings': [],",
                "  'metadata': {'worker_seen_output': req.get('output_path')}",
                "}, open(args.response, 'w', encoding='utf-8'))",
            ]
        ),
        encoding="utf-8",
    )


def test_subprocess_backend_runs_worker_protocol(tmp_path):
    worker = tmp_path / "worker.py"
    _write_worker(worker)
    runtime = SubprocessModelRuntime(
        model_id="fake_model",
        source_path=str(tmp_path),
        python_executable=sys.executable,
        worker_script=str(worker),
        environment_name="pytest",
        timeout_seconds=10,
    )
    backend = SubprocessBackend(runtime, task_root=tmp_path / "tasks")
    request = WorkerRequest(
        request_id="case-1",
        operation="enhance",
        input_paths=[str(tmp_path / "input.png")],
        output_path="out.png",
        checkpoint_id="ckpt",
        device="cuda:0",
        precision="fp16",
    )
    response = backend.run(request)
    assert response.success is True
    assert response.model_id == "fake_model"
    assert response.checkpoint_id == "ckpt"
    assert response.is_mock is False
    assert response.metadata["execution_backend"] == "subprocess"
    saved_request = json.loads((tmp_path / "tasks" / "case-1" / "request.json").read_text(encoding="utf-8"))
    assert saved_request["output_path"].endswith("out.png")


def test_subprocess_backend_rejects_output_outside_task_dir(tmp_path):
    worker = tmp_path / "worker.py"
    _write_worker(worker)
    backend = SubprocessBackend(
        SubprocessModelRuntime("fake_model", str(tmp_path), sys.executable, str(worker)),
        task_root=tmp_path / "tasks",
    )
    with pytest.raises(ValueError):
        backend.run(WorkerRequest(request_id="case-2", operation="enhance", output_path=str(tmp_path / "escape.png")))


def test_model_runtime_config_supports_legacy_weights(monkeypatch):
    from visionrestore.core import model_config

    monkeypatch.setattr(
        model_config,
        "get_model_config",
        lambda: {
            "external_python": "python-global",
            "models": {
                "retinexformer": {
                    "source_path": "src",
                    "execution_backend": "subprocess",
                    "weights": {"lol_v2_real": {"path": "a.pth", "default": True}},
                }
            },
        },
    )
    cfg = model_config.get_model_runtime_config("retinexformer")
    assert cfg.python_executable == "python-global"
    assert cfg.weight_profiles[0].checkpoint_id == "lol_v2_real"
    assert cfg.weight_profiles[0].default is True
