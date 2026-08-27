import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from visionrestore.core.config import PROJECT_ROOT, get_settings
from visionrestore.schemas.worker import WorkerRequest, WorkerResponse


@dataclass(frozen=True)
class SubprocessModelRuntime:
    model_id: str
    source_path: str
    python_executable: str
    worker_script: str
    environment_name: str = ""
    timeout_seconds: int = 240


class SubprocessBackend:
    def __init__(self, runtime: SubprocessModelRuntime, task_root: Path | None = None):
        self.runtime = runtime
        self.settings = get_settings()
        self.task_root = task_root or (self.settings.data_dir / "tasks")
        self._process: subprocess.Popen | None = None

    def health_check(self) -> dict:
        missing = []
        if not Path(self.runtime.source_path).exists():
            missing.append("source_path")
        if not Path(self.runtime.python_executable).exists():
            missing.append("python_executable")
        if not Path(self.runtime.worker_script).exists():
            missing.append("worker_script")
        return {
            "model_id": self.runtime.model_id,
            "available": not missing,
            "status": "found" if not missing else "environment_missing",
            "missing": missing,
            "environment_name": self.runtime.environment_name,
            "execution_backend": "subprocess",
        }

    def run(self, request: WorkerRequest) -> WorkerResponse:
        health = self.health_check()
        if not health["available"]:
            return WorkerResponse(
                success=False,
                model_id=self.runtime.model_id,
                checkpoint_id=request.checkpoint_id,
                error=f"Subprocess runtime is not ready: {', '.join(health['missing'])}",
                metadata=health,
            )
        task_dir = self._create_task_dir(request.request_id)
        safe_request = self._materialize_request(request, task_dir)
        request_path = task_dir / "request.json"
        response_path = task_dir / "response.json"
        stdout_path = task_dir / "stdout.log"
        stderr_path = task_dir / "stderr.log"
        request_path.write_text(safe_request.model_dump_json(indent=2), encoding="utf-8")
        cmd = [self.runtime.python_executable, self.runtime.worker_script, "--request", str(request_path), "--response", str(response_path)]
        process_env = os.environ.copy()
        local_packages = PROJECT_ROOT / "data" / "python_packages"
        if local_packages.exists():
            existing_pythonpath = process_env.get("PYTHONPATH", "")
            process_env["PYTHONPATH"] = str(local_packages) + (
                os.pathsep + existing_pythonpath if existing_pythonpath else ""
            )
        started = time.perf_counter()
        return_code: int | None = None
        try:
            self._process = subprocess.Popen(
                cmd,
                cwd=str(self.runtime.source_path),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=process_env,
            )
            stdout, stderr = self._process.communicate(timeout=self.runtime.timeout_seconds)
            return_code = self._process.returncode
            stdout_path.write_text(stdout or "", encoding="utf-8")
            stderr_path.write_text(stderr or "", encoding="utf-8")
        except subprocess.TimeoutExpired:
            self.cancel()
            return WorkerResponse(success=False, model_id=self.runtime.model_id, checkpoint_id=request.checkpoint_id, runtime_ms=int((time.perf_counter() - started) * 1000), error=f"Worker timed out after {self.runtime.timeout_seconds}s", metadata={"task_dir": str(task_dir), "execution_backend": "subprocess"})
        finally:
            self._process = None
        runtime_ms = int((time.perf_counter() - started) * 1000)
        if return_code != 0:
            return WorkerResponse(success=False, model_id=self.runtime.model_id, checkpoint_id=request.checkpoint_id, runtime_ms=runtime_ms, error=f"Worker process failed with code {return_code}", metadata={"task_dir": str(task_dir), "execution_backend": "subprocess"})
        if not response_path.exists():
            return WorkerResponse(success=False, model_id=self.runtime.model_id, checkpoint_id=request.checkpoint_id, runtime_ms=runtime_ms, error="Worker did not write response.json", metadata={"task_dir": str(task_dir), "execution_backend": "subprocess"})
        try:
            response = WorkerResponse.model_validate(json.loads(response_path.read_text(encoding="utf-8")))
        except Exception as exc:
            return WorkerResponse(success=False, model_id=self.runtime.model_id, checkpoint_id=request.checkpoint_id, runtime_ms=runtime_ms, error=f"Invalid worker response: {exc.__class__.__name__}", metadata={"task_dir": str(task_dir), "execution_backend": "subprocess"})
        response.runtime_ms = response.runtime_ms or runtime_ms
        response.metadata = {**response.metadata, "task_dir": str(task_dir), "execution_backend": "subprocess", "stdout_log": str(stdout_path), "stderr_log": str(stderr_path)}
        return response

    def cancel(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.kill()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def _create_task_dir(self, request_id: str) -> Path:
        safe_id = request_id.strip() or str(uuid4())
        task_dir = self.task_root / safe_id
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir.resolve()

    def _materialize_request(self, request: WorkerRequest, task_dir: Path) -> WorkerRequest:
        output_path = Path(request.output_path)
        if not output_path.is_absolute():
            output_path = task_dir / output_path
        resolved_output = output_path.resolve()
        if not self._is_relative_to(resolved_output, task_dir):
            raise ValueError("Worker output_path must stay inside its task directory")
        return request.model_copy(update={"output_path": str(resolved_output)})

    @staticmethod
    def _is_relative_to(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False
