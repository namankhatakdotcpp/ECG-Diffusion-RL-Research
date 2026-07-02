"""
experiment_logger.py

Structured experiment logging and results-ledger infrastructure for the
ECG diffusion conditioning research roadmap.

Every experiment script wraps its main body in an ExperimentLogger context
manager. On exit (success, failure, or crash) it:

  1. Writes a full console transcript to Logs/<experiment_id>_<timestamp>.log
  2. Appends one JSON record to Reports/results_ledger.jsonl
  3. Regenerates Reports/MASTER_LOG.md from the full ledger, including
     automated sanity-check flags (e.g. a run whose logged parameter does
     not match its logged outcome — see build_master_log.py).

Nothing about how an experiment script prints needs to change. Everything
written to stdout/stderr inside the `with` block is captured into the log
file in addition to still appearing on the console in real time.

Usage
-----
    from experiment_logger import ExperimentLogger

    with ExperimentLogger(
        experiment_id="exp2_dataset_scaling_5000",
        stage="Stage_1_Diagnosis",
        root_dir=Path("Roadmap/Stage_1_Diagnosis"),
        params={"dataset_size_requested": 5000, "n_epochs": 200},
        seed=42,
    ) as exp:
        ...  # experiment code, unmodified
        exp.log_metric("accuracy", 0.0167)
        exp.log_metric("n_train_records_actual", 380)
        exp.log_artifact(csv_path, "raw per-epoch metrics")

The ledger is append-only JSON Lines (one JSON object per line), so a run
that crashes mid-write never corrupts previous entries, and multiple
processes can append concurrently without a read-modify-write race.
"""

from __future__ import annotations

import io
import json
import platform
import subprocess
import sys
import traceback
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class _Tee(io.TextIOBase):
    """Writes every string to both the original stream and a capture buffer."""

    def __init__(self, original: Any, capture: io.StringIO) -> None:
        self._original = original
        self._capture = capture

    def write(self, s: str) -> int:
        self._original.write(s)
        self._original.flush()
        self._capture.write(s)
        return len(s)

    def flush(self) -> None:
        self._original.flush()


def _git_info(repo_dir: Path) -> dict[str, Optional[Any]]:
    def _run(args: list[str]) -> Optional[str]:
        try:
            out = subprocess.run(
                args, cwd=str(repo_dir), capture_output=True, text=True, timeout=5,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    status = _run(["git", "status", "--porcelain"])
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status) if status is not None else None,
    }


def _torch_info() -> dict[str, Optional[Any]]:
    try:
        import torch  # local import — this module must not hard-require torch
        return {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        }
    except Exception:
        return {"torch_version": None, "cuda_available": None, "cuda_device_name": None}


class ExperimentLogger(AbstractContextManager):
    """
    Context manager that captures console output, environment metadata, and
    structured metrics/artifacts for a single experiment run, then appends
    the result to a JSON-Lines ledger and regenerates a markdown digest.

    The wrapped code's exceptions are logged (with full traceback) and then
    re-raised — this logger records failures, it does not swallow them.
    """

    def __init__(
        self,
        experiment_id: str,
        stage: str,
        root_dir: Path,
        params: Optional[dict[str, Any]] = None,
        seed: Optional[int] = None,
        repo_dir: Optional[Path] = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.stage = stage
        self.root_dir = Path(root_dir)
        self.params = dict(params or {})
        self.seed = seed
        self.repo_dir = Path(repo_dir) if repo_dir else Path.cwd()

        self._metrics: dict[str, Any] = {}
        self._artifacts: list[dict[str, str]] = []
        self._notes: list[str] = []

        self.logs_dir = self.root_dir / "Logs"
        self.reports_dir = self.root_dir / "Reports"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self._stdout_capture = io.StringIO()
        self._stderr_capture = io.StringIO()
        self._orig_stdout: Any = None
        self._orig_stderr: Any = None
        self._start: Optional[datetime] = None
        self._end: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public logging API — call these from inside the `with` block
    # ------------------------------------------------------------------
    def log_metric(self, name: str, value: Any) -> None:
        self._metrics[name] = value

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        self._metrics.update(metrics)

    def log_param(self, name: str, value: Any) -> None:
        self.params[name] = value

    def log_artifact(self, path: Path, description: str = "") -> None:
        self._artifacts.append({"path": str(path), "description": description})

    def log_note(self, text: str) -> None:
        self._notes.append(text)

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------
    def __enter__(self) -> "ExperimentLogger":
        self._start = datetime.now(timezone.utc)
        self._orig_stdout, self._orig_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(self._orig_stdout, self._stdout_capture)
        sys.stderr = _Tee(self._orig_stderr, self._stderr_capture)
        print(f"=== [{self.experiment_id}] started {self._start.isoformat()} ===")
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> bool:
        self._end = datetime.now(timezone.utc)
        if exc_type is None:
            status = "success"
        elif exc_type in (AssertionError, ValueError, RuntimeError):
            status = "failed"
        else:
            status = "crashed"

        exception_text: Optional[str] = None
        if exc_type is not None:
            exception_text = "".join(
                traceback.format_exception(exc_type, exc_value, exc_tb)
            )
            print(f"=== [{self.experiment_id}] {status.upper()} ===")
            print(exception_text)

        duration = (self._end - self._start).total_seconds()
        print(
            f"=== [{self.experiment_id}] finished {self._end.isoformat()} "
            f"({duration:.1f}s) status={status} ==="
        )

        sys.stdout, sys.stderr = self._orig_stdout, self._orig_stderr

        log_path = (
            self.logs_dir
            / f"{self.experiment_id}_{self._start.strftime('%Y%m%dT%H%M%SZ')}.log"
        )
        log_path.write_text(
            self._stdout_capture.getvalue()
            + "\n----- STDERR -----\n"
            + self._stderr_capture.getvalue()
        )

        record = {
            "experiment_id": self.experiment_id,
            "stage": self.stage,
            "timestamp_start": self._start.isoformat(),
            "timestamp_end": self._end.isoformat(),
            "duration_seconds": round(duration, 2),
            "status": status,
            "seed": self.seed,
            "params": self.params,
            "metrics": self._metrics,
            "artifacts": self._artifacts,
            "notes": self._notes,
            "log_file": str(log_path.relative_to(self.root_dir)),
            "exception": exception_text,
            "environment": {
                **_git_info(self.repo_dir),
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                **_torch_info(),
            },
        }

        ledger_path = self.reports_dir / "results_ledger.jsonl"
        with open(ledger_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Local import: build_master_log does not import this module, so
        # this is not a circular import — it just keeps the two concerns
        # (running an experiment vs. rendering the digest) in separate files.
        sys.path.insert(0, str(Path(__file__).parent))
        from build_master_log import regenerate

        regenerate(self.root_dir)

        return False  # never swallow the exception
