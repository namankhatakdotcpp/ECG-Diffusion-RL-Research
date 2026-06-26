import os
import shutil
import time
from pathlib import Path

_ENV_VAR = "ECG_RUN_ID"
_MARKER = ".run_id"


def get_or_create_run_id() -> str:
    """One id per logical run. Set once by whichever script starts first;
    inherited automatically by any subprocess it spawns (os.environ is
    inherited by subprocess.run() by default), so an orchestrator and its
    stage scripts share one id and don't re-archive each other's output."""
    run_id = os.environ.get(_ENV_VAR)
    if run_id is None:
        run_id = time.strftime("%Y%m%d_%H%M%S")
        os.environ[_ENV_VAR] = run_id
    return run_id


def snapshot_before_write(path: Path) -> None:
    """Call once at the top of any script that OWNS an output directory
    (i.e. produces a full batch of results there, not just adds one file
    to a shared folder). If `path` holds content from a previous run,
    rename (not copy — zero extra disk) it to a sibling backup dir first."""
    path = Path(path)
    run_id = get_or_create_run_id()
    marker = path / _MARKER

    if path.exists() and any(path.iterdir()):
        if marker.exists() and marker.read_text().strip() == run_id:
            return  # already claimed by this run — no-op
        old_id = (marker.read_text().strip() if marker.exists()
                  else time.strftime("%Y%m%d_%H%M%S", time.localtime(path.stat().st_mtime)))
        backup_path = path.parent / f"{path.name}_backup_{old_id}"
        n = 1
        while backup_path.exists():
            n += 1
            backup_path = path.parent / f"{path.name}_backup_{old_id}_{n}"
        shutil.move(str(path), str(backup_path))

    path.mkdir(parents=True, exist_ok=True)
    marker.write_text(run_id)
