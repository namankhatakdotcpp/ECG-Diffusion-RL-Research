"""
Stage 3 -- per-result metadata.json, one level down from run_all.py's
own baseline_manifest.json fix (same provenance discipline: a result
without a record of exactly which code/config/checkpoint produced it
is unverifiable later, same failure mode as Stage 1's EMA checkpoint
mixup). Reuses mentor_eval/run_all.py's own hashing helpers rather than
reimplementing sha256/git-commit logic a second time.

Schema (Results/<S3-XXX>/metadata.json):
    variant            -- e.g. "layerscale"
    run_id             -- e.g. "S3-002"
    commit             -- git HEAD at write time
    config_sha256      -- sha256(config.yaml) at write time
    checkpoint_sha256  -- sha256(best checkpoint), null until training produces one
    start_time         -- UTC ISO timestamp, set once, at "queued"->"training" transition
    status             -- one of: queued, training, done, killed, train_failed, eval_failed
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mentor_eval.run_all import _sha256_file, _git_commit_hash  # noqa: E402

RESULTS_ROOT = REPO_ROOT / "Roadmap" / "Stage_3_Architecture_Improvements" / "Results"
CONFIG_PATH = REPO_ROOT / "config.yaml"

VALID_STATUSES = {"queued", "training", "done", "killed", "train_failed", "eval_failed"}


def _metadata_path(run_id: str) -> Path:
    return RESULTS_ROOT / run_id / "metadata.json"


def write_metadata(
    run_id: str,
    variant: str,
    status: str,
    checkpoint_path: Optional[Path] = None,
    start_time: Optional[str] = None,
) -> Path:
    assert status in VALID_STATUSES, f"status must be one of {VALID_STATUSES}, got {status}"
    out_path = _metadata_path(run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)

    metadata = {
        "run_id": run_id,
        "variant": variant,
        "commit": _git_commit_hash(REPO_ROOT),
        "config_sha256": _sha256_file(CONFIG_PATH) if CONFIG_PATH.exists() else None,
        "checkpoint_sha256": (
            _sha256_file(checkpoint_path) if checkpoint_path and Path(checkpoint_path).exists()
            else existing.get("checkpoint_sha256")
        ),
        "start_time": start_time or existing.get("start_time") or datetime.now(timezone.utc).isoformat(),
        "status": status,
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
    }

    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return out_path
