"""
mentor_eval/run_all.py — orchestrate the full mentor-review pipeline.

Runs every mentor_eval module in order, into outputs/mentor_review/<item>/.
Modules that need a trained checkpoint, intermediate checkpoints, or a
training log will print their own [BLOCKED] message and be skipped here
(not treated as a hard failure) — this script reports which stages
actually ran at the end, then regenerates SUMMARY.md to reflect the
current state of outputs/mentor_review/.

Usage:
    python -m mentor_eval.run_all
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_config, get_logger
from utils.backup import snapshot_before_write


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit_hash(root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(root),
        capture_output=True, text=True,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def write_baseline_manifest(cfg, root: Path) -> Path:
    """Record which checkpoint this pipeline run actually evaluated.

    Without this, "checksum-verified baseline" (Stage3_Roadmap.md Sec. 6)
    is a claim, not a guarantee -- a checkpoint silently overwritten
    between runs would otherwise be undetectable from run_all.py's
    output alone (same failure mode as Stage 1's EMA checkpoint mixup).
    """
    ckpt_path = Path(cfg.paths.outputs.models) / "diffusion_best.pt"
    manifest = {
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": _sha256_file(ckpt_path) if ckpt_path.exists() else None,
        "checkpoint_exists": ckpt_path.exists(),
        "git_commit": _git_commit_hash(root),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_dir = Path(cfg.paths.outputs.results).parent / "mentor_review"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return out_path

STAGES = [
    ("1. Dataset audit",                  ["-m", "mentor_eval.dataset_audit"]),
    ("3. Per-lead/class real-ECG figures", ["-m", "mentor_eval.lead_class_figures"]),
    ("5. Zoomed clinical regions",         ["-m", "mentor_eval.zoomed_clinical"]),
    ("4. Real vs. generated comparison",   ["-m", "mentor_eval.real_vs_generated"]),
    ("6. Training progression",            ["-m", "mentor_eval.training_progression"]),
    ("7. Loss curves",                     ["-m", "mentor_eval.loss_curves"]),
    ("8. Similarity metrics",              ["-m", "mentor_eval.similarity_metrics"]),
    ("9. Classification validation",       ["-m", "mentor_eval.classification_validation"]),
]


def main() -> None:
    cfg = load_config()
    log = get_logger("run_all", cfg=cfg)
    snapshot_before_write(Path(cfg.paths.outputs.results).parent / "mentor_review")
    root = Path(__file__).resolve().parents[1]

    results = []
    for name, args in STAGES:
        log.info(f"=== {name} ===")
        proc = subprocess.run([sys.executable, *args], cwd=str(root))
        ok = proc.returncode == 0
        results.append((name, ok))
        log.info(f"{'OK' if ok else 'BLOCKED/FAILED'}: {name}")

    print("\n" + "=" * 60)
    print("mentor_eval pipeline summary")
    print("=" * 60)
    for name, ok in results:
        print(f"  [{'x' if ok else ' '}] {name}")

    from mentor_eval.write_summary import write_summary
    write_summary(cfg)
    print("\n✓ SUMMARY.md (re)generated under outputs/mentor_review/")

    manifest_path = write_baseline_manifest(cfg, root)
    print(f"✓ baseline_manifest.json written to {manifest_path}")


if __name__ == "__main__":
    main()
