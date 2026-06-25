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

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_config, get_logger

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


if __name__ == "__main__":
    main()
