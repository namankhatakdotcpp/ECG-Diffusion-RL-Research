"""
mentor_eval/run_subband_analysis.py — orchestrate the Sharma-style
multiscale-energy (MEES) visualization suite, parallel to run_all.py.

This is an ADDITIONAL set of figures/metrics — it does not replace or
modify anything produced by run_all.py (item 5's plain-table confusion
matrix is the one piece wired into classification_validation.py directly,
additively, alongside its existing colored heatmap).

Runs:
  1+2. mentor_eval.subband_decomposition   — energy table (real data) +
       real-vs-generated box plots (needs checkpoint)
  3.   mentor_eval.subband_annotated_beat  — annotated single-beat figures
       (real data)
  4.   mentor_eval.subband_similarity_metrics — per-subband Mahalanobis/
       Bhattacharyya/cosine (needs checkpoint)

Usage:
    python -m mentor_eval.run_subband_analysis
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_config, get_logger

STAGES = [
    ("1+2. Subband energy table + box plots", ["-m", "mentor_eval.subband_decomposition"]),
    ("3. Annotated single-beat figures",       ["-m", "mentor_eval.subband_annotated_beat"]),
    ("4. Subband-level similarity metrics",    ["-m", "mentor_eval.subband_similarity_metrics"]),
]


def main() -> None:
    cfg = load_config()
    log = get_logger("run_subband_analysis", cfg=cfg)
    root = Path(__file__).resolve().parents[1]

    results = []
    for name, args in STAGES:
        log.info(f"=== {name} ===")
        proc = subprocess.run([sys.executable, *args], cwd=str(root))
        ok = proc.returncode == 0
        results.append((name, ok))
        log.info(f"{'OK' if ok else 'BLOCKED/FAILED'}: {name}")

    print("\n" + "=" * 60)
    print("subband_analysis pipeline summary")
    print("=" * 60)
    for name, ok in results:
        print(f"  [{'x' if ok else ' '}] {name}")
    print("  [ ] item 5 (plain-table confusion matrix) is wired into "
          "classification_validation.py directly — run `python -m mentor_eval.run_all` "
          "(or classification_validation.py alone) for that output.")

    from mentor_eval.write_subband_summary import write_subband_summary
    write_subband_summary(cfg)
    from mentor_eval.subband_features import subband_output_dir
    print(f"\n✓ SUMMARY.md (re)generated under {subband_output_dir(cfg)}/")


if __name__ == "__main__":
    main()
