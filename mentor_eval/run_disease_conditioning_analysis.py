"""
mentor_eval/run_disease_conditioning_analysis.py — orchestrate the disease
conditioning diagnostic pipeline.

Runs items 1–3 from the conditioning analysis plan in order:
  1. conditioning_diagnostic.py   — confusion table, are diffusion classes separable?
  2. cfg_sweep.py                 — CFG availability check (not supported; writes note)
  3. embedding_visualization.py   — UMAP/t-SNE of penultimate-layer features

Outputs go to outputs/conditioning_analysis/ (separate from outputs/mentor_review/).
A SUMMARY.md is written there at the end, generated from real disk state.

Item 4 (expanded classification metrics) is additive to classification_validation.py
and is wired into the existing run_all.py pipeline — not repeated here.

Usage:
    python -m mentor_eval.run_disease_conditioning_analysis
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_config, get_logger
from utils.backup import snapshot_before_write

OUT_DIR = Path("outputs/conditioning_analysis")

STAGES = [
    ("1. Conditioning diagnostic",          ["-m", "mentor_eval.conditioning_diagnostic"]),
    ("2. CFG availability check",           ["-m", "mentor_eval.cfg_sweep"]),
    ("3. Embedding visualization",          ["-m", "mentor_eval.embedding_visualization"]),
]


def _write_summary() -> None:
    lines = [
        "# Disease Conditioning Analysis — Summary\n",
        "Generated from real disk state. Each entry shows what exists on disk.\n",
        f"Output directory: `{OUT_DIR}/`\n\n",
    ]

    file_descriptions = {
        "conditioning_confusion.csv":  "Confusion table: rows=diffusion class, cols=MentorClassifier prediction",
        "conditioning_heatmap.png":    "Colored heatmap of the confusion table (diagonal=conditioning works)",
        "mentor_classifier.pt":        "Cached MentorClassifier weights (reused by embedding_visualization.py)",
        "cfg_sweep_result.txt":        "CFG availability check — explains why scale sweep is not runnable",
        "embedding_umap.png":          "UMAP of 128-dim penultimate features, real=circles, generated=triangles",
        "embedding_tsne.png":          "t-SNE of 128-dim penultimate features, real=circles, generated=triangles",
        "embedding_features.csv":      "Raw 2D UMAP/t-SNE coordinates + class + source labels",
    }

    lines.append("## Files\n\n")
    lines.append("| File | Status | Description |\n")
    lines.append("|------|--------|-------------|\n")
    for fname, desc in file_descriptions.items():
        fpath = OUT_DIR / fname
        status = "exists" if fpath.exists() else "missing (blocked)"
        lines.append(f"| `{fname}` | {status} | {desc} |\n")

    lines.append("\n## Interpretation guide\n\n")
    lines.append(
        "- **conditioning_heatmap.png**: if conditioning is working, each row should "
        "have most mass on a different column. If every row concentrates on one column "
        "(e.g., AFIB), conditioning has collapsed — the class label is not steering "
        "generation.\n"
    )
    lines.append(
        "- **embedding_umap/tsne.png**: if conditioning works, real-class clusters "
        "should overlap with same-class generated triangles. If generated samples "
        "form their own blob away from all real clusters, that is visual proof of "
        "conditioning failure.\n"
    )
    lines.append(
        "- **cfg_sweep_result.txt**: explains that CFG requires retraining with "
        "unconditional dropout (p_uncond > 0) — not available in the current model.\n"
    )

    summary_path = OUT_DIR / "SUMMARY.md"
    summary_path.write_text("".join(lines))
    print(f"\nSUMMARY.md written → {summary_path}")


def main() -> None:
    cfg = load_config()
    log = get_logger("run_disease_conditioning_analysis", cfg=cfg)
    snapshot_before_write(OUT_DIR)
    root = Path(__file__).resolve().parents[1]

    results = []
    for name, args in STAGES:
        log.info(f"=== {name} ===")
        proc = subprocess.run([sys.executable, *args], cwd=str(root))
        ok = proc.returncode == 0
        results.append((name, ok))
        log.info(f"{'OK' if ok else 'BLOCKED/FAILED'}: {name}")

    print("\n" + "=" * 60)
    print("Disease conditioning analysis — stage summary")
    print("=" * 60)
    for name, ok in results:
        print(f"  [{'x' if ok else ' '}] {name}")

    _write_summary()
    print(f"\n✓ All outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
