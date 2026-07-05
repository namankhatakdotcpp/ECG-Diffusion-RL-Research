"""
Stage 3 -- comparison bar charts, generated from compare_candidates.py's
own Stage3_Comparison.csv (never recomputes metrics itself, so there is
exactly one place -- compare_candidates.py -- that reads mentor_eval
output; this script only visualizes what that already produced).

Handles the current real state gracefully: if every candidate is still
"not yet evaluated", every metric column is empty, and this script
prints which plots were skipped rather than emitting empty/misleading
charts or fabricating placeholder bars.

Usage:
    python compare_candidates.py          # must run first, produces the CSV
    python make_comparison_plots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[3]
REPORTS_DIR = REPO_ROOT / "Roadmap" / "Stage_3_Architecture_Improvements" / "Reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CSV_PATH = REPORTS_DIR / "Stage3_Comparison.csv"

# (CSV column, output filename, y-axis label)
PLOTS = [
    ("Generated Accuracy", "accuracy_comparison.png", "Accuracy (generated data)"),
    ("Generated Macro-F1", "macro_f1_comparison.png", "Macro-F1 (generated data)"),
    ("Generated Macro-AUC", "macro_auc_comparison.png", "Macro-AUC (generated data)"),
    ("Similarity Cosine (mean)", "similarity_cosine_comparison.png", "Mean cosine similarity (real vs. generated)"),
]


def _bar_plot(df: pd.DataFrame, column: str, out_path: Path, ylabel: str) -> bool:
    """Returns False (and writes nothing) if every value in this column is
    missing -- an empty/all-zero bar chart would misrepresent "not
    evaluated yet" as "evaluated at zero", which is not true."""
    values = pd.to_numeric(df[column], errors="coerce")
    if values.isna().all():
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"{c}\n({v})" for c, v in zip(df["Candidate"], df["Variant"])]
    plotted = values.notna()
    ax.bar(
        [l for l, p in zip(labels, plotted) if p],
        [v for v, p in zip(values, plotted) if p],
        color="#4C72B0",
    )
    if (~plotted).any():
        skipped = [c for c, p in zip(df["Candidate"], plotted) if not p]
        ax.set_title(f"{ylabel}\n(not yet available: {', '.join(skipped)})", fontsize=10)
    else:
        ax.set_title(ylabel)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Candidate (variant)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    if not CSV_PATH.exists():
        print(f"[BLOCKED] {CSV_PATH} not found -- run compare_candidates.py first. Nothing fabricated.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    written, skipped = [], []
    for column, filename, ylabel in PLOTS:
        out_path = FIGURES_DIR / filename
        if _bar_plot(df, column, out_path, ylabel):
            written.append(out_path)
        else:
            skipped.append((filename, column))

    for path in written:
        print(f"Wrote {path}")
    for filename, column in skipped:
        print(f"[SKIPPED] {filename} -- no candidate has a value yet for '{column}'")


if __name__ == "__main__":
    main()
