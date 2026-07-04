"""
Stage 2 Tier 0 -- shared gain-sweep plots.

LIFTED (copied, not moved) from
stage2_tier0_item2_localized_gain/item2_plots_and_report.py (Item 2A) --
original untouched. Parameterized on a `variant` label (e.g. "localized",
"uniform") so Item 2B and later items can reuse the same three plots
without duplicating the matplotlib boilerplate.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common.statistics import DIRECTION_FLOOR


def plot_recovery_vs_gain(df: pd.DataFrame, fig_dir: Path, variant: str = "") -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["gain"], df["recovery_pct"], marker="o", color="steelblue")
    ax.axhline(70, linestyle="--", color="green", label="SUPPORTED (>=70%)")
    ax.axhline(30, linestyle="--", color="orange", label="Partial floor (>=30%)")
    ax.set_xlabel("Gain g")
    ax.set_ylabel("Block 6 recovery fraction (%)")
    ax.set_title(f"{variant} gain: Block 6 recovery vs. gain".strip())
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = fig_dir / "recovery_vs_gain.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_direction_vs_gain(df: pd.DataFrame, fig_dir: Path, variant: str = "") -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["gain"], df["min_direction_consistency"], marker="o", color="crimson")
    ax.axhline(DIRECTION_FLOOR, linestyle="--", color="gray", label=f"Direction floor ({DIRECTION_FLOOR})")
    ax.set_xlabel("Gain g")
    ax.set_ylabel("Min direction consistency (layers 2-6)")
    ax.set_title(f"{variant} gain: direction consistency vs. gain".strip())
    ax.set_ylim(0.98, 1.001)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = fig_dir / "direction_vs_gain.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_propagation_efficiency(df: pd.DataFrame, fig_dir: Path, variant: str = "") -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_df = df[df["propagation_efficiency"].notna()]
    ax.plot(plot_df["gain"], plot_df["propagation_efficiency"], marker="o", color="darkorange")
    ax.set_xlabel("Gain g")
    ax.set_ylabel("Propagation efficiency (block 6 / injected)")
    ax.set_title(f"{variant} gain: propagation efficiency vs. gain\n"
                 "(g=1.0 omitted -- InjectedDelta=0)".strip())
    fig.tight_layout()
    path = fig_dir / "propagation_efficiency.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_residual_ratio_vs_block(df: pd.DataFrame, fig_dir: Path) -> Path:
    """Item 3 addition -- direct analogue of Item 1's own magnitude-vs-layer
    plot, but for R_k (the within-pass residual-update ratio) instead of
    Item 1's cross-class output-magnitude delta. Expects columns `block`
    and `R_k_combined_pooled` (plus optionally `R_k_class_A_pooled`/
    `R_k_class_B_pooled` for the per-class overlay)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["block"], df["R_k_combined_pooled"], marker="o", color="teal", label="Combined (A+B)")
    if "R_k_class_A_pooled" in df.columns:
        ax.plot(df["block"], df["R_k_class_A_pooled"], marker="s", color="steelblue",
                 alpha=0.6, label="Class A", linestyle="--")
    if "R_k_class_B_pooled" in df.columns:
        ax.plot(df["block"], df["R_k_class_B_pooled"], marker="^", color="crimson",
                 alpha=0.6, label="Class B", linestyle="--")
    ax.set_xlabel("Transformer block index (1 = earliest)")
    ax.set_ylabel("R_k -- residual update ratio (within-pass)")
    ax.set_title("Item 3: residual-update ratio vs. block")
    ax.set_xticks(df["block"])
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = fig_dir / "residual_ratio_vs_block.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
