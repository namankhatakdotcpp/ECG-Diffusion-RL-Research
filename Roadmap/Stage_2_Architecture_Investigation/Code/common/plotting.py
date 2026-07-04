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


def plot_scale_shift_fraction_vs_block(df: pd.DataFrame, fig_dir: Path) -> Path:
    """Item 5 addition -- stacked bar of scale_fraction vs shift_fraction
    per block. Expects columns `block`, `scale_fraction`, `shift_fraction`."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["block"], df["scale_fraction"], label="Scale (scale1+scale2)", color="darkorange")
    ax.bar(df["block"], df["shift_fraction"], bottom=df["scale_fraction"],
           label="Shift (shift1+shift2)", color="steelblue")
    ax.axhline(0.5, linestyle="--", color="gray", alpha=0.7, label="Even split (0.5)")
    ax.set_xlabel("Transformer block index (1 = earliest)")
    ax.set_ylabel("Fraction of adaLN weight-matrix Frobenius-norm-squared")
    ax.set_title("Item 5: adaLN scale vs. shift capacity allocation per block")
    ax.set_xticks(df["block"])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = fig_dir / "scale_shift_fraction_vs_block.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_attention_entropy_vs_block(df: pd.DataFrame, fig_dir: Path, threshold: float = 0.05) -> Path:
    """Item 6 addition -- entropy vs block, class A/B overlaid, plus
    the |diff| on a secondary axis with the locked class-blindness
    threshold marked. Expects columns `block`, `entropy_class_A`,
    `entropy_class_B`, `entropy_diff`."""
    fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
    ax1.plot(df["block"], df["entropy_class_A"], marker="o", color="steelblue", label="Class A (NORM)")
    ax1.plot(df["block"], df["entropy_class_B"], marker="s", color="crimson",
              alpha=0.8, linestyle="--", label="Class B (pooled)")
    ax1.set_xlabel("Transformer block index (1 = earliest)")
    ax1.set_ylabel("Attention entropy H (nats)")
    ax1.set_xticks(df["block"])
    ax1.legend(fontsize=8, loc="upper right")

    ax2 = ax1.twinx()
    ax2.bar(df["block"], df["entropy_diff"], alpha=0.25, color="gray", label="|diff|")
    ax2.axhline(threshold, linestyle=":", color="black", alpha=0.7, label=f"Threshold ({threshold})")
    ax2.set_ylabel("|entropy diff| (nats)")
    ax2.legend(fontsize=8, loc="upper left")

    ax1.set_title("Item 6: attention entropy vs. block, class A vs. B")
    fig.tight_layout()
    path = fig_dir / "attention_entropy_vs_block.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_representation_collapse(df: pd.DataFrame, fig_dir: Path, chance: float) -> Path:
    """Item 8 addition -- Fisher ratio and linear-probe accuracy vs block,
    one line per timestep. Expects columns `block`, `timestep`,
    `fisher_ratio`, `probe_accuracy`."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {100: "steelblue", 500: "darkorange", 900: "crimson"}
    for t_val, g in df.groupby("timestep"):
        g = g.sort_values("block")
        c = colors.get(t_val, "gray")
        ax1.plot(g["block"], g["fisher_ratio"], marker="o", color=c, label=f"t={t_val}")
        ax2.plot(g["block"], g["probe_accuracy"], marker="s", color=c, label=f"t={t_val}")

    ax1.set_xlabel("Block")
    ax1.set_ylabel("Fisher ratio")
    ax1.set_title("Fisher ratio vs. block")
    ax1.set_xticks(sorted(df["block"].unique()))
    ax1.legend(fontsize=8)

    ax2.axhline(chance, linestyle="--", color="gray", label=f"Chance ({chance:.3f})")
    ax2.set_xlabel("Block")
    ax2.set_ylabel("Linear-probe test accuracy")
    ax2.set_title("Linear-probe accuracy vs. block")
    ax2.set_xticks(sorted(df["block"].unique()))
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)

    fig.suptitle("Item 8: representation-collapse analysis")
    fig.tight_layout()
    path = fig_dir / "representation_collapse_vs_block.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
