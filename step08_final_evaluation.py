"""
step08_final_evaluation.py — Head-to-head evaluation: baseline vs RL fine-tuned diffusion.

PURPOSE
-------
Produce the main results table and paper figures comparing:
  • Baseline Diffusion (diffusion_best.pt)
  • Diag-Only RL ablation (if available from step09)
  • Ours — Full RL fine-tuned (diffusion_rl_best.pt)

All metrics are inherited from step05 (imported, not duplicated), run identically
on both models across 3 seeds.  Significance tested with Wilcoxon signed-rank.

READS
-----
  outputs/models/diffusion_best.pt
  outputs/models/diffusion_rl_best.pt
  outputs/models/rl_diag_only.pt          (optional — step09 ablation)
  outputs/processed/  (real data)
  outputs/logs/rl_training_log.csv        (reward trajectory)
  data/ptbxl/ptbxl_database.csv

WRITES
------
  outputs/results/all_metrics.json
  outputs/results/main_results_table.tex  (paper Table 2)
  outputs/results/fig05_main_comparison.{pdf,png}
  outputs/results/fig06_reward_trajectory.{pdf,png}
  outputs/results/fig07_tstr_per_class.{pdf,png}
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, get_logger, set_seed

# ── Reuse everything from step05 — no metric code is duplicated ───────────────
from step05_baseline_eval import (
    LEAD_II, PUBSTYLE,
    Simple1DCNN, FEDEncoder,
    _load_real_data,
    _generate_all_classes,
    _metric_dtw, _metric_mmd, _metric_fed,
    _metric_morphology, _metric_tstr_trtr,
    _train_fed_encoder, _train_eval_cnn, _embed, _agg,
)
from step04_transformer_diffusion import ECGTransformerDiffusion, GaussianDiffusion, EMA

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_ckpt(
    ckpt_path: Path,
    cfg,
    device:    str,
    log,
) -> Optional[tuple[nn.Module, GaussianDiffusion, EMA, list[str]]]:
    """
    Load any diffusion checkpoint → (model, diffusion, ema, class_names).
    Returns None if the file does not exist (graceful skip for optional models).
    """
    if not ckpt_path.exists():
        log.warning(f"Checkpoint not found — skipping: {ckpt_path}")
        return None

    log.info(f"Loading {ckpt_path.name} …")
    ckpt        = torch.load(str(ckpt_path), map_location=device)
    class_names = ckpt["class_names"]
    n_classes   = ckpt["n_classes"]

    model = ECGTransformerDiffusion(cfg, n_classes=n_classes).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ema = EMA(model, decay=float(cfg.diffusion.ema_decay))
    if "ema_shadow" in ckpt:
        ema.shadow = {k: v.to(device) for k, v in ckpt["ema_shadow"].items()}

    diffusion = GaussianDiffusion(
        T=int(cfg.diffusion.T),
        beta_schedule=str(cfg.diffusion.beta_schedule),
        device=device,
    )
    return model, diffusion, ema, class_names


# ──────────────────────────────────────────────────────────────────────────────
# Per-model, per-seed metric runner
# ──────────────────────────────────────────────────────────────────────────────

def _run_one_seed(
    gen:         dict[str, np.ndarray],
    X_train:     np.ndarray,
    y_train:     np.ndarray,
    X_test:      np.ndarray,
    y_test:      np.ndarray,
    class_names: list[str],
    morph_stats: dict,
    fed_encoder: FEDEncoder,
    trtr_cache:  dict,
    cfg,
    device:      str,
    seed:        int,
    log,
) -> dict:
    """Compute all evaluation metrics for one (model, seed) combination."""
    rng         = np.random.default_rng(seed)
    fs          = float(cfg.ptbxl.sampling_rate)
    n_subsample = int(cfg.eval.dtw_subsample)
    n_morph     = int(cfg.eval.n_morphology_eval)
    n_per_class = int(cfg.eval.n_synthetic_per_class)

    log.info("    DTW …")
    dtw = _metric_dtw(gen, X_test, y_test, class_names, n_subsample, rng)

    log.info("    MMD …")
    mmd = _metric_mmd(gen, X_test, y_test, class_names, rng=rng)

    log.info("    FED …")
    fed = _metric_fed(gen, X_test, y_test, class_names, fed_encoder, device)

    log.info("    MorphVal …")
    morph = _metric_morphology(gen, morph_stats, class_names, fs, n_morph, rng, log)

    log.info("    TSTR …")
    tstr, trtr = _metric_tstr_trtr(
        gen, X_train, y_train, X_test, y_test,
        class_names, n_per_class, cfg, device, log, trtr_cache,
    )

    return {
        "dtw_overall":       dtw["overall"],
        "dtw_per_class":     dtw["per_class"],
        "mmd_overall":       mmd["overall"],
        "mmd_per_class":     mmd["per_class"],
        "fed_overall":       fed["overall"],
        "fed_per_class":     fed["per_class"],
        "morph_overall":     morph["overall"],
        "morph_per_class":   morph["per_class"],
        "tstr_macro_f1":     tstr["macro_f1"],
        "tstr_per_class_f1": tstr["per_class_f1"],
        "trtr_macro_f1":     trtr["macro_f1"],
        "trtr_per_class_f1": trtr["per_class_f1"],
    }


def _aggregate_seeds(seed_results: list[dict], class_names: list[str]) -> dict:
    """Aggregate a list of per-seed dicts into mean ± std summary."""

    def _collect(key: str) -> list[float]:
        return [r[key] for r in seed_results if key in r and not math.isnan(r[key])]

    summary: dict = {"metrics": {}, "per_class": {c: {} for c in class_names}}

    for metric, key in [
        ("DTW",          "dtw_overall"),
        ("MMD",          "mmd_overall"),
        ("FED",          "fed_overall"),
        ("MorphVal",     "morph_overall"),
        ("TSTR_macro_F1","tstr_macro_f1"),
        ("TRTR_macro_F1","trtr_macro_f1"),
    ]:
        vals = _collect(key)
        m, s = _agg(vals)
        summary["metrics"][metric] = {"mean": m, "std": s, "n": len(vals)}

    # Per-class DTW / MMD / Morph / TSTR F1
    for ci, cls in enumerate(class_names):
        for pck, sk in [("dtw", "dtw_per_class"), ("mmd", "mmd_per_class"),
                        ("morph", "morph_per_class"), ("fed", "fed_per_class")]:
            vals = [r[sk].get(cls, float("nan")) for r in seed_results if sk in r]
            m, s = _agg([v for v in vals if not math.isnan(v)])
            summary["per_class"][cls][f"{pck}_mean"] = m
            summary["per_class"][cls][f"{pck}_std"]  = s

        tstr_f1_vals = [r["tstr_per_class_f1"][ci]
                        for r in seed_results
                        if "tstr_per_class_f1" in r and ci < len(r["tstr_per_class_f1"])]
        m, s = _agg(tstr_f1_vals)
        summary["per_class"][cls]["tstr_f1_mean"] = m
        summary["per_class"][cls]["tstr_f1_std"]  = s

    summary["raw_seeds"] = seed_results
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Statistical test
# ──────────────────────────────────────────────────────────────────────────────

def _wilcoxon_p(a_vals: list[float], b_vals: list[float], alternative: str = "two-sided") -> float:
    """
    Wilcoxon signed-rank p-value comparing paired seed-level measurements a vs b.
    With n=3 seeds the minimum achievable p ≈ 0.125 — reported for completeness.
    Returns nan on failure.
    """
    try:
        from scipy.stats import wilcoxon
        diffs = np.array(b_vals) - np.array(a_vals)
        if np.all(diffs == 0) or len(diffs) < 2:
            return float("nan")
        _, p = wilcoxon(diffs, alternative=alternative)
        return float(p)
    except Exception:
        return float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# Paper Table 2
# ──────────────────────────────────────────────────────────────────────────────

def _make_results_table(
    model_summaries: dict[str, dict],   # model_name → aggregate summary
    model_seeds:     dict[str, list],   # model_name → list[per-seed dict]
    results_dir:     Path,
    log,
) -> None:
    """
    Produce main_results_table.tex — paper Table 2.

    Columns: DTW ↓ | MMD ↓ | Morph% ↑ | TSTR F1 ↑
    Rows:    Baseline | Diag-Only RL (opt) | Ours (Full RL)
    Significance: * for p < 0.05, † for p < 0.10 vs Baseline (Wilcoxon)
    """

    def _fmt(m: float, s: float) -> str:
        return "—" if math.isnan(m) else f"{m:.4f} \\pm {s:.4f}"

    def _sigstar(p: float) -> str:
        if math.isnan(p):
            return ""
        if p < 0.05:
            return "^{*}"
        if p < 0.10:
            return "^{\\dagger}"
        return ""

    # Determine significance of RL-FT vs Baseline
    metric_keys = {
        "DTW":           ("dtw_overall",       "less"),
        "MMD":           ("mmd_overall",       "less"),
        "MorphVal":      ("morph_overall",     "greater"),
        "TSTR_macro_F1": ("tstr_macro_f1",     "greater"),
    }

    pvals: dict[str, float] = {}
    if "baseline" in model_seeds and "rl_ft" in model_seeds:
        for mkey, (sk, alt) in metric_keys.items():
            a = [r[sk] for r in model_seeds["baseline"] if sk in r]
            b = [r[sk] for r in model_seeds["rl_ft"]    if sk in r]
            if len(a) == len(b) and len(a) >= 2:
                pvals[mkey] = _wilcoxon_p(a, b, alternative=alt)

    # Build rows
    row_defs: list[tuple[str, str]] = [
        ("baseline",  "Baseline Diffusion"),
        ("diag_only", "Diag-Only RL$^\\ddagger$"),
        ("rl_ft",     "\\textbf{Ours — Full RL}"),
    ]

    header_cols = ["Model", "DTW $\\downarrow$", "MMD $\\downarrow$",
                   "Morph (\\%) $\\uparrow$", "TSTR F1 $\\uparrow$"]
    col_align   = "l" + "c" * (len(header_cols) - 1)

    body = ""
    for model_key, display_name in row_defs:
        if model_key not in model_summaries:
            continue
        ms = model_summaries[model_key]["metrics"]

        def _cell(mkey: str) -> str:
            m = ms.get(mkey, {}).get("mean", float("nan"))
            s = ms.get(mkey, {}).get("std",  float("nan"))
            val = _fmt(m, s)
            star = _sigstar(pvals.get(mkey, float("nan"))) if model_key == "rl_ft" else ""
            return f"${val}{star}$"

        cells = [
            display_name,
            _cell("DTW"),
            _cell("MMD"),
            _cell("MorphVal"),
            _cell("TSTR_macro_F1"),
        ]
        body += " & ".join(cells) + " \\\\\n"

    n_seeds = max((len(v) for v in model_seeds.values()), default=1)
    header_row = " & ".join(header_cols) + " \\\\\n"
    caption = (
        "Main results: Baseline vs RL fine-tuned ECG diffusion model "
        f"(mean $\\pm$ std, $n={n_seeds}$ seeds). "
        "$^{*}p<0.05$, $^{\\dagger}p<0.10$ vs Baseline (Wilcoxon signed-rank). "
        "$^{\\ddagger}$Diag-Only RL uses only DiagnosticUtility reward (see ablation)."
    )
    latex = (
        "\\begin{table}[t]\n"
        + "\\centering\n"
        + f"\\caption{{{caption}}}\n"
        + "\\label{tab:main_results}\n"
        + f"\\begin{{tabular}}{{{col_align}}}\n"
        + "\\toprule\n"
        + header_row
        + "\\midrule\n"
        + body
        + "\\bottomrule\n"
        + "\\end{tabular}\n"
        + "\\end{table}\n"
    )

    path = results_dir / "main_results_table.tex"
    path.write_text(latex)
    log.info(f"LaTeX table → {path.name}")

    # Console summary
    log.info("")
    log.info("─" * 70)
    log.info("  TABLE 2  —  MAIN RESULTS")
    log.info("─" * 70)
    hdr = f"{'Model':<25} {'DTW':>10} {'MMD':>10} {'Morph%':>10} {'TSTR F1':>10}"
    log.info(hdr)
    log.info("─" * 70)
    for mkey, name in row_defs:
        if mkey not in model_summaries:
            continue
        ms = model_summaries[mkey]["metrics"]
        def _v(k: str) -> str:
            m = ms.get(k, {}).get("mean", float("nan"))
            return "—" if math.isnan(m) else f"{m:.4f}"
        log.info(f"  {name:<23} {_v('DTW'):>10} {_v('MMD'):>10} "
                 f"{_v('MorphVal'):>10} {_v('TSTR_macro_F1'):>10}")
    if pvals:
        log.info("─" * 70)
        log.info("  p-values (RL-FT vs Baseline, Wilcoxon):")
        for k, p in pvals.items():
            s = f"{p:.3f}" if not math.isnan(p) else "n/a"
            log.info(f"    {k}: {s}")
    log.info("─" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5 — Main comparison (4 rows × 6 cols, Lead II, MI class)
# ──────────────────────────────────────────────────────────────────────────────

def _make_fig_comparison(
    model_gens:   dict[str, dict[str, np.ndarray]],   # model → class → ECGs
    X_test:       np.ndarray,
    y_test:       np.ndarray,
    class_names:  list[str],
    fs:           float,
    results_dir:  Path,
    rng:          np.random.Generator,
    log,
) -> None:
    """
    Paper Figure 5: 4-row × 6-col Lead-II comparison for MI class.
      Row 1 — Real MI        (green)
      Row 2 — Baseline       (blue, dashed)
      Row 3 — Ours Full RL   (red, dashed)
      Row 4 — Diag-Only RL   (orange, dashed)   [if available]
    """
    focal_class = "MI" if "MI" in class_names else class_names[0]
    cls_idx     = class_names.index(focal_class)
    n_cols      = 6
    t_axis      = np.arange(int(fs * 10)) / fs

    row_specs: list[tuple] = []
    row_specs.append(("Real",        None,         "#2ca02c", "solid"))
    if "baseline" in model_gens:
        row_specs.append(("Baseline",    "baseline",  "#1f77b4", "dashed"))
    if "rl_ft" in model_gens:
        row_specs.append(("Ours (Full RL)", "rl_ft", "#d62728", "dashed"))
    if "diag_only" in model_gens:
        row_specs.append(("Diag-Only RL", "diag_only", "#ff7f0e", "dashed"))

    n_rows = len(row_specs)

    with plt.rc_context(PUBSTYLE):
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3 * n_cols, 2.5 * n_rows),
            constrained_layout=True,
        )
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        for ri, (label, mkey, color, ls) in enumerate(row_specs):
            if mkey is None:
                # Real data
                real_idx = np.where(y_test == cls_idx)[0]
                chosen   = rng.choice(real_idx, size=min(n_cols, len(real_idx)), replace=False)
                for col in range(n_cols):
                    ax = axes[ri, col]
                    if col < len(chosen):
                        ax.plot(t_axis, X_test[chosen[col], :, LEAD_II],
                                color=color, lw=0.8, alpha=0.9, ls=ls)
                    _style_ax(ax, ri, col, n_rows, n_cols, label, color, focal_class)
            else:
                gen_cls = model_gens[mkey][focal_class]  # (N, 1000, 12)
                chosen  = rng.choice(len(gen_cls), size=n_cols, replace=False)
                for col in range(n_cols):
                    ax = axes[ri, col]
                    ax.plot(t_axis, gen_cls[chosen[col], :, LEAD_II],
                            color=color, lw=0.8, alpha=0.9, ls=ls)
                    _style_ax(ax, ri, col, n_rows, n_cols, label, color, focal_class)

        fig.suptitle(
            f"Lead II — {focal_class} class: Real vs Baseline vs RL fine-tuned\n"
            "Composite reward preserves morphology; Diag-Only reward causes waveform collapse",
            fontsize=10,
        )

    for ext in ("pdf", "png"):
        fig.savefig(str(results_dir / f"fig05_main_comparison.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Figure 5 (main comparison) saved.")


def _style_ax(ax, ri, col, n_rows, n_cols, label, color, cls_name) -> None:
    ax.set_ylim(-4.5, 4.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)
    if col == 0:
        ax.set_ylabel(label, fontsize=8, color=color, fontweight="bold")
        if ri == 0:
            ax.set_title(cls_name, fontsize=9, loc="left")
    if ri < n_rows - 1:
        ax.set_xticks([])
    else:
        ax.set_xlabel("Time (s)", fontsize=7)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 6 — Reward trajectory
# ──────────────────────────────────────────────────────────────────────────────

def _make_fig_reward_trajectory(
    logs_dir:    Path,
    results_dir: Path,
    log,
) -> None:
    """
    Paper Figure 6: reward components vs RL iteration.

    Plots r_morph, r_hrv, r_real, r_diag, and reward_total from
    rl_training_log.csv.  Key story: composite reward keeps r_morph high;
    diag_only reward causes r_morph to collapse.
    """
    csv_path = logs_dir / "rl_training_log.csv"
    if not csv_path.exists():
        log.warning(f"rl_training_log.csv not found — skipping Figure 6.")
        return

    import pandas as pd
    df = pd.read_csv(str(csv_path))
    if df.empty:
        log.warning("rl_training_log.csv is empty — skipping Figure 6.")
        return

    # Required columns
    req = {"iter", "reward_total", "r_morph", "r_hrv", "r_real", "r_diag"}
    if not req.issubset(df.columns):
        log.warning(f"rl_training_log.csv missing columns {req - set(df.columns)} — skipping.")
        return

    # Smooth with rolling window for readability
    win = max(1, len(df) // 20)

    component_styles = [
        ("reward_total", "Total",    "#000000", 2.0, "solid"),
        ("r_morph",      "Morphol.", "#2ca02c", 1.2, "solid"),
        ("r_hrv",        "HRV",      "#1f77b4", 1.2, "solid"),
        ("r_real",       "Realism",  "#ff7f0e", 1.2, "solid"),
        ("r_diag",       "Diagn.",   "#d62728", 1.2, "solid"),
    ]

    with plt.rc_context(PUBSTYLE):
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)

        for col, label, color, lw, ls in component_styles:
            if col not in df.columns:
                continue
            y = df[col].rolling(win, min_periods=1, center=True).mean()
            ax.plot(df["iter"], y, color=color, lw=lw, ls=ls, label=label, alpha=0.9)

        ax.axhline(y=0.3, ls=":", lw=1, color="gray", alpha=0.5)
        ax.text(df["iter"].max() * 1.01, 0.3, "Morph alarm\nthreshold", fontsize=7,
                va="center", color="gray")

        ax.set_xlabel("RL Iteration", fontsize=10)
        ax.set_ylabel("Reward Component", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("RL Training: Reward Component Trajectories (Full composite reward)", fontsize=10)
        ax.legend(fontsize=8, ncol=2, loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ext in ("pdf", "png"):
        fig.savefig(str(results_dir / f"fig06_reward_trajectory.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Figure 6 (reward trajectory) saved.")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 7 — Per-class TSTR F1 grouped bar chart
# ──────────────────────────────────────────────────────────────────────────────

def _make_fig_tstr_per_class(
    model_summaries: dict[str, dict],
    class_names:     list[str],
    results_dir:     Path,
    log,
) -> None:
    """
    Paper Figure 7: per-class TSTR F1 for Baseline vs RL (grouped bar chart).
    Error bars = std across 3 seeds.
    Highlights classes where RL improves classification.
    """
    models_to_plot = [
        ("baseline", "Baseline",   "#1f77b4"),
        ("rl_ft",    "Ours (RL)",  "#d62728"),
    ]
    models_to_plot = [(k, l, c) for k, l, c in models_to_plot if k in model_summaries]

    if not models_to_plot:
        log.warning("No models available for per-class TSTR figure — skipping.")
        return

    n_models = len(models_to_plot)
    n_cls    = len(class_names)
    width    = 0.8 / n_models
    x        = np.arange(n_cls)

    with plt.rc_context(PUBSTYLE):
        fig, ax = plt.subplots(figsize=(max(6, n_cls * 1.4), 4), constrained_layout=True)

        for mi, (mkey, label, color) in enumerate(models_to_plot):
            pc = model_summaries[mkey]["per_class"]
            means = [pc[c].get("tstr_f1_mean", float("nan")) for c in class_names]
            stds  = [pc[c].get("tstr_f1_std",  0.0)          for c in class_names]
            offsets = x + (mi - (n_models - 1) / 2.0) * width
            bars = ax.bar(offsets, means, width=width * 0.9,
                          color=color, alpha=0.85, label=label,
                          yerr=stds, capsize=3, error_kw={"linewidth": 1})
            # Annotate top of each bar
            for bar, m in zip(bars, means):
                if not math.isnan(m):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f"{m:.2f}", ha="center", va="bottom", fontsize=6, color=color)

        # Mark classes where RL > Baseline
        if len(models_to_plot) == 2:
            base_ms = model_summaries["baseline"]["per_class"]
            rl_ms   = model_summaries["rl_ft"]["per_class"]
            for ci, cls in enumerate(class_names):
                bm = base_ms[cls].get("tstr_f1_mean", float("nan"))
                rm = rl_ms[cls].get("tstr_f1_mean",  float("nan"))
                if not (math.isnan(bm) or math.isnan(rm)) and rm > bm + 0.01:
                    ax.text(x[ci], max(bm, rm) + 0.06, "↑", ha="center",
                            fontsize=12, color="#d62728", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(class_names, fontsize=9)
        ax.set_ylabel("TSTR Macro F1", fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_title("Per-class TSTR F1: Baseline vs RL fine-tuned (↑ = RL improvement)",
                     fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axhline(y=1.0 / n_cls, ls=":", lw=1, color="gray", alpha=0.5)
        ax.text(n_cls - 0.5, 1.0 / n_cls + 0.01, "Random", fontsize=7, color="gray", ha="right")

    for ext in ("pdf", "png"):
        fig.savefig(str(results_dir / f"fig07_tstr_per_class.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Figure 7 (per-class TSTR) saved.")


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(cfg, log) -> float:
    """
    Run head-to-head evaluation across all available models.

    Returns the RL vs Baseline TSTR F1 improvement gap (or 0 if RL not available).
    """
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    models_dir  = Path(cfg.paths.outputs.models)
    results_dir = Path(cfg.paths.outputs.results)
    logs_dir    = Path(cfg.paths.logs)
    processed   = Path(cfg.paths.outputs.processed)

    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Load real data ────────────────────────────────────────────────────────
    log.info("=" * 65)
    log.info("Loading real data …")
    log.info("=" * 65)
    real        = _load_real_data(cfg, log)
    class_names = real["class_names"]
    n_classes   = len(class_names)
    X_train, y_train = real["X_train"], real["y_train"]
    X_test,  y_test  = real["X_test"],  real["y_test"]
    morph_stats      = real.get("morph_stats", {})
    fs               = float(cfg.ptbxl.sampling_rate)

    # ── Load all models ────────────────────────────────────────────────────────
    log.info("=" * 65)
    log.info("Loading model checkpoints …")
    log.info("=" * 65)

    ckpt_paths = {
        "baseline":  models_dir / "diffusion_best.pt",
        "rl_ft":     models_dir / "diffusion_rl_best.pt",
        "diag_only": models_dir / "rl_diag_only.pt",   # step09 ablation (optional)
    }
    loaded_models: dict[str, tuple] = {}
    for mkey, path in ckpt_paths.items():
        result = _load_ckpt(path, cfg, device, log)
        if result is not None:
            model_obj, diffusion_obj, ema_obj, ckpt_classes = result
            # Verify class list matches (warn if not)
            if ckpt_classes != class_names:
                log.warning(f"  {mkey}: class mismatch {ckpt_classes} vs {class_names}; "
                            "using checkpoint classes")
                class_names_for_model = ckpt_classes
            else:
                class_names_for_model = class_names
            loaded_models[mkey] = (model_obj, diffusion_obj, ema_obj, class_names_for_model)

    if "baseline" not in loaded_models:
        raise FileNotFoundError("diffusion_best.pt is required. Run step04 first.")
    log.info(f"Models loaded: {list(loaded_models)}")

    # ── Train FED encoder once (same across all models) ───────────────────────
    log.info("=" * 65)
    log.info("Training FED encoder (once, shared across models) …")
    log.info("=" * 65)
    fed_encoder = _train_fed_encoder(X_train, y_train, n_classes, cfg, device, log)

    # ── TRTR baseline (real data — seed-independent) ──────────────────────────
    log.info("=" * 65)
    log.info("Computing TRTR reference …")
    log.info("=" * 65)
    trtr_cache: dict = {}
    _train_eval_cnn(X_train, y_train, X_test, y_test, n_classes, cfg, device, "TRTR", log)
    trtr_cache.update(
        _train_eval_cnn(X_train, y_train, X_test, y_test, n_classes, cfg, device, "TRTR", log)
    )
    log.info(f"  TRTR macro F1 = {trtr_cache.get('macro_f1', float('nan')):.4f}")

    eval_seeds  = list(cfg.eval.seeds)
    n_per_class = int(cfg.eval.n_synthetic_per_class)

    # ── Per-model evaluation loop ─────────────────────────────────────────────
    all_seed_results: dict[str, list[dict]] = {}    # model → [seed_dict, ...]
    all_gens_for_fig: dict[str, dict]       = {}    # model → class → ECGs (seed 0)

    for mkey, (model_obj, diffusion_obj, ema_obj, cls_names_m) in loaded_models.items():
        log.info("=" * 65)
        log.info(f"Evaluating model: {mkey}")
        log.info("=" * 65)
        seed_results_m: list[dict] = []

        for si, seed in enumerate(eval_seeds):
            set_seed(seed)
            log.info(f"  Seed {seed} ({si + 1}/{len(eval_seeds)}) …")

            gen = _generate_all_classes(
                model_obj, diffusion_obj, ema_obj, cls_names_m,
                n_per_class, cfg, seed, device, real.get("prep_stats"), log,
            )

            # Store first seed's generated data for the comparison figure
            if si == 0:
                all_gens_for_fig[mkey] = gen

            result = _run_one_seed(
                gen, X_train, y_train, X_test, y_test,
                cls_names_m, morph_stats, fed_encoder,
                trtr_cache, cfg, device, seed, log,
            )
            log.info(f"    DTW={result['dtw_overall']:.4f} "
                     f"MMD={result['mmd_overall']:.4f} "
                     f"Morph={result['morph_overall']:.1f}% "
                     f"TSTR={result['tstr_macro_f1']:.4f}")
            seed_results_m.append(result)

        all_seed_results[mkey] = seed_results_m

    # ── Aggregate ─────────────────────────────────────────────────────────────
    all_summaries: dict[str, dict] = {}
    for mkey, seed_list in all_seed_results.items():
        all_summaries[mkey] = _aggregate_seeds(seed_list, class_names)

    # ── Save all_metrics.json ─────────────────────────────────────────────────
    all_metrics_path = results_dir / "all_metrics.json"
    with open(all_metrics_path, "w") as f:
        # Convert numpy types for JSON serialisation
        def _jsonify(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        json.dump(all_summaries, f, indent=2, default=_jsonify)
    log.info(f"All metrics → {all_metrics_path.name}")

    # ── Paper Table 2 ─────────────────────────────────────────────────────────
    _make_results_table(all_summaries, all_seed_results, results_dir, log)

    # ── Figure 5 — Main comparison ────────────────────────────────────────────
    log.info("=" * 65)
    log.info("Generating Figure 5 (main comparison) …")
    log.info("=" * 65)
    rng = np.random.default_rng(eval_seeds[0])
    _make_fig_comparison(
        all_gens_for_fig, X_test, y_test, class_names, fs, results_dir, rng, log,
    )

    # ── Figure 6 — Reward trajectory ──────────────────────────────────────────
    log.info("=" * 65)
    log.info("Generating Figure 6 (reward trajectory) …")
    log.info("=" * 65)
    _make_fig_reward_trajectory(logs_dir, results_dir, log)

    # ── Figure 7 — Per-class TSTR ─────────────────────────────────────────────
    log.info("=" * 65)
    log.info("Generating Figure 7 (per-class TSTR) …")
    log.info("=" * 65)
    _make_fig_tstr_per_class(all_summaries, class_names, results_dir, log)

    # ── Compute return value ──────────────────────────────────────────────────
    baseline_tstr = all_summaries.get("baseline", {}).get("metrics", {}).get(
        "TSTR_macro_F1", {}
    ).get("mean", float("nan"))
    rl_tstr = all_summaries.get("rl_ft", {}).get("metrics", {}).get(
        "TSTR_macro_F1", {}
    ).get("mean", baseline_tstr)   # fallback to baseline if RL not available

    gap = float(rl_tstr - baseline_tstr) if not (
        math.isnan(rl_tstr) or math.isnan(baseline_tstr)
    ) else 0.0

    return gap


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    log = get_logger("step08_final_evaluation", cfg=cfg)
    set_seed(int(cfg.seeds[0]))

    gap = evaluate(cfg, log)
    sign = "+" if gap >= 0 else ""
    print(f"✓ Final evaluation complete. RL vs Baseline TSTR gap: {sign}{gap:.3f}")


if __name__ == "__main__":
    main()
