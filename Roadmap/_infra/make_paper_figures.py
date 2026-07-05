"""
Roadmap/_infra/make_paper_figures.py -- structural/summary figures for
the paper that do NOT depend on unresolved Stage 3 training results.

Scope, deliberately limited to what's honest right now:
  Figure 1 -- Overall pipeline (structural: data -> model -> eval)
  Figure 2 -- Stage 3 architecture comparison (structural: which block
              gets which gain mechanism, per variant -- this is a
              description of code that already exists in
              model_variants.py, not a result)
  Figure 3 -- Stage 2 findings (REAL, already-verified numbers from
              Roadmap/Stage_2_Architecture_Investigation/STAGE2_STATUS.md
              -- Stage 2 is 100% complete, so this is not a placeholder)
  Figure 6 -- Clinical evaluation pipeline (structural: mentor_eval's
              actual module flow)
  Figure 7 -- Overall roadmap timeline (structural, current status only)

Deliberately OUT of scope here (see Roadmap/Stage_3_Architecture_Improvements/
Stage3_Status.md for why): a "Figure 4 candidate comparison" is not
duplicated here -- that's compare_candidates.py + make_comparison_plots.py,
which already read real (currently empty) Stage 3 data honestly. A
"Figure 5 RL pipeline" is deliberately NOT built: RL infrastructure and
its reward design are still undesigned (per 2026-07-05 review), so a
pipeline diagram for it would depict a system that doesn't exist yet.

Usage:
    python make_paper_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, FancyBboxPatch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
OUT_DIR = REPO_ROOT / "Roadmap" / "_infra" / "paper_figures"


def _box(ax, xy, w, h, text, facecolor="#DCE6F1", fontsize=9):
    box = FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02", linewidth=1.2,
        edgecolor="#333333", facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
             fontsize=fontsize, wrap=True)


def _arrow(ax, start, end):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.4))


def figure1_pipeline() -> Path:
    """Structural: PTB-XL -> preprocessing -> conditioned diffusion model
    -> generation -> mentor_eval. Reflects step01-step09's actual module
    names, not an idealized/aspirational pipeline."""
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3)
    ax.axis("off")

    stages = [
        "PTB-XL raw\n(step01)", "Preprocessing\n(step02)",
        "Conditioned\ndiffusion model\n(step04)", "Generation\n(checkpoint_utils)",
        "mentor_eval\n(classification,\nsimilarity, subband)",
    ]
    x = 0.3
    w, h = 1.8, 1.4
    centers = []
    for label in stages:
        _box(ax, (x, 0.8), w, h, label)
        centers.append((x + w, 0.8 + h / 2))
        x += w + 0.5
    for i in range(len(centers) - 1):
        _arrow(ax, centers[i], (centers[i][0] + 0.5 - w, centers[i][1]))

    ax.set_title("Figure 1 -- Overall Pipeline", fontsize=11)
    fig.tight_layout()
    out_path = OUT_DIR / "figure1_pipeline.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def figure2_architecture_comparison() -> Path:
    """Structural: which of the 6 blocks gets which gain mechanism, per
    Stage 3 variant -- directly reflects model_variants.build_variant_model()'s
    actual block_classes construction, not a hypothetical design."""
    variants = {
        "S3-001 baseline": ["plain"] * 6,
        "S3-002 layerscale": ["gamma"] * 6,
        "S3-003 late_gain": ["plain"] * 4 + ["gamma"] * 2,
        "S3-004 residual_scaling": ["residual-scale"] * 6,
        "S3-005 hybrid": ["gamma"] * 4 + ["gamma+boost"] * 2,
        "S3-006 final_norm_gain": ["plain"] * 6 + ["final_gamma (post-norm)"],
    }
    color_by_kind = {
        # Both "gamma" (LayerScale/late_gain/hybrid) and "residual-scale"
        # (S3-004) apply the identical x + gamma * branch_output structure
        # (model_variants.py: TransformerBlockLayerScale and
        # TransformerBlockResidualScale both do this to the SAME two
        # branches -- neither touches the x/skip term itself). The only
        # real difference is gamma's SHAPE: a per-channel vector
        # (LayerScale, size model_dim) vs. one scalar per branch (residual
        # scaling) -- a granularity difference, not a different
        # application site. Labeled "residual-scale" rather than e.g.
        # "scalar-gamma" purely to avoid a reader assuming it's a minor
        # variant of LayerScale rather than its own tested hypothesis
        # (Item 3's block-level, not channel-level, granularity).
        "plain": "#DCDCDC", "gamma": "#9ECAE1", "residual-scale": "#6BAED6",
        "gamma+boost": "#3182BD", "final_gamma (post-norm)": "#F4A582",
    }

    fig, ax = plt.subplots(figsize=(10, 4))
    row_labels = list(variants.keys())
    for row_idx, (name, blocks) in enumerate(variants.items()):
        y = len(row_labels) - row_idx - 1
        for col_idx, kind in enumerate(blocks[:6]):
            ax.add_patch(plt.Rectangle((col_idx, y), 1, 0.8, facecolor=color_by_kind.get(kind, "#FFFFFF"),
                                        edgecolor="#333333"))
        if len(blocks) > 6:
            ax.text(6.2, y + 0.4, blocks[6], fontsize=8, va="center")

    ax.set_xlim(0, 9)
    ax.set_ylim(0, len(row_labels))
    ax.set_yticks([len(row_labels) - i - 0.6 for i in range(len(row_labels))])
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xticks([i + 0.5 for i in range(6)])
    ax.set_xticklabels([f"Block {i+1}" for i in range(6)], fontsize=8)
    ax.set_title("Figure 2 -- Stage 3 Architecture Comparison (per-block gain mechanism)", fontsize=11)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c) for c in color_by_kind.values()]
    ax.legend(handles, color_by_kind.keys(), loc="upper right", fontsize=7, ncol=1)
    fig.tight_layout()
    out_path = OUT_DIR / "figure2_architecture_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def figure3_stage2_findings() -> Path:
    """REAL numbers, cited from Roadmap/Stage_2_Architecture_Investigation/
    STAGE2_STATUS.md (Stage 2 is 100% complete and verified -- this is not
    placeholder data). Item 1's cross-validated magnitude decline
    (Item 4's fixed-timestep numbers: 0.173/0.124/0.107 at t=100/500/900)
    and Item 3's Wilcoxon-confirmed block1-vs-block6 attenuation."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    timesteps = [100, 500, 900]
    magnitudes = [0.173, 0.124, 0.107]
    axes[0].plot(timesteps, magnitudes, marker="o", color="#3182BD")
    axes[0].set_title("Item 1/4: magnitude decline\nvs. timestep (t)", fontsize=10)
    axes[0].set_xlabel("Diffusion timestep t")
    axes[0].set_ylabel("Forward-pass magnitude")
    for t, m in zip(timesteps, magnitudes):
        axes[0].annotate(f"{m}", (t, m), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")

    task01_decline_pct = 67.9
    axes[1].bar(["Task 0.1: block1->block6\nconditioning-signal decline"], [task01_decline_pct], color="#DE2D26")
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Net decline (%)")
    axes[1].set_title("Phase 0 Task 0.1 (dilution)\nWilcoxon p=6.1e-05, n=15", fontsize=10)
    axes[1].annotate(f"{task01_decline_pct}%", (0, task01_decline_pct),
                      textcoords="offset points", xytext=(0, 8), fontsize=9, ha="center")

    fig.suptitle("Figure 3 -- Stage 2 / Phase 0 Verified Findings", fontsize=11)
    fig.tight_layout()
    out_path = OUT_DIR / "figure3_stage2_findings.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def figure6_clinical_eval_pipeline() -> Path:
    """Structural: mentor_eval's actual module flow for one checkpoint --
    reflects run_stage3_queue.py's real per-candidate eval loop
    (classification_validation, similarity_metrics) plus the
    not-yet-wired-in subband/Sharma step, marked as such rather than
    implied to already run automatically."""
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 3.5)
    ax.axis("off")

    _box(ax, (0.2, 1.3), 1.8, 1.2, "Trained\ncheckpoint\n(.pt)", facecolor="#DCDCDC")
    _box(ax, (2.4, 2.0), 2.2, 1.2, "classification_validation.py\n(real + generated accuracy)")
    _box(ax, (2.4, 0.4), 2.2, 1.2, "similarity_metrics.py\n(Mahalanobis/Bhattacharyya/cosine)")
    _box(ax, (5.0, 2.0), 2.4, 1.2, "compare_candidates.py\n(Stage3_Comparison.md/.csv)")
    _box(ax, (5.0, 0.4), 2.4, 1.2, "subband_similarity_metrics.py\n(Sharma-inspired, ST/T-wave)\n[NOT auto-run by queue]",
         facecolor="#FDE0DD", fontsize=8)

    _arrow(ax, (2.0, 1.9), (2.4, 2.6))
    _arrow(ax, (2.0, 1.9), (2.4, 1.0))
    _arrow(ax, (4.6, 2.6), (5.0, 2.6))
    _arrow(ax, (4.6, 1.0), (5.0, 1.0))

    ax.set_title("Figure 6 -- Clinical Evaluation Pipeline (per checkpoint)", fontsize=11)
    fig.tight_layout()
    out_path = OUT_DIR / "figure6_clinical_eval_pipeline.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def figure7_roadmap_timeline() -> Path:
    """Structural status snapshot, current as of this run -- reflects the
    actual per-stage status (Stage 1/2 complete, Stage 3 Phase 0 complete
    and coding ~done, GPU training in progress, RL/paper not started),
    not an idealized schedule. Regenerate this figure rather than
    treating an old copy as current -- status changes as work lands."""
    stages = [
        ("Stage 1: Diagnosis", 1.0),
        ("Stage 2: Architecture Investigation", 1.0),
        ("Stage 3 Phase 0 (pre-registration)", 1.0),
        ("Stage 3 candidate coding + fixes", 0.95),
        ("Stage 3 GPU training (S3-001..006)", 0.4),
        ("Cross-candidate comparison", 0.05),
        ("RL fine-tuning", 0.0),
        ("Paper", 0.0),
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = [s[0] for s in stages][::-1]
    values = [s[1] for s in stages][::-1]
    colors = ["#31A354" if v == 1.0 else ("#FEB24C" if 0 < v < 1.0 else "#D9D9D9") for v in values]
    ax.barh(labels, values, color=colors)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Fraction complete (snapshot, not a schedule)")
    ax.set_title("Figure 7 -- Roadmap Status Snapshot", fontsize=11)
    fig.tight_layout()
    out_path = OUT_DIR / "figure7_roadmap_timeline.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fn in (figure1_pipeline, figure2_architecture_comparison, figure3_stage2_findings,
               figure6_clinical_eval_pipeline, figure7_roadmap_timeline):
        path = fn()
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
