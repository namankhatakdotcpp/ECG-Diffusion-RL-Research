"""
Stage 1 / Experiment 1.5 — Checkpoint Verification ("did we stop too early?").

step04_transformer_diffusion.py already saves a checkpoint every
`save_every` epochs (config default 25 -> epoch025, epoch050, ..., epoch200
for a 200-epoch run) in addition to `diffusion_best.pt`. This experiment
costs almost nothing on top of Experiment 1 because no additional training
happens: it just loads each already-saved checkpoint and evaluates
conditioning quality at that point in training, so we can plot conditioning
strength vs. epoch. If the curve is still rising at epoch 200, training
stopped too early and more epochs (not more data, not a new architecture)
may be the fix Stage 1 should recommend.

Per checkpoint, computes:
  - val_loss (already stored in the checkpoint dict)
  - sensitivity_metric: mean ||eps_A - eps_B|| / eps_scale across all class
    pairs vs. class 0 (same magnitude-only measure as
    conditioning_sensitivity_probe.py, computed here per-checkpoint rather
    than once for the final model only)
  - collapse_frac and macro_f1 vs. the fixed MentorClassifier, from a SMALL
    generation pass (20 samples/class, not 100) — kept small because this
    runs once per checkpoint (up to 8 checkpoints for a 200-epoch run) and
    is meant to be cheap, not a replacement for Experiment 1's full-scale
    evaluation of the final checkpoint.

Writes to Roadmap/Stage_1_Diagnosis/Outputs/Experiment_1_Baseline/:
  checkpoint_verification.csv
Writes to Roadmap/Stage_1_Diagnosis/Figures/Experiment_1_Baseline/:
  conditioning_vs_epoch.png
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Code/ for common_probes

from utils import load_config, get_logger
from step04_transformer_diffusion import (
    ECGTransformerDiffusion, GaussianDiffusion, _resolve_device,
)
from mentor_eval.class_mapping import MENTOR_CLASSES, MENTOR_TO_TRAINED_CLASS
from mentor_eval.classification_validation import MentorClassifier
from common_probes import sensitivity_metric, collapse_and_macro_f1

OUT_DIR = REPO_ROOT / "Roadmap" / "Stage_1_Diagnosis" / "Outputs" / "Experiment_1_Baseline"
FIG_DIR = REPO_ROOT / "Roadmap" / "Stage_1_Diagnosis" / "Figures" / "Experiment_1_Baseline"
CACHED_CLF = REPO_ROOT / "outputs" / "conditioning_analysis" / "mentor_classifier.pt"
N_GEN_PER_CLASS = 20
SEED = 42


def main() -> None:
    cfg = load_config()
    log = get_logger("checkpoint_verification", cfg=cfg)
    device = _resolve_device(cfg)

    models_dir = Path(cfg.paths.outputs.models)
    ckpt_paths = sorted(models_dir.glob("diffusion_ckpt_ep*.pt"))
    if not ckpt_paths:
        print(f"[BLOCKED] No diffusion_ckpt_ep*.pt found in {models_dir}. "
              "Run Experiment 1 (step04_transformer_diffusion.py) first — "
              "it saves a checkpoint every save_every epochs automatically.")
        return

    if not CACHED_CLF.exists():
        print(f"[BLOCKED] No cached MentorClassifier at {CACHED_CLF}. "
              "Run Experiment 1's classification_validation or embedding_visualization "
              "step first (they train and cache it), or Experiment 3/4.")
        return
    clf_state = torch.load(str(CACHED_CLF), map_location=device)

    stats_path = Path(cfg.paths.outputs.processed) / "preprocessing_stats.json"
    prep_stats = json.load(open(stats_path)) if stats_path.exists() else None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for ckpt_path in ckpt_paths:
        m = re.search(r"ep(\d+)", ckpt_path.stem)
        epoch = int(m.group(1)) if m else None
        log.info(f"=== Checkpoint epoch {epoch} ({ckpt_path.name}) ===")

        ckpt = torch.load(str(ckpt_path), map_location=device)
        class_names = ckpt["class_names"]
        n_classes = ckpt["n_classes"]
        model = ECGTransformerDiffusion(cfg, n_classes=n_classes).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        diffusion = GaussianDiffusion(T=int(cfg.diffusion.T), beta_schedule=str(cfg.diffusion.beta_schedule), device=device)

        clf = MentorClassifier(n_classes=len(MENTOR_CLASSES)).to(device)
        clf.load_state_dict(clf_state)
        clf.eval()

        sens = sensitivity_metric(model, device, n_classes, cfg)
        collapse_frac, macro_f1 = collapse_and_macro_f1(
            model, diffusion, class_names, clf, device, cfg, prep_stats,
            MENTOR_CLASSES, MENTOR_TO_TRAINED_CLASS, N_GEN_PER_CLASS, SEED,
        )

        row = {
            "epoch": epoch, "val_loss": ckpt.get("val_loss"),
            "sensitivity_metric": sens, "collapse_frac": collapse_frac, "macro_f1": macro_f1,
        }
        rows.append(row)
        log.info(f"  {row}")
        del model, diffusion
        if device == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows).sort_values("epoch")
    df.to_csv(OUT_DIR / "checkpoint_verification.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, metric, title in [
        (axes[0, 0], "val_loss", "Validation loss vs epoch"),
        (axes[0, 1], "sensitivity_metric", "Conditioning magnitude (sensitivity probe) vs epoch"),
        (axes[1, 0], "macro_f1", "Generated-sample macro F1 vs epoch"),
        (axes[1, 1], "collapse_frac", "Collapse fraction vs epoch"),
    ]:
        ax.plot(df["epoch"], df[metric], marker="o")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
    fig.suptitle("Experiment 1.5 — Checkpoint Verification: was training stopped too early?")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "conditioning_vs_epoch.png", dpi=200)
    plt.close(fig)

    log.info(f"Done. See {OUT_DIR / 'checkpoint_verification.csv'} and {FIG_DIR / 'conditioning_vs_epoch.png'}")
    print(f"✓ Checkpoint verification complete. See {OUT_DIR} and {FIG_DIR}")


if __name__ == "__main__":
    main()
