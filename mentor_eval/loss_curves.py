"""
mentor_eval/loss_curves.py — train/validation loss vs. epoch.

step04_transformer_diffusion.py ALREADY logs both train_loss and val_loss to
logs/diffusion_training_log.csv (one row per log_interval steps with
train_loss only, plus one row per save_every epochs with both train_loss
and val_loss filled in). No change to step04 was needed — validation loss
logging already existed; this script just parses and plots it.

REQUIRES logs/diffusion_training_log.csv to exist. This local machine has
never run step04 training (logs/ only has .gitkeep) — re-run this script
on the GPU server where the actual training log lives.

Writes:
  outputs/mentor_review/loss_curves/train_val_loss.png

Usage:
    python -m mentor_eval.loss_curves [--log-path PATH] [--out-dir PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_config, get_logger
from utils.backup import snapshot_before_write

PUBSTYLE = {
    "font.size": 11, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
}


def run(log_path: Path, out_dir: Path, log) -> None:
    if not log_path.exists():
        print(
            f"\n[BLOCKED] No training log found at {log_path}.\n"
            f"  step04_transformer_diffusion.py writes this file during training\n"
            f"  (logs/diffusion_training_log.csv). This local machine has never run\n"
            f"  step04 — training only happened on the GPU server. Re-run this script\n"
            f"  there, or copy diffusion_training_log.csv back here.\n"
            f"  No figure was written — nothing fabricated.\n"
        )
        sys.exit(1)

    df = pd.read_csv(log_path)
    train_rows = df.dropna(subset=["train_loss"])
    val_rows = df.dropna(subset=["val_loss"]) if "val_loss" in df.columns else pd.DataFrame()

    if val_rows.empty:
        log.warning(
            "val_loss column is present but every row is empty — "
            "this would mean validation loss logging broke during this run. Flagging, not fabricating."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(PUBSTYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        # Per-step train loss (light), per-epoch train loss at save points (line)
        ax.plot(train_rows["step"], train_rows["train_loss"], color="#1B3A6B",
                alpha=0.25, linewidth=0.6, label="train_loss (per step)")
        epoch_train = df.dropna(subset=["val_loss"])[["epoch", "train_loss"]] if not val_rows.empty else pd.DataFrame()
        if not epoch_train.empty:
            ax.plot(epoch_train["epoch"], epoch_train["train_loss"], color="#1B3A6B",
                    marker="o", linewidth=1.6, label="train_loss (at validation epochs)")
        if not val_rows.empty:
            ax.plot(val_rows["epoch"], val_rows["val_loss"], color="#C0392B",
                    marker="o", linewidth=1.6, label="val_loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE, noise-prediction)")
        ax.set_title("Diffusion model training — train vs. validation loss")
        ax.legend()
        out_path = out_dir / "train_val_loss.png"
        fig.savefig(str(out_path))
        plt.close(fig)
    log.info(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot diffusion training/validation loss curves.")
    parser.add_argument("--log-path", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("loss_curves", cfg=cfg)

    log_path = Path(args.log_path) if args.log_path else Path(cfg.paths.logs) / "diffusion_training_log.csv"
    out_dir  = Path(args.out_dir) if args.out_dir else Path(cfg.paths.outputs.results).parent / "mentor_review" / "loss_curves"
    snapshot_before_write(out_dir)
    run(log_path, out_dir, log)
    print(f"✓ Loss curve written to {out_dir}")


if __name__ == "__main__":
    main()
