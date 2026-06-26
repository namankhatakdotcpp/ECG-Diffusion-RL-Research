"""
mentor_eval/training_progression.py — visual training progression across
intermediate diffusion checkpoints.

Auto-discovers whichever diffusion_ckpt_ep*.pt files actually exist under
outputs/models/ (step04 saves one every cfg.diffusion.save_every epochs —
25 by default). For each discovered epoch, generates ONE sample from the
SAME fixed seed and class so the comparison is fair, and lays them out as a
single multi-panel figure (one panel per epoch, one lead).

REQUIRES at least 2 intermediate checkpoints. If outputs/models/ has none
(true on this local machine — training only happened on the GPU server),
this prints a clear message and exits without fabricating a progression.

Writes:
  outputs/mentor_review/training_progression/progression_<class>_<lead>.png

Usage:
    python -m mentor_eval.training_progression [--class-name NORM] [--lead II] [--seed 42]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_config, get_logger, set_seed
from utils.backup import snapshot_before_write
from mentor_eval.checkpoint_utils import load_checkpoint, generate_for_class

PUBSTYLE = {
    "font.size": 11, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
}
_EPOCH_RE = re.compile(r"diffusion_ckpt_ep(\d+)\.pt$")


def discover_checkpoints(models_dir: Path) -> list[tuple[int, Path]]:
    """Return [(epoch, path), ...] sorted by epoch, for every diffusion_ckpt_ep*.pt found."""
    found = []
    for p in sorted(Path(models_dir).glob("diffusion_ckpt_ep*.pt")):
        m = _EPOCH_RE.search(p.name)
        if m:
            found.append((int(m.group(1)), p))
    return sorted(found, key=lambda t: t[0])


def run(models_dir: Path, out_dir: Path, class_name: str, lead_name: str,
        lead_names: list[str], cfg, seed: int, log) -> None:
    checkpoints = discover_checkpoints(models_dir)

    if len(checkpoints) < 2:
        print(
            f"\n[BLOCKED] Found {len(checkpoints)} intermediate checkpoint(s) "
            f"(diffusion_ckpt_ep*.pt) under {models_dir}.\n"
            f"  Need at least 2 to show a training progression.\n"
            f"  This local machine never ran step04 training — checkpoints only\n"
            f"  exist on the GPU server. Re-run this script there.\n"
            f"  No progression figure was written — nothing fabricated.\n"
        )
        sys.exit(1)

    epochs = [e for e, _ in checkpoints]
    log.info(f"Found {len(checkpoints)} checkpoints at epochs: {epochs}")

    stats_path = Path(cfg.paths.outputs.processed) / "preprocessing_stats.json"
    stats = None
    if stats_path.exists():
        import json
        stats = json.load(open(stats_path))

    lead_idx = lead_names.index(lead_name)
    fs = 100.0
    time_axis = np.arange(1000) / fs

    panels = []
    for epoch, ckpt_path in checkpoints:
        loaded = load_checkpoint(ckpt_path, cfg)
        if loaded is None:
            continue
        samples, err = generate_for_class(loaded, class_name, n_samples=1, cfg=cfg, seed=seed, stats=stats)
        if err:
            log.warning(f"Epoch {epoch}: {err}")
            continue
        panels.append((epoch, samples[0][:, lead_idx]))

    if len(panels) < 2:
        print(
            f"\n[BLOCKED] Could not generate samples for class '{class_name}' from "
            f"enough checkpoints (only {len(panels)} succeeded). No figure written.\n"
        )
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(PUBSTYLE):
        fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3), sharey=True)
        if len(panels) == 1:
            axes = [axes]
        for ax, (epoch, sig) in zip(axes, panels):
            ax.plot(time_axis, sig, color="#1B3A6B", linewidth=0.9)
            ax.set_title(f"Epoch {epoch}")
            ax.set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")
        fig.suptitle(
            f"Training progression — class {class_name}, Lead {lead_name} "
            f"(same seed={seed} across all checkpoints)", fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        out_path = out_dir / f"progression_{class_name}_{lead_name}.png"
        fig.savefig(str(out_path))
        plt.close(fig)
    log.info(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diffusion training progression across checkpoints.")
    parser.add_argument("--models-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--class-name", type=str, default="NORM", help="Trained model class name (NORM/MI/STTC/CD/HYP/OTHER)")
    parser.add_argument("--lead", type=str, default="II")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("training_progression", cfg=cfg)
    set_seed(args.seed)

    models_dir = Path(args.models_dir) if args.models_dir else Path(cfg.paths.outputs.models)
    out_dir     = Path(args.out_dir) if args.out_dir else Path(cfg.paths.outputs.results).parent / "mentor_review" / "training_progression"
    snapshot_before_write(out_dir)
    lead_names  = list(cfg.ptbxl.lead_names)

    run(models_dir, out_dir, args.class_name, args.lead, lead_names, cfg, args.seed, log)
    print(f"✓ Training progression figure written to {out_dir}")


if __name__ == "__main__":
    main()
