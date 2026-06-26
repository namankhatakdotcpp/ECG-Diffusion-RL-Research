"""
mentor_eval/real_vs_generated.py — real PTB-XL ECG vs. model-generated ECG,
side by side, per class, per lead.

REQUIRES outputs/models/diffusion_best.pt (trained checkpoint). This repo's
local machine does not have one — training only happened on the GPU server.
Running this script before that checkpoint exists will print a clear error
and exit; it will NOT fabricate placeholder figures.

The generated side uses the TRAINED model's class scheme (NORM/MI/STTC/CD/
HYP/OTHER), bridged from the mentor's 4 classes via
mentor_eval.class_mapping.MENTOR_TO_TRAINED_CLASS. AFIB has no bridge (the
model never learned a distinct AFIB class) — that panel will say so
explicitly instead of silently using OTHER.

Writes:
  outputs/mentor_review/real_vs_generated/<class>/<lead>.png

Usage:
    python -m mentor_eval.real_vs_generated [--ckpt PATH] [--out-dir PATH] [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_config, get_logger, set_seed
from utils.backup import snapshot_before_write
from mentor_eval.class_mapping import (
    MENTOR_CLASSES, MENTOR_TO_TRAINED_CLASS, load_ptbxl_database, filter_to_mentor_classes,
)
from mentor_eval.checkpoint_utils import load_checkpoint, generate_for_class

PUBSTYLE = {
    "font.size": 11, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
}


def _pick_real_signal(candidates, ptbxl_dir: Path, rng: np.random.Generator):
    order = rng.permutation(len(candidates))
    for idx in order[:30]:
        rec = candidates.iloc[int(idx)]
        try:
            sig = wfdb.rdrecord(str(ptbxl_dir / str(rec["filename_lr"]))).p_signal
        except Exception:
            continue
        if sig.shape == (1000, 12) and np.isfinite(sig).all():
            return sig
    return None


def run(ckpt_path: Path, out_dir: Path, lead_names: list[str], cfg, seed: int, log) -> None:
    loaded = load_checkpoint(ckpt_path, cfg)
    if loaded is None:
        print(
            f"\n[BLOCKED] No checkpoint found at {ckpt_path}.\n"
            f"  This script needs the trained diffusion model to generate samples.\n"
            f"  Train on the GPU server (step04_transformer_diffusion.py), then re-run this\n"
            f"  script there (or copy diffusion_best.pt back here).\n"
            f"  No figures were written — nothing fabricated.\n"
        )
        sys.exit(1)
    log.info(f"Loaded checkpoint: epoch={loaded.epoch}, val_loss={loaded.val_loss}, classes={loaded.class_names}")

    ptbxl_dir = Path(cfg.paths.data.ptbxl)
    db = load_ptbxl_database(ptbxl_dir)
    filtered = filter_to_mentor_classes(db)
    rng = np.random.default_rng(seed)

    stats_path = Path(cfg.paths.outputs.processed) / "preprocessing_stats.json"
    stats = None
    if stats_path.exists():
        import json
        stats = json.load(open(stats_path))

    fs = 100.0
    time_axis = np.arange(1000) / fs

    with plt.rc_context(PUBSTYLE):
        for cls in MENTOR_CLASSES:
            class_dir = out_dir / cls
            class_dir.mkdir(parents=True, exist_ok=True)

            candidates = filtered[filtered["mentor_class"] == cls]
            if candidates.empty:
                log.warning(f"No real records for class {cls} — skipping.")
                continue
            real_sig = _pick_real_signal(candidates, ptbxl_dir, rng)
            if real_sig is None:
                log.warning(f"Could not load a readable real record for class {cls} — skipping.")
                continue

            trained_cls = MENTOR_TO_TRAINED_CLASS.get(cls)
            if trained_cls is None:
                note = (
                    f"'{cls}' has no dedicated class in the trained model "
                    f"(merged into OTHER in step03 — not enough samples). "
                    f"Cannot generate a true '{cls}' sample."
                )
                log.warning(note)
                (class_dir / "GENERATION_NOT_AVAILABLE.txt").write_text(note + "\n")
                continue

            gen_samples, err = generate_for_class(
                loaded, trained_cls, n_samples=1, cfg=cfg, seed=seed, stats=stats,
            )
            if err:
                log.warning(err)
                (class_dir / "GENERATION_NOT_AVAILABLE.txt").write_text(err + "\n")
                continue
            gen_sig = gen_samples[0]  # (1000, 12)

            y_lo = min(real_sig.min(), gen_sig.min())
            y_hi = max(real_sig.max(), gen_sig.max())

            for lead_idx, lead_name in enumerate(lead_names):
                fig, axes = plt.subplots(1, 2, figsize=(11, 2.6), sharey=True)
                axes[0].plot(time_axis, real_sig[:, lead_idx], color="#1B3A6B", linewidth=0.9)
                axes[0].set_title("Real (PTB-XL)")
                axes[1].plot(time_axis, gen_sig[:, lead_idx], color="#C0392B", linewidth=0.9)
                axes[1].set_title(f"Generated (model class: {trained_cls})")
                for ax in axes:
                    ax.set_xlabel("Time (s)")
                    ax.set_ylim(y_lo - 0.1, y_hi + 0.1)
                axes[0].set_ylabel("Amplitude (mV)" if stats else "Amplitude (z-score)")
                fig.suptitle(f"{cls} — Lead {lead_name}", fontsize=12)
                fig.tight_layout(rect=(0, 0, 1, 0.92))
                fig.savefig(str(class_dir / f"{lead_name}.png"))
                plt.close(fig)
            log.info(f"Saved 12 lead comparison figures for class {cls} → {class_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Real vs. generated ECG comparison, per class/lead.")
    parser.add_argument("--ckpt", type=str, default=None, help="Override path to diffusion_best.pt")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("real_vs_generated", cfg=cfg)
    set_seed(args.seed)

    ckpt_path = Path(args.ckpt) if args.ckpt else Path(cfg.paths.outputs.models) / "diffusion_best.pt"
    out_dir   = Path(args.out_dir) if args.out_dir else Path(cfg.paths.outputs.results).parent / "mentor_review" / "real_vs_generated"
    snapshot_before_write(out_dir)
    lead_names = list(cfg.ptbxl.lead_names)

    run(ckpt_path, out_dir, lead_names, cfg, args.seed, log)
    print(f"✓ Real-vs-generated comparison figures written to {out_dir}")


if __name__ == "__main__":
    main()
