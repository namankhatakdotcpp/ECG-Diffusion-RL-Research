"""
mentor_eval/lead_class_figures.py — per-lead, per-class real-ECG comparison figures.

For each of the 12 leads, produce ONE figure with 4 vertically-stacked
subplots: Normal / STEMI / NSTEMI / AFIB (one representative real ECG per
class, single clean trace, no overlapping colors).

Reads real PTB-XL signals directly (not the model's training class scheme —
see mentor_eval/class_mapping.py for why).

Writes:
  outputs/mentor_review/lead_class_figures/lead_I.png ... lead_V6.png

Usage:
    python -m mentor_eval.lead_class_figures [--ptbxl-dir PATH] [--out-dir PATH]
                                              [--n-per-class N]
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
from mentor_eval.class_mapping import (
    MENTOR_CLASSES, load_ptbxl_database, filter_to_mentor_classes,
)

PUBSTYLE = {
    "font.size": 11, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
}
CLASS_COLOR = {
    "Normal": "#1B3A6B", "STEMI": "#C0392B", "NSTEMI": "#C9872A", "AFIB": "#8E44AD",
}


def _load_signal(ptbxl_dir: Path, filename_lr: str) -> np.ndarray:
    record = wfdb.rdrecord(str(ptbxl_dir / filename_lr))
    return record.p_signal  # (1000, 12)


def _pick_one_record_per_class(
    filtered_db, ptbxl_dir: Path, rng: np.random.Generator, log,
) -> dict[str, np.ndarray]:
    """Return {class_name: (1000, 12) signal} for one representative, readable record per class."""
    picked: dict[str, np.ndarray] = {}
    for cls in MENTOR_CLASSES:
        candidates = filtered_db[filtered_db["mentor_class"] == cls]
        if candidates.empty:
            log.warning(f"No records found for class {cls} — skipping in figures.")
            continue
        order = rng.permutation(len(candidates))
        for idx in order:
            rec = candidates.iloc[int(idx)]
            try:
                sig = _load_signal(ptbxl_dir, str(rec["filename_lr"]))
                if sig.shape == (1000, 12) and np.isfinite(sig).all():
                    picked[cls] = sig
                    break
            except Exception:
                continue
        if cls not in picked:
            log.warning(f"Could not find a readable record for class {cls}.")
    return picked


def make_lead_class_figures(
    ptbxl_dir: Path, out_dir: Path, lead_names: list[str], seed: int, log,
) -> None:
    db = load_ptbxl_database(ptbxl_dir)
    filtered = filter_to_mentor_classes(db)
    log.info(f"Class counts available: {dict(filtered['mentor_class'].value_counts())}")

    rng = np.random.default_rng(seed)
    picked = _pick_one_record_per_class(filtered, ptbxl_dir, rng, log)
    if not picked:
        raise RuntimeError("Could not load any representative records — aborting.")

    out_dir.mkdir(parents=True, exist_ok=True)
    fs = 100.0
    time_axis = np.arange(1000) / fs

    with plt.rc_context(PUBSTYLE):
        for lead_idx, lead_name in enumerate(lead_names):
            present_classes = [c for c in MENTOR_CLASSES if c in picked]
            fig, axes = plt.subplots(
                len(present_classes), 1, figsize=(12, 2.2 * len(present_classes)), sharex=True,
            )
            if len(present_classes) == 1:
                axes = [axes]

            for ax, cls in zip(axes, present_classes):
                ax.plot(time_axis, picked[cls][:, lead_idx], color=CLASS_COLOR[cls], linewidth=0.9)
                ax.set_ylabel(cls, fontsize=11, fontweight="bold")
                ax.set_ylim(-2.5, 2.5)

            axes[-1].set_xlabel("Time (s)")
            fig.suptitle(f"Lead {lead_name} — Normal vs. disease classes (real PTB-XL ECGs)", fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, 0.96))

            out_path = out_dir / f"lead_{lead_name}.png"
            fig.savefig(str(out_path))
            plt.close(fig)
            log.info(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-lead, per-class real-ECG comparison figures.")
    parser.add_argument("--ptbxl-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("lead_class_figures", cfg=cfg)
    set_seed(args.seed)

    ptbxl_dir = Path(args.ptbxl_dir) if args.ptbxl_dir else Path(cfg.paths.data.ptbxl)
    out_dir   = Path(args.out_dir) if args.out_dir else Path(cfg.paths.outputs.results).parent / "mentor_review" / "lead_class_figures"
    lead_names = list(cfg.ptbxl.lead_names)

    make_lead_class_figures(ptbxl_dir, out_dir, lead_names, args.seed, log)
    print(f"✓ 12 lead-comparison figures written to {out_dir}")


if __name__ == "__main__":
    main()
