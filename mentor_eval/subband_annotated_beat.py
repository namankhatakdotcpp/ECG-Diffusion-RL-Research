"""
mentor_eval/subband_annotated_beat.py — annotated single-beat figure,
mirroring Sharma et al.'s Fig. 2: one trace per class, with the T-wave,
ST-segment, and QRS/Q-wave regions circled (dashed ellipse) and labeled
directly on the plot.

Reuses mentor_eval.zoomed_clinical.delineate_one_beat (the existing
neurokit2 R-peak + wavelet delineation already built for item 5) to get
beat boundaries — this module does NOT redetect anything, it only draws
annotations over boundaries already computed by that function.

Writes:
  outputs/sharma_inspired_analysis/annotated_beat_<class>.png

Usage:
    python -m mentor_eval.subband_annotated_beat [--lead V1] [--out-dir PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_config, get_logger, set_seed
from utils.backup import snapshot_before_write
from mentor_eval.class_mapping import (
    MENTOR_CLASSES, load_ptbxl_database, filter_to_mentor_classes,
)
from mentor_eval.zoomed_clinical import delineate_one_beat, LEAD_II_IDX
from mentor_eval.subband_features import subband_output_dir

PUBSTYLE = {
    "font.size": 11, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
}
CLASS_COLOR = {"Normal": "#1B3A6B", "STEMI": "#C0392B", "NSTEMI": "#C9872A", "AFIB": "#8E44AD"}

# Window shown around the beat, wide enough to display all 3 annotated
# regions plus context — matches the spirit of Sharma Fig. 2's full-trace view.
WINDOW_PAD_SEC = 0.30


def _annotate_region(ax, t_ms: np.ndarray, sig_window: np.ndarray, start_idx: int, end_idx: int,
                      r_peak_idx: int, fs: float, label: str, color: str,
                      ellipse_height: float, label_y: float) -> None:
    """Draw a dashed ellipse around [start_idx, end_idx) (sample indices into
    the full signal) and label it, matching Sharma Fig. 2's annotation style.

    ellipse_height is shared across all 3 regions (a fixed fraction of the
    panel's total y-range) so QRS's large swing doesn't dwarf the much
    smaller ST/T regions — only the center and width vary per region.
    """
    t0_ms = (start_idx - r_peak_idx) / fs * 1000.0
    t1_ms = (end_idx - r_peak_idx) / fs * 1000.0
    center_t = (t0_ms + t1_ms) / 2.0

    mask = (t_ms >= t0_ms) & (t_ms <= t1_ms)
    if not mask.any():
        return
    y_vals = sig_window[mask]
    center_y = (y_vals.max() + y_vals.min()) / 2.0
    width = max(t1_ms - t0_ms, 30.0) + 25.0

    ax.add_patch(Ellipse((center_t, center_y), width=width, height=ellipse_height,
                          fill=False, linestyle="--", edgecolor=color, linewidth=1.6))
    ax.annotate(
        label, xy=(center_t, center_y), xytext=(center_t, label_y),
        ha="center", fontsize=9, color=color, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=color, linewidth=1.0),
    )


def make_annotated_beat_figure(signal: np.ndarray, lead_idx: int, lead_name: str,
                                bounds, fs: float, class_name: str, color: str, out_path: Path) -> None:
    pad = int(WINDOW_PAD_SEC * fs)
    lo = max(0, bounds.qrs_start - pad)
    hi = min(len(signal), bounds.t_end + pad)
    sig_window = signal[lo:hi, lead_idx]
    t_ms = (np.arange(lo, hi) - bounds.r_peak_idx) / fs * 1000.0

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t_ms, sig_window, color=color, linewidth=1.2)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

    y_range = sig_window.max() - sig_window.min()
    ellipse_height = max(y_range * 0.45, 0.15)
    label_y = sig_window.max() + y_range * 0.35

    _annotate_region(ax, t_ms, sig_window, bounds.qrs_start, bounds.qrs_end,
                      bounds.r_peak_idx, fs, "QRS / Q-wave", "#444444",
                      ellipse_height, label_y + y_range * 0.30)
    _annotate_region(ax, t_ms, sig_window, bounds.st_start, bounds.st_end,
                      bounds.r_peak_idx, fs, "ST-segment", "#C0392B",
                      ellipse_height, label_y)
    _annotate_region(ax, t_ms, sig_window, bounds.t_start, bounds.t_end,
                      bounds.r_peak_idx, fs, "T-wave", "#1B3A6B",
                      ellipse_height, label_y + y_range * 0.15)

    ax.set_ylim(sig_window.min() - y_range * 0.2, label_y + y_range * 0.55)
    ax.set_xlabel("ms from R-peak")
    ax.set_ylabel("Amplitude (z-scored)")
    fallback_note = " [fallback offsets]" if bounds.used_fallback else " [neurokit2 delineation]"
    ax.set_title(f"{class_name} — Lead {lead_name} — annotated beat{fallback_note}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path))
    plt.close(fig)


def run(ptbxl_dir: Path, out_dir: Path, lead_name: str, lead_names: list[str], seed: int, log) -> None:
    lead_idx = lead_names.index(lead_name)
    fs = 100.0

    db = load_ptbxl_database(ptbxl_dir)
    filtered = filter_to_mentor_classes(db)
    rng = np.random.default_rng(seed)

    for cls in MENTOR_CLASSES:
        candidates = filtered[filtered["mentor_class"] == cls]
        if candidates.empty:
            log.warning(f"No records for class {cls} — skipping.")
            continue

        order = rng.permutation(len(candidates))
        done = False
        for idx in order[:30]:
            rec = candidates.iloc[int(idx)]
            try:
                sig = wfdb.rdrecord(str(ptbxl_dir / str(rec["filename_lr"]))).p_signal
            except Exception:
                continue
            if sig.shape != (1000, 12) or not np.isfinite(sig).all():
                continue

            bounds = delineate_one_beat(sig[:, LEAD_II_IDX], fs, log)
            if bounds is None:
                continue

            out_path = out_dir / f"annotated_beat_{cls}.png"
            make_annotated_beat_figure(sig, lead_idx, lead_name, bounds, fs, cls, CLASS_COLOR[cls], out_path)
            log.info(f"Saved {out_path}  (fallback={bounds.used_fallback})")
            done = True
            break

        if not done:
            log.warning(f"Could not delineate any beat for class {cls} after 30 attempts.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotated single-beat figure per class (Sharma Fig. 2 style).")
    parser.add_argument("--lead", type=str, default="V1")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("subband_annotated_beat", cfg=cfg)
    set_seed(args.seed)

    ptbxl_dir = Path(cfg.paths.data.ptbxl)
    out_dir = Path(args.out_dir) if args.out_dir else subband_output_dir(cfg)
    snapshot_before_write(out_dir)
    lead_names = list(cfg.ptbxl.lead_names)

    run(ptbxl_dir, out_dir, args.lead, lead_names, args.seed, log)
    print(f"✓ Annotated beat figures written to {out_dir}")


if __name__ == "__main__":
    main()
