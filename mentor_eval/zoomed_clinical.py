"""
mentor_eval/zoomed_clinical.py — zoomed clinical-region plots around one beat.

Given a 12-lead ECG, detects R-peaks and delineates QRS/ST-segment/T-wave
boundaries with neurokit2 (nk.ecg_peaks + nk.ecg_delineate), rather than
hardcoding fixed-offset windows. R-peak/QRS timing is shared across leads
within one heartbeat, so delineation is run once on Lead II (best R-peak
SNR) and the resulting boundaries are reused to window every other lead.

If delineation fails for a given beat (common on noisy/abnormal beats,
e.g. AFIB has no stable QRS-T morphology), this falls back to documented
physiological offsets relative to the R-peak (literature values, not
arbitrary guesses) and LOGS that the fallback was used — it does not
silently fabricate a boundary.

Produces, per class, a 3-panel zoomed figure (QRS / ST segment / T wave)
for one representative beat in a chosen lead.

Writes:
  outputs/mentor_review/zoomed_clinical/<class>_<lead>_zoomed.png

Usage:
    python -m mentor_eval.zoomed_clinical [--lead V1] [--ptbxl-dir PATH] [--out-dir PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_config, get_logger, set_seed
from utils.backup import snapshot_before_write
from mentor_eval.class_mapping import (
    MENTOR_CLASSES, load_ptbxl_database, filter_to_mentor_classes,
)

LEAD_II_IDX = 1  # standard lead for R-peak/QRS-T delineation

# Literature fallback offsets relative to R-peak (seconds), used only when
# neurokit2 delineation fails for a given beat.
_FALLBACK_OFFSETS_SEC = {
    "qrs_start": -0.04, "qrs_end": 0.04,     # QRS ~80ms, centred on R
    "st_start":  0.04,  "st_end":  0.16,     # J-point to T-onset, ~80-120ms
    "t_start":   0.16,  "t_end":   0.40,
}


class BeatBoundaries:
    def __init__(self, r_peak_idx: int, qrs_start: int, qrs_end: int,
                 st_start: int, st_end: int, t_start: int, t_end: int,
                 used_fallback: bool):
        self.r_peak_idx = r_peak_idx
        self.qrs_start, self.qrs_end = qrs_start, qrs_end
        self.st_start, self.st_end = st_start, st_end
        self.t_start, self.t_end = t_start, t_end
        self.used_fallback = used_fallback


def delineate_one_beat(lead_ii_signal: np.ndarray, fs: float, log) -> Optional[BeatBoundaries]:
    """Detect R-peaks on Lead II and delineate QRS/ST/T boundaries for the
    first clean beat. Returns None if no R-peak could be found at all.
    """
    try:
        _, r_info = nk.ecg_peaks(lead_ii_signal, sampling_rate=int(fs))
        r_peaks = r_info["ECG_R_Peaks"]
    except Exception as exc:
        log.warning(f"nk.ecg_peaks failed: {exc}")
        return None

    if len(r_peaks) < 2:
        log.warning("Fewer than 2 R-peaks detected — cannot delineate a full beat.")
        return None

    r_idx = int(r_peaks[1])  # skip first beat (often clipped at signal start)

    try:
        # "dwt" (discrete wavelet transform) delineation gives full onset/offset
        # boundaries for QRS (R_Onsets/R_Offsets) and T-wave (T_Onsets/T_Offsets)
        # directly — unlike "peak" mode, which only returns peak locations.
        _, wave_info = nk.ecg_delineate(
            lead_ii_signal, r_info, sampling_rate=int(fs), method="dwt",
        )
        qrs_starts = np.asarray(wave_info.get("ECG_R_Onsets", []), dtype=float)
        qrs_ends = np.asarray(wave_info.get("ECG_R_Offsets", []), dtype=float)
        t_onsets = np.asarray(wave_info.get("ECG_T_Onsets", []), dtype=float)
        t_offsets = np.asarray(wave_info.get("ECG_T_Offsets", []), dtype=float)

        def _nearest_after(arr: np.ndarray, ref: int) -> Optional[int]:
            cand = arr[(arr >= ref) & np.isfinite(arr)]
            return int(cand.min()) if len(cand) else None

        def _nearest_before(arr: np.ndarray, ref: int) -> Optional[int]:
            cand = arr[(arr <= ref) & np.isfinite(arr)]
            return int(cand.max()) if len(cand) else None

        qrs_start = _nearest_before(qrs_starts, r_idx)
        qrs_end = _nearest_after(qrs_ends, r_idx)
        t_on = _nearest_after(t_onsets, r_idx) if qrs_end is None else _nearest_after(t_onsets, qrs_end)
        t_off = _nearest_after(t_offsets, r_idx) if t_on is None else _nearest_after(t_offsets, t_on)

        if None not in (qrs_start, qrs_end, t_on, t_off) and qrs_start < r_idx < qrs_end < t_on < t_off:
            return BeatBoundaries(
                r_peak_idx=r_idx,
                qrs_start=qrs_start, qrs_end=qrs_end,
                st_start=qrs_end, st_end=t_on,
                t_start=t_on, t_end=t_off,
                used_fallback=False,
            )
        log.info("Delineation incomplete/inconsistent for this beat — using literature fallback offsets.")
    except Exception as exc:
        log.info(f"nk.ecg_delineate failed ({exc}) — using literature fallback offsets.")

    off = _FALLBACK_OFFSETS_SEC
    return BeatBoundaries(
        r_peak_idx=r_idx,
        qrs_start=r_idx + int(off["qrs_start"] * fs), qrs_end=r_idx + int(off["qrs_end"] * fs),
        st_start=r_idx + int(off["st_start"] * fs),   st_end=r_idx + int(off["st_end"] * fs),
        t_start=r_idx + int(off["t_start"] * fs),     t_end=r_idx + int(off["t_end"] * fs),
        used_fallback=True,
    )


def plot_zoomed_regions(
    signal: np.ndarray, lead_idx: int, lead_name: str, bounds: BeatBoundaries,
    fs: float, class_name: str, color: str, out_path: Path,
) -> None:
    """3-panel zoomed plot: QRS / ST segment / T wave, single lead, single beat."""
    sig = signal[:, lead_idx]
    pad = int(0.02 * fs)  # 20 ms padding so boundaries aren't clipped at panel edges

    windows = [
        ("QRS complex", bounds.qrs_start, bounds.qrs_end),
        ("ST segment",  bounds.st_start,  bounds.st_end),
        ("T wave",      bounds.t_start,   bounds.t_end),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    for ax, (title, start, end) in zip(axes, windows):
        lo, hi = max(0, start - pad), min(len(sig), end + pad)
        t_ms = (np.arange(lo, hi) - bounds.r_peak_idx) / fs * 1000.0
        ax.plot(t_ms, sig[lo:hi], color=color, linewidth=1.4)
        ax.axvspan((start - bounds.r_peak_idx) / fs * 1000.0,
                   (end - bounds.r_peak_idx) / fs * 1000.0,
                   color=color, alpha=0.12)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("ms from R-peak")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

    axes[0].set_ylabel("mV (z-scored)")
    fallback_note = "  [fallback offsets used]" if bounds.used_fallback else "  [neurokit2 delineation]"
    fig.suptitle(f"{class_name} — Lead {lead_name} — zoomed clinical regions{fallback_note}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(ptbxl_dir: Path, out_dir: Path, lead_name: str, lead_names: list[str],
        seed: int, log) -> None:
    lead_idx = lead_names.index(lead_name)
    fs = 100.0

    db = load_ptbxl_database(ptbxl_dir)
    filtered = filter_to_mentor_classes(db)

    color_map = {"Normal": "#1B3A6B", "STEMI": "#C0392B", "NSTEMI": "#C9872A", "AFIB": "#8E44AD"}
    rng = np.random.default_rng(seed)

    for cls in MENTOR_CLASSES:
        candidates = filtered[filtered["mentor_class"] == cls]
        if candidates.empty:
            log.warning(f"No records for class {cls} — skipping.")
            continue

        order = rng.permutation(len(candidates))
        done = False
        for idx in order[:30]:  # try up to 30 records before giving up on this class
            rec = candidates.iloc[int(idx)]
            try:
                record = wfdb.rdrecord(str(ptbxl_dir / str(rec["filename_lr"])))
                sig = record.p_signal
            except Exception:
                continue
            if sig.shape != (1000, 12) or not np.isfinite(sig).all():
                continue

            bounds = delineate_one_beat(sig[:, LEAD_II_IDX], fs, log)
            if bounds is None:
                continue

            out_path = out_dir / f"{cls}_{lead_name}_zoomed.png"
            plot_zoomed_regions(sig, lead_idx, lead_name, bounds, fs, cls, color_map[cls], out_path)
            log.info(f"Saved {out_path}  (fallback={bounds.used_fallback})")
            done = True
            break

        if not done:
            log.warning(f"Could not delineate any beat for class {cls} after 30 attempts.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Zoomed QRS/ST/T-wave plots per class.")
    parser.add_argument("--lead", type=str, default="V1", help="Lead to zoom into (e.g. V1)")
    parser.add_argument("--ptbxl-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("zoomed_clinical", cfg=cfg)
    set_seed(args.seed)

    ptbxl_dir = Path(args.ptbxl_dir) if args.ptbxl_dir else Path(cfg.paths.data.ptbxl)
    out_dir   = Path(args.out_dir) if args.out_dir else Path(cfg.paths.outputs.results).parent / "mentor_review" / "zoomed_clinical"
    snapshot_before_write(out_dir)
    lead_names = list(cfg.ptbxl.lead_names)

    run(ptbxl_dir, out_dir, args.lead, lead_names, args.seed, log)
    print(f"✓ Zoomed clinical-region plots written to {out_dir}")


if __name__ == "__main__":
    main()
