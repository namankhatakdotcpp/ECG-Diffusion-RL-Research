"""
mentor_eval/subband_decomposition.py — Sharma-style multiscale energy
analysis (mirrors their Section II.A.2 / Fig. 7).

ITEM 1 — subband energy table
  For each class (Normal/STEMI/NSTEMI/AFIB) and lead, compute mean + variance
  of per-record subband energy across many real records — same structure as
  the paper's reported numbers ("For cA6 scale, mean and variance values for
  lead-I signal are 0.2 and 0.044 ... for MI"). Runs on real PTB-XL data only.

ITEM 2 — within-class variation box plots (mirrors Fig. 7)
  One figure per subband (4 figures total), each with REAL vs. GENERATED box
  plots faceted by class (Normal/STEMI/NSTEMI — AFIB excluded, no trained
  model class, per mentor_eval/class_mapping.py). REQUIRES
  outputs/models/diffusion_best.pt; prints [BLOCKED] and writes nothing if
  absent.

Writes:
  outputs/sharma_inspired_analysis/subband_energy_table.csv
  outputs/sharma_inspired_analysis/boxplot_<subband>.png

Usage:
    python -m mentor_eval.subband_decomposition [--ckpt PATH] [--out-dir PATH]
                                                  [--n-per-class N] [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_config, get_logger, set_seed
from mentor_eval.class_mapping import (
    MENTOR_CLASSES, MENTOR_TO_TRAINED_CLASS, load_ptbxl_database, filter_to_mentor_classes,
)
from mentor_eval.checkpoint_utils import load_checkpoint, generate_for_class
from mentor_eval.subband_features import (
    SUBBAND_NAMES, SUBBAND_CLINICAL_LABEL, subband_frequency_ranges, subband_energy_per_lead,
    subband_output_dir,
)

PUBSTYLE = {
    "font.size": 11, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
}
BOX_CLASSES = ["Normal", "STEMI", "NSTEMI"]  # AFIB excluded - no trained model class


# ──────────────────────────────────────────────────────────────────────────────
# Item 1 — subband energy table (real data)
# ──────────────────────────────────────────────────────────────────────────────

def _load_real_signals_for_class(candidates, ptbxl_dir: Path, n: int, rng: np.random.Generator) -> np.ndarray:
    order = rng.permutation(len(candidates))
    sigs = []
    for idx in order:
        if len(sigs) >= n:
            break
        rec = candidates.iloc[int(idx)]
        try:
            sig = wfdb.rdrecord(str(ptbxl_dir / str(rec["filename_lr"]))).p_signal
        except Exception:
            continue
        if sig.shape == (1000, 12) and np.isfinite(sig).all():
            sigs.append(sig)
    return np.array(sigs)


def build_energy_table(ptbxl_dir: Path, lead_names: list[str], n_per_class: int, seed: int, log) -> pd.DataFrame:
    db = load_ptbxl_database(ptbxl_dir)
    filtered = filter_to_mentor_classes(db)
    rng = np.random.default_rng(seed)

    rows = []
    for cls in MENTOR_CLASSES:
        candidates = filtered[filtered["mentor_class"] == cls]
        if candidates.empty:
            log.warning(f"No records for class {cls} — skipping in energy table.")
            continue
        signals = _load_real_signals_for_class(candidates, ptbxl_dir, n_per_class, rng)
        if len(signals) == 0:
            log.warning(f"No readable records for class {cls} — skipping.")
            continue
        log.info(f"{cls}: computing subband energy over {len(signals)} real records …")

        # per_lead_energy[band][record_idx] = (12,) array
        per_band_energy = {b: np.zeros((len(signals), len(lead_names))) for b in SUBBAND_NAMES}
        for i, sig in enumerate(signals):
            for band in SUBBAND_NAMES:
                per_band_energy[band][i] = subband_energy_per_lead(sig, band)

        for band in SUBBAND_NAMES:
            arr = per_band_energy[band]  # (n_records, 12)
            for lead_idx, lead_name in enumerate(lead_names):
                rows.append({
                    "class": cls, "subband": band, "lead": lead_name,
                    "mean_energy": float(arr[:, lead_idx].mean()),
                    "var_energy": float(arr[:, lead_idx].var()),
                    "n_records": len(signals),
                })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Item 2 — real vs. generated box plots (needs checkpoint)
# ──────────────────────────────────────────────────────────────────────────────

def _record_level_band_energy(signals: np.ndarray, band: str) -> np.ndarray:
    """(N, 1000, 12) -> (N,) one energy scalar per record (averaged across
    leads) for box-plot purposes — per-lead detail is already in the Item 1
    table; this collapses the lead axis so real-vs-generated-by-class stays
    legible in a single panel."""
    return np.array([subband_energy_per_lead(sig, band).mean() for sig in signals])


def make_boxplots(ckpt_path: Path, ptbxl_dir: Path, out_dir: Path, cfg, n_per_class: int, seed: int, log) -> bool:
    loaded = load_checkpoint(ckpt_path, cfg)
    if loaded is None:
        print(
            f"\n[BLOCKED] No checkpoint found at {ckpt_path}.\n"
            f"  Item 2 box plots compare REAL vs. GENERATED subband energy — there is\n"
            f"  nothing to compare without the trained model. Train on the GPU server,\n"
            f"  then re-run this script there.\n"
            f"  No box plots were written — nothing fabricated.\n"
        )
        return False

    db = load_ptbxl_database(ptbxl_dir)
    filtered = filter_to_mentor_classes(db)
    rng = np.random.default_rng(seed)

    stats_path = Path(cfg.paths.outputs.processed) / "preprocessing_stats.json"
    prep_stats = None
    if stats_path.exists():
        import json
        prep_stats = json.load(open(stats_path))

    real_by_class: dict[str, np.ndarray] = {}
    gen_by_class: dict[str, np.ndarray] = {}
    for cls in BOX_CLASSES:
        candidates = filtered[filtered["mentor_class"] == cls]
        real_signals = _load_real_signals_for_class(candidates, ptbxl_dir, n_per_class, rng)
        if len(real_signals):
            real_by_class[cls] = real_signals

        trained_cls = MENTOR_TO_TRAINED_CLASS.get(cls)
        if trained_cls is None:
            continue
        gen_signals, err = generate_for_class(loaded, trained_cls, n_samples=n_per_class, cfg=cfg, seed=seed, stats=prep_stats)
        if err:
            log.warning(err)
            continue
        gen_by_class[cls] = gen_signals

    out_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(PUBSTYLE):
        for band in SUBBAND_NAMES:
            fig, ax = plt.subplots(figsize=(8, 5))
            positions, labels, data = [], [], []
            pos = 0
            for cls in BOX_CLASSES:
                if cls in real_by_class:
                    data.append(_record_level_band_energy(real_by_class[cls], band))
                    labels.append(f"{cls}\n(real)")
                    positions.append(pos); pos += 1
                if cls in gen_by_class:
                    data.append(_record_level_band_energy(gen_by_class[cls], band))
                    labels.append(f"{cls}\n(generated)")
                    positions.append(pos); pos += 1
                pos += 0.6  # gap between class groups

            bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
            for patch, lbl in zip(bp["boxes"], labels):
                patch.set_facecolor("#1B3A6B" if "real" in lbl else "#C0392B")
                patch.set_alpha(0.6)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel("Subband energy (lead-averaged)")
            lo, hi = subband_frequency_ranges(100.0)[band]
            ax.set_title(f"{band} ({lo:.2f}-{hi:.2f} Hz, {SUBBAND_CLINICAL_LABEL[band]})")
            fig.tight_layout()
            out_path = out_dir / f"boxplot_{band}.png"
            fig.savefig(str(out_path))
            plt.close(fig)
            log.info(f"Saved {out_path}")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Subband energy table + real-vs-generated box plots.")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--n-per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("subband_decomposition", cfg=cfg)
    set_seed(args.seed)

    ptbxl_dir = Path(cfg.paths.data.ptbxl)
    ckpt_path = Path(args.ckpt) if args.ckpt else Path(cfg.paths.outputs.models) / "diffusion_best.pt"
    out_dir = Path(args.out_dir) if args.out_dir else subband_output_dir(cfg)
    lead_names = list(cfg.ptbxl.lead_names)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Item 1: building subband energy table (real data) …")
    table = build_energy_table(ptbxl_dir, lead_names, args.n_per_class, args.seed, log)
    table.to_csv(out_dir / "subband_energy_table.csv", index=False)
    print(f"✓ Subband energy table written ({len(table)} rows) → {out_dir / 'subband_energy_table.csv'}")

    log.info("Item 2: real-vs-generated box plots …")
    ok = make_boxplots(ckpt_path, ptbxl_dir, out_dir, cfg, args.n_per_class, args.seed, log)
    if ok:
        print(f"✓ Box plots written to {out_dir}")


if __name__ == "__main__":
    main()
