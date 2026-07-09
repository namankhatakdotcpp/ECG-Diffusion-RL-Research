"""
step03_eda_and_class_mapping.py — dynamic EDA, class mapping, and morphology statistics.

The final class list is determined HERE from the data (not hardcoded). Every
downstream step reads class_names.json produced by this script.

Stages:
  1. Load PTB-XL metadata → build SCP-code → superclass mapping from
     scp_statements.csv → apply min_class_samples threshold → save JSON artefacts
  2. Class distribution figure (paper Figure 1)
  3. 12-lead ECG examples figure (paper Figure 2)
  4. PQRST morphology statistics per class (→ reward function in step06)
  5. HRV statistics for NORM class (→ reward function in step06)
  6. Validation table
  7. A3-subband (slow-wave) reference distribution per class, for
     step06's A3Reward — same feature extraction as Stage 3's
     mentor_eval/subband_similarity_metrics.py (bior4.4, J=3), reused not
     reimplemented (see mentor_eval/subband_features.py)

Reads from:
  data/ptbxl/ptbxl_database.csv   (PTB-XL metadata with strat_fold + scp_codes)
  data/ptbxl/scp_statements.csv   (SCP code descriptions + diagnostic_class)
  outputs/processed/X_train.npy   (preprocessed signals, for Stages 3–5, 7)
  outputs/processed/record_ids_train.npy

Writes to:
  outputs/processed/class_mapping.json    {scp_code: class_name}
  outputs/processed/class_names.json      [ordered list of final class names]
  outputs/processed/class_counts.json     {class_name: {train,val,test}}
  outputs/processed/morphology_stats.json {class: {pr_ms,qrs_ms,qt_ms,hr_bpm}}
  outputs/processed/hrv_stats.json        {NORM: {sdnn_ms, rmssd_ms}}
  outputs/processed/a3_subband_stats.json {class: {mean: [12], cov: [12x12], n}}
  outputs/results/fig01_class_distribution.{pdf,png}
  outputs/results/fig02_ecg_examples.pdf

Usage:
    python step03_eda_and_class_mapping.py
"""

from __future__ import annotations

import ast
import json
import sys
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, get_logger, set_seed, assign_primary_class

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Publication style
# ──────────────────────────────────────────────────────────────────────────────

PUBSTYLE = {
    "font.size":       12,
    "font.family":     "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid":       False,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
}

# Ordered preference for superclass names (standard ECG literature ordering)
_STANDARD_ORDER = ["NORM", "MI", "STTC", "CD", "HYP", "AFIB"]

# Explicit overrides not reliably in scp_statements.csv
_HARD_OVERRIDES: dict[str, str] = {
    "AFIB": "AFIB",
    "AFLT": "AFIB",
}

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _savefig(fig: plt.Figure, stem: str, results_dir: Path) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(results_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 helpers — label parsing and class mapping
# ──────────────────────────────────────────────────────────────────────────────

def _parse_scp(raw: str) -> dict[str, float]:
    """Parse the scp_codes column (Python dict repr or JSON)."""
    try:
        return ast.literal_eval(str(raw))
    except (ValueError, SyntaxError):
        try:
            return json.loads(str(raw).replace("'", '"'))
        except Exception:
            return {}


def _build_code_to_class(
    scp_csv: Path,
    log,
) -> dict[str, str]:
    """
    Build a code→superclass lookup from scp_statements.csv.

    PTB-XL's scp_statements.csv has a `diagnostic_class` column with values
    NORM / MI / STTC / CD / HYP (or NaN for rhythm/quality codes).
    We add AFIB as an explicit 6th class via _HARD_OVERRIDES.
    """
    mapping: dict[str, str] = {}

    if scp_csv.exists():
        scp_df = pd.read_csv(scp_csv, index_col=0)
        dc_col = next(
            (c for c in scp_df.columns if "diagnostic_class" in c.lower()), None
        )
        if dc_col:
            for code, row in scp_df.iterrows():
                dc = str(row[dc_col]).strip().upper()
                if dc and dc != "NAN":
                    mapping[str(code).upper()] = dc
            log.info(f"scp_statements.csv: {len(mapping)} code→class entries loaded")
        else:
            log.warning("scp_statements.csv: no 'diagnostic_class' column — using hard-coded map")
    else:
        log.warning("scp_statements.csv not found — using hard-coded overrides only")

    # Apply explicit overrides (AFIB, AFLT — often missing in PTB-XL DC column)
    mapping.update({k.upper(): v for k, v in _HARD_OVERRIDES.items()})
    return mapping


def _assign_primary(scp_dict: dict[str, float], code_map: dict[str, str]) -> str:
    """
    Return the superclass with the highest-confidence SCP code.

    Ties at the maximum confidence are broken by
    utils.label_assignment.TIE_BREAK_PRIORITY (MI > STTC > CD > HYP > NORM
    > OTHER, a clinical-severity ordering — not dict-iteration order, see
    Roadmap/Stage_0_Pipeline_Audit/Reports/Pipeline_Code_Audit.md Finding 5).
    Delegates to the same shared function step04_transformer_diffusion.py's
    _load_class_labels() uses, so the two selection rules cannot silently
    diverge (Finding 4 confirmed they agreed before this fix; this keeps
    that guarantee structural rather than incidental).
    """
    if not scp_dict:
        return "OTHER"
    result = assign_primary_class(scp_dict, code_map)
    return result if result is not None else "OTHER"


def _determine_classes(
    train_labels: pd.Series,
    val_labels:   pd.Series,
    test_labels:  pd.Series,
    min_samples:  int,
    log,
) -> tuple[list[str], dict[str, dict[str, int]]]:
    """
    Return final class list and per-split counts.

    Algorithm:
      1. Any superclass with >= min_samples in training → kept as-is.
      2. Rare diagnostic codes with < min_samples → merged into OTHER.
      3. OTHER is kept only if it too has >= min_samples.
      4. Final list follows _STANDARD_ORDER, then alphabetical extras.
    """
    train_counts = train_labels.value_counts().to_dict()
    val_counts   = val_labels.value_counts().to_dict()
    test_counts  = test_labels.value_counts().to_dict()

    keep = {cls for cls, n in train_counts.items() if n >= min_samples}

    # Standard ordering + alphabetical extras + OTHER last
    final = [c for c in _STANDARD_ORDER if c in keep]
    extra = sorted(c for c in keep if c not in _STANDARD_ORDER and c != "OTHER")
    final.extend(extra)
    if "OTHER" in keep:
        final.append("OTHER")

    log.info(f"Classes meeting min_samples={min_samples}: {final}")
    if "OTHER" not in final:
        log.info("  'OTHER' below threshold — dropped (rare codes ignored)")

    counts: dict[str, dict[str, int]] = {}
    for cls in final:
        counts[cls] = {
            "train": int(train_counts.get(cls, 0)),
            "val":   int(val_counts.get(cls, 0)),
            "test":  int(test_counts.get(cls, 0)),
        }
    return final, counts


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — Class distribution figure
# ──────────────────────────────────────────────────────────────────────────────

def stage2_class_figure(
    final_classes: list[str],
    counts:        dict[str, dict[str, int]],
    results_dir:   Path,
    log,
) -> None:
    """Horizontal stacked bar chart: records per class × split (paper Figure 1)."""
    with plt.rc_context(PUBSTYLE):
        fig, ax = plt.subplots(figsize=(9, max(4, len(final_classes) * 0.65)))

        split_colors = {"train": "#4c78a8", "val": "#72b7b2", "test": "#f58518"}
        split_labels = {"train": "Train (folds 1–8)", "val": "Val (fold 9)",
                        "test": "Test (fold 10)"}

        y_pos    = np.arange(len(final_classes))
        left_acc = np.zeros(len(final_classes))

        for split in ("train", "val", "test"):
            vals = np.array([counts[c][split] for c in final_classes], dtype=float)
            bars = ax.barh(y_pos, vals, left=left_acc, color=split_colors[split],
                           label=split_labels[split], height=0.6,
                           edgecolor="white", linewidth=0.5)
            # Annotate each non-trivial segment
            for i, (bar, v) in enumerate(zip(bars, vals)):
                if v > 0:
                    ax.text(left_acc[i] + v / 2, i, f"{int(v):,}",
                            va="center", ha="center", fontsize=8,
                            color="white" if v > 300 else "#333")
            left_acc += vals

        ax.set_yticks(y_pos)
        ax.set_yticklabels(final_classes, fontsize=11, fontweight="bold")
        ax.set_xlabel("Number of records", fontsize=12)
        ax.set_title(
            "PTB-XL record distribution across dynamically-determined classes\n"
            "(class imbalance motivates diffusion-based synthetic augmentation)",
            fontsize=11, pad=10,
        )
        ax.legend(loc="lower right", fontsize=9)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x):,}")
        )
        ax.set_xlim(0, ax.get_xlim()[1] * 1.05)

    _savefig(fig, "fig01_class_distribution", results_dir)
    log.info("Saved fig01_class_distribution.{pdf,png}")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 — ECG examples figure (12-lead, NORM vs MI)
# ──────────────────────────────────────────────────────────────────────────────

def stage3_ecg_figure(
    X_train:       np.ndarray,
    train_primary: np.ndarray,
    final_classes: list[str],
    record_ids:    np.ndarray,
    fs:            float,
    results_dir:   Path,
    rng:           np.random.Generator,
    log,
) -> None:
    """Clinical-style 12-lead plot: NORM (left) and MI (right) after preprocessing."""
    target_cls = [c for c in ("NORM", "MI") if c in final_classes]
    if not target_cls:
        log.warning("Neither NORM nor MI found in final classes — skipping ECG figure.")
        return

    cls_to_color = {"NORM": "#2ca02c", "MI": "#d62728"}
    t = np.arange(X_train.shape[1]) / fs   # time axis in seconds

    with plt.rc_context(PUBSTYLE):
        fig, axes = plt.subplots(
            12, len(target_cls),
            figsize=(6 * len(target_cls), 20),
            sharey=False,
        )
        if len(target_cls) == 1:
            axes = axes[:, np.newaxis]   # keep 2-D shape

        for col, cls in enumerate(target_cls):
            idx_pool = np.where(train_primary == cls)[0]
            chosen   = rng.choice(idx_pool)
            ecg_id   = int(record_ids[chosen])
            color    = cls_to_color.get(cls, "#1f77b4")
            sig      = X_train[chosen]          # (1000, 12)

            axes[0, col].set_title(
                f"{cls}  (ecg_id={ecg_id})", color=color, fontsize=11, pad=6
            )
            for lead in range(12):
                ax = axes[lead, col]
                ax.plot(t, sig[:, lead], color=color, linewidth=0.7, alpha=0.9)
                ax.set_ylabel(LEAD_NAMES[lead], fontsize=9, rotation=0,
                              labelpad=30, va="center", color="#333")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.tick_params(labelsize=7)
                if lead < 11:
                    ax.set_xticks([])
                else:
                    ax.set_xlabel("Time (s)", fontsize=9)

        fig.suptitle(
            "Preprocessed 12-lead ECG examples\n"
            "(0.5–40 Hz bandpass, per-lead z-score, clipped to [−4, 4] σ)",
            fontsize=11, y=1.01,
        )
        plt.tight_layout()

    _savefig(fig, "fig02_ecg_examples", results_dir)
    log.info("Saved fig02_ecg_examples.{pdf,png}")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 4 — Morphology statistics
# ──────────────────────────────────────────────────────────────────────────────

def _morphology_one(
    signal_1d: np.ndarray,
    fs:        float,
) -> Optional[dict[str, list[float]]]:
    """
    Extract per-beat PQRST intervals from a single lead-II signal (1000 samples).

    Returns dict with lists (one value per beat) or None on failure.
    Physiological plausibility windows are applied to filter delineation artefacts.
    """
    try:
        import neurokit2 as nk
    except ImportError:
        raise ImportError("neurokit2 required — install with: pip install neurokit2")

    try:
        signals, info = nk.ecg_process(signal_1d.astype(np.float64), sampling_rate=fs)
    except Exception:
        return None

    r_peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
    if len(r_peaks) < 3:
        return None

    ms = 1000.0 / fs   # samples → ms conversion factor

    # RR intervals (ms) + heart rate (bpm)
    rr_ms = np.diff(r_peaks) * ms
    valid_rr = rr_ms[(rr_ms >= 300) & (rr_ms <= 2000)]
    if len(valid_rr) == 0:
        return None

    result: dict[str, list[float]] = {
        "rr_ms":  valid_rr.tolist(),
        "hr_bpm": (60_000.0 / valid_rr).tolist(),
    }

    def _locs(col: str) -> np.ndarray:
        if col not in signals.columns:
            return np.array([], dtype=int)
        return np.where(signals[col].fillna(0).astype(int) == 1)[0]

    p_locs = _locs("ECG_P_Peaks")
    q_locs = _locs("ECG_Q_Peaks")
    s_locs = _locs("ECG_S_Peaks")
    t_locs = _locs("ECG_T_Peaks")

    pr_list, qrs_list, qt_list = [], [], []
    q_win = int(0.06 * fs)   # 60 ms search window around R
    t_win = int(0.25 * fs)   # 250 ms after R for T-peak

    for r in r_peaks:
        # PR: nearest P-peak before R
        pre_p = p_locs[p_locs < r]
        if len(pre_p):
            pr = (r - pre_p[-1]) * ms
            if 60 < pr < 400:
                pr_list.append(pr)

        # QRS: Q before R, S after R (within ±60 ms)
        q_near = q_locs[(q_locs < r) & (q_locs > r - q_win)]
        s_near = s_locs[(s_locs > r) & (s_locs < r + q_win)]
        if len(q_near) and len(s_near):
            qrs = (s_near[0] - q_near[-1]) * ms
            if 40 < qrs < 200:
                qrs_list.append(qrs)

        # QT: Q before R, T after R (within 250 ms)
        t_near = t_locs[(t_locs > r) & (t_locs < r + t_win)]
        if len(q_near) and len(t_near):
            qt = (t_near[0] - q_near[-1]) * ms
            if 200 < qt < 650:
                qt_list.append(qt)

    if pr_list:  result["pr_ms"]  = pr_list
    if qrs_list: result["qrs_ms"] = qrs_list
    if qt_list:  result["qt_ms"]  = qt_list

    return result if len(result) > 2 else None


def _aggregate(values: list[float]) -> dict[str, float]:
    """IQR-filtered mean ± std from a list of per-beat measurements."""
    arr = np.array(values)
    q1, q3 = np.percentile(arr, [25, 75])
    arr = arr[(arr >= q1 - 3 * (q3 - q1)) & (arr <= q3 + 3 * (q3 - q1))]
    return {"mean": round(float(arr.mean()), 2), "std": round(float(arr.std()), 2),
            "n": int(len(arr))}


def stage4_morphology(
    X_train:       np.ndarray,
    train_primary: np.ndarray,
    final_classes: list[str],
    fs:            float,
    processed_dir: Path,
    rng:           np.random.Generator,
    log,
    n_per_class:   int = 200,
    lead_idx:      int = 1,    # Lead II
) -> dict:
    """Compute PQRST morphology stats for every class; save morphology_stats.json."""
    try:
        import neurokit2  # noqa: F401 — check import once, error clearly
    except ImportError:
        log.warning("neurokit2 not installed — skipping morphology analysis.\n"
                    "Run: pip install neurokit2")
        return {}

    log.info(f"Morphology analysis: n={n_per_class}/class, lead=II, fs={fs} Hz")
    all_stats: dict[str, dict] = {}

    for cls in final_classes:
        idx_pool = np.where(train_primary == cls)[0]
        if len(idx_pool) == 0:
            log.warning(f"  {cls}: no training records — skipped")
            continue

        sample_idx = rng.choice(idx_pool, size=min(n_per_class, len(idx_pool)), replace=False)
        log.info(f"  {cls}: processing {len(sample_idx)} records …")

        acc: dict[str, list[float]] = {k: [] for k in ("pr_ms", "qrs_ms", "qt_ms",
                                                         "rr_ms", "hr_bpm")}
        n_ok = n_fail = 0
        for i in sample_idx:
            result = _morphology_one(X_train[i, :, lead_idx], fs)
            if result is None:
                n_fail += 1
                continue
            for key in acc:
                acc[key].extend(result.get(key, []))
            n_ok += 1

        log.info(f"    {cls}: {n_ok} OK, {n_fail} failed")

        cls_stats: dict[str, dict] = {}
        for key, vals in acc.items():
            if len(vals) >= 3:
                cls_stats[key] = _aggregate(vals)
                log.info(f"    {cls}.{key}: mean={cls_stats[key]['mean']:.1f} "
                         f"± {cls_stats[key]['std']:.1f}  (n={cls_stats[key]['n']})")

        all_stats[cls] = cls_stats

    out_path = processed_dir / "morphology_stats.json"
    with open(out_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    log.info(f"Saved morphology_stats.json → {out_path}")
    return all_stats


# ──────────────────────────────────────────────────────────────────────────────
# Stage 5 — HRV statistics (NORM only)
# ──────────────────────────────────────────────────────────────────────────────

def stage5_hrv(
    X_train:       np.ndarray,
    train_primary: np.ndarray,
    fs:            float,
    processed_dir: Path,
    rng:           np.random.Generator,
    log,
    n_records:     int = 200,
    lead_idx:      int = 1,
) -> dict:
    """Compute SDNN and RMSSD for NORM class; save hrv_stats.json."""
    try:
        import neurokit2 as nk  # noqa: F401
    except ImportError:
        log.warning("neurokit2 not installed — skipping HRV analysis.")
        return {}

    norm_idx = np.where(train_primary == "NORM")[0]
    if len(norm_idx) == 0:
        log.warning("No NORM records in training set — skipping HRV.")
        return {}

    sample_idx = rng.choice(norm_idx, size=min(n_records, len(norm_idx)), replace=False)
    log.info(f"HRV analysis: {len(sample_idx)} NORM records, lead=II")

    sdnn_list, rmssd_list = [], []
    n_ok = n_fail = 0

    for i in sample_idx:
        try:
            import neurokit2 as nk
            _, info = nk.ecg_process(
                X_train[i, :, lead_idx].astype(np.float64), sampling_rate=fs
            )
            r_peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
            if len(r_peaks) < 4:
                n_fail += 1
                continue
            rr = np.diff(r_peaks) * (1000.0 / fs)
            rr = rr[(rr >= 300) & (rr <= 2000)]
            if len(rr) < 3:
                n_fail += 1
                continue
            sdnn_list.append(float(np.std(rr, ddof=1)))
            rmssd_list.append(float(np.sqrt(np.mean(np.diff(rr) ** 2))))
            n_ok += 1
        except Exception:
            n_fail += 1

    log.info(f"  HRV: {n_ok} OK, {n_fail} failed")

    hrv_stats: dict = {}
    if sdnn_list:
        hrv_stats["NORM"] = {
            "sdnn_ms":  _aggregate(sdnn_list),
            "rmssd_ms": _aggregate(rmssd_list),
        }
        log.info(f"  SDNN:  {hrv_stats['NORM']['sdnn_ms']['mean']:.1f} "
                 f"± {hrv_stats['NORM']['sdnn_ms']['std']:.1f} ms")
        log.info(f"  RMSSD: {hrv_stats['NORM']['rmssd_ms']['mean']:.1f} "
                 f"± {hrv_stats['NORM']['rmssd_ms']['std']:.1f} ms")

    out_path = processed_dir / "hrv_stats.json"
    with open(out_path, "w") as f:
        json.dump(hrv_stats, f, indent=2)
    log.info(f"Saved hrv_stats.json → {out_path}")
    return hrv_stats


# ──────────────────────────────────────────────────────────────────────────────
# Stage 7 — A3-subband (slow-wave) reference distribution per class
# ──────────────────────────────────────────────────────────────────────────────

def stage7_a3_subband_stats(
    X_train:       np.ndarray,
    train_primary: np.ndarray,
    final_classes: list,
    processed_dir: Path,
    rng:           np.random.Generator,
    log,
    n_per_class:   int = 200,
) -> dict:
    """
    Per-class mean + covariance of the A3-subband (0-6.25 Hz at fs=100 Hz:
    P-wave, T-wave, ST-segment, baseline) 12-lead energy feature, for
    step06_reward_function.py's A3Reward.

    Directly motivated by Stage 3's dominant finding, replicated across every
    architecture evaluated there: A3-subband divergence is the largest
    remaining gap between generated and real ECGs
    (see Stage3_Subband_Master_Comparison.md).

    Feature extraction is `mentor_eval.subband_features.
    extract_subband_energy_batch` — the EXACT function Stage 3's evaluation
    script (mentor_eval/subband_similarity_metrics.py) already uses (bior4.4
    wavelet, J=3, mean-squared-coefficient energy per lead). Imported, not
    reimplemented, so Stage 3 evaluation and this Stage 4 reward reference
    distribution cannot silently drift apart on what "A3 energy" means.

    MIN_SAMPLES_FOR_COVARIANCE_MULTIPLIER (5x the 12-dim feature = 60
    samples/class minimum) is the same stability threshold
    mentor_eval/similarity_metrics.py already enforces for its own
    Mahalanobis/Bhattacharyya calculations — classes below it are flagged,
    not silently computed with an ill-conditioned covariance.
    """
    from mentor_eval.subband_features import extract_subband_energy_batch, SUBBAND_NAMES, WAVELET, LEVELS
    from mentor_eval.similarity_metrics import MIN_SAMPLES_FOR_COVARIANCE_MULTIPLIER

    a3_band_idx = SUBBAND_NAMES.index("A3")
    n_leads     = X_train.shape[2]
    min_needed  = MIN_SAMPLES_FOR_COVARIANCE_MULTIPLIER * n_leads

    log.info(f"A3-subband stats: n={n_per_class}/class (min {min_needed} for stable covariance)")
    all_stats: dict[str, dict] = {}

    for cls in final_classes:
        idx_pool = np.where(train_primary == cls)[0]
        if len(idx_pool) < min_needed:
            log.warning(
                f"  {cls}: only {len(idx_pool)} training records "
                f"(< {min_needed} needed) — skipped, no A3 reference for this class."
            )
            continue

        sample_idx = rng.choice(idx_pool, size=min(n_per_class, len(idx_pool)), replace=False)
        feats_full = extract_subband_energy_batch(X_train[sample_idx])       # (n, 4*12)
        feats_a3   = feats_full[:, a3_band_idx * n_leads:(a3_band_idx + 1) * n_leads]  # (n, 12)

        mean = feats_a3.mean(axis=0)
        cov  = np.cov(feats_a3, rowvar=False)

        all_stats[cls] = {
            "mean": mean.tolist(),
            "cov":  cov.tolist(),
            "n":    int(len(sample_idx)),
        }
        log.info(f"  {cls}: n={len(sample_idx)}  mean_energy={mean.mean():.4f}")

    # Metadata so A3Reward can sanity-check the file it's loading rather than
    # trusting it blindly — catches silent drift if this file is ever
    # regenerated months from now with different wavelet/level/normalisation
    # choices without anyone updating the reward code to match.
    import datetime
    metadata = {
        "dataset":         "PTB-XL (real only — this is a training reference, must never include generated samples)",
        "split":           "train",
        "n_per_class_requested": n_per_class,
        "min_samples_for_covariance": min_needed,
        "wavelet":         WAVELET,
        "decomposition_level": LEVELS,
        "subband":         "A3",
        "feature_normalization": "none (raw z-score-space signal in, mean-squared-coefficient energy out — same space as X_train.npy/X_test.npy)",
        "created_utc":     datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    out_path = processed_dir / "a3_subband_stats.json"
    with open(out_path, "w") as f:
        json.dump({"_metadata": metadata, "classes": all_stats}, f, indent=2)
    log.info(f"Saved a3_subband_stats.json → {out_path}")
    return all_stats


# ──────────────────────────────────────────────────────────────────────────────
# Stage 6 — Validation table
# ──────────────────────────────────────────────────────────────────────────────

def stage6_validation(
    X_train:         np.ndarray,
    final_classes:   list[str],
    counts:          dict[str, dict[str, int]],
    morphology_stats: dict,
    log,
) -> bool:
    """Print markdown validation table; return True if all checks pass."""
    checks: list[tuple[str, str, str, bool]] = []

    def _chk(metric: str, value: str, expected: str, ok: bool) -> None:
        checks.append((metric, value, expected, ok))

    # NaN / Inf in training signals
    nan_ok  = not np.isnan(X_train).any()
    inf_ok  = not np.isinf(X_train).any()
    _chk("No NaN in X_train",          str(nan_ok), "True",      nan_ok)
    _chk("No Inf in X_train",          str(inf_ok), "True",      inf_ok)

    # Signal range
    xmin, xmax = float(X_train.min()), float(X_train.max())
    _chk("X_train min ≥ −4",  f"{xmin:.3f}", "≥ −4.0",  xmin >= -4.0)
    _chk("X_train max ≤  4",  f"{xmax:.3f}", "≤  4.0",  xmax <=  4.0)
    xmean, xstd = float(X_train.mean()), float(X_train.std())
    _chk("X_train mean ≈ 0",  f"{xmean:.4f}", "≈ 0.0",   abs(xmean) < 0.5)
    _chk("X_train std  ≈ 1",  f"{xstd:.4f}",  "≈ 1.0",   0.5 < xstd < 2.0)

    # No class has 0 records in val or test
    for cls in final_classes:
        for split in ("val", "test"):
            n = counts[cls].get(split, 0)
            _chk(f"{cls} records in {split} > 0", str(n), "> 0", n > 0)

    # Morphology plausibility (if available)
    for cls in ("NORM", "MI"):
        ms = morphology_stats.get(cls, {})
        if "pr_ms" in ms:
            pr = ms["pr_ms"]["mean"]
            _chk(f"{cls} PR interval within 80–250 ms",
                 f"{pr:.1f} ms", "80–250 ms", 80 <= pr <= 250)
        if "qrs_ms" in ms:
            qrs = ms["qrs_ms"]["mean"]
            _chk(f"{cls} QRS duration within 40–160 ms",
                 f"{qrs:.1f} ms", "40–160 ms", 40 <= qrs <= 160)

    # Print
    cw = [50, 20, 20, 10]
    hdr = f"| {'Metric':<{cw[0]}} | {'Value':>{cw[1]}} | {'Expected':>{cw[2]}} | {'Result':>{cw[3]}} |"
    sep = "|-" + "-|-".join("-" * w for w in cw) + "-|"
    print()
    print(hdr); print(sep)
    for metric, value, expected, ok in checks:
        result = "✓ PASS" if ok else "✗ FAIL"
        print(f"| {metric:<{cw[0]}} | {value:>{cw[1]}} | {expected:>{cw[2]}} | {result:>{cw[3]}} |")
    print()

    all_pass = all(ok for _, _, _, ok in checks)
    if not all_pass:
        log.warning("Some validation checks FAILED — review the table above.")
    else:
        log.info("All validation checks PASSED.")
    return all_pass


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    log = get_logger("step03_eda_and_class_mapping", cfg=cfg)
    set_seed(cfg.seeds[0])
    rng = np.random.default_rng(int(cfg.seeds[0]))

    ptbxl_dir     = Path(cfg.paths.data.ptbxl)
    processed_dir = Path(cfg.paths.outputs.processed)
    results_dir   = Path(cfg.paths.outputs.results)
    for d in (processed_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    fs          = float(cfg.ptbxl.sampling_rate)
    train_folds = list(cfg.ptbxl.train_fold)
    val_folds   = list(cfg.ptbxl.val_fold)
    test_folds  = list(cfg.ptbxl.test_fold)
    min_samples = int(cfg.ptbxl.min_class_samples)

    # ── Load PTB-XL database CSV ─────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STAGE 1 — Build class mapping from PTB-XL metadata")
    log.info("=" * 60)

    db_csv = ptbxl_dir / "ptbxl_database.csv"
    if not db_csv.exists():
        log.error(f"ptbxl_database.csv not found at {db_csv}. Run step01 first.")
        sys.exit(1)

    db_df = pd.read_csv(db_csv, index_col="ecg_id")
    log.info(f"Loaded {len(db_df):,} records from ptbxl_database.csv")

    # Build SCP-code → superclass lookup
    code_map = _build_code_to_class(ptbxl_dir / "scp_statements.csv", log)

    # Assign primary class per record
    log.info("Assigning primary diagnostic class per record …")
    db_df["primary_class"] = db_df["scp_codes"].apply(
        lambda raw: _assign_primary(_parse_scp(raw), code_map)
    )

    # Split by official folds
    train_mask = db_df["strat_fold"].isin(train_folds)
    val_mask   = db_df["strat_fold"].isin(val_folds)
    test_mask  = db_df["strat_fold"].isin(test_folds)

    train_df = db_df[train_mask]
    val_df   = db_df[val_mask]
    test_df  = db_df[test_mask]
    log.info(f"Splits: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    # Determine final class list
    final_classes, counts = _determine_classes(
        train_df["primary_class"],
        val_df["primary_class"],
        test_df["primary_class"],
        min_samples,
        log,
    )

    # Build full SCP-code → final class mapping (codes not in final_classes → OTHER)
    def _remap(cls: str) -> Optional[str]:
        if cls in final_classes:
            return cls
        return "OTHER" if "OTHER" in final_classes else None

    full_mapping: dict[str, str] = {}
    for code, superclass in code_map.items():
        remapped = _remap(superclass)
        if remapped:
            full_mapping[code] = remapped

    # Save artefacts
    with open(processed_dir / "class_mapping.json", "w") as f:
        json.dump(full_mapping, f, indent=2, sort_keys=True)
    with open(processed_dir / "class_names.json", "w") as f:
        json.dump(final_classes, f, indent=2)
    with open(processed_dir / "class_counts.json", "w") as f:
        json.dump(counts, f, indent=2)

    log.info(f"Saved class_mapping.json  ({len(full_mapping)} SCP codes mapped)")
    log.info(f"Saved class_names.json    {final_classes}")
    log.info(f"Saved class_counts.json")

    for cls in final_classes:
        c = counts[cls]
        log.info(f"  {cls:>8s}: train={c['train']:>5,}  val={c['val']:>4,}  test={c['test']:>4,}")

    # ── Load signal arrays (needed for Stages 3–5) ───────────────────────────
    X_train:       Optional[np.ndarray] = None
    train_primary: Optional[np.ndarray] = None
    record_ids:    Optional[np.ndarray] = None

    x_train_path = processed_dir / "X_train.npy"
    ids_path     = processed_dir / "record_ids_train.npy"

    if x_train_path.exists() and ids_path.exists():
        log.info("Loading X_train.npy + record_ids_train.npy …")
        X_train    = np.load(str(x_train_path))        # (N, 1000, 12)
        record_ids = np.load(str(ids_path))            # (N,) ecg_id values

        # Map record_id → primary_class string
        id_to_class = db_df["primary_class"].to_dict()   # ecg_id (int) → class str
        train_primary = np.array(
            [_remap(id_to_class.get(int(eid), "OTHER")) or "OTHER"
             for eid in record_ids],
            dtype=object,
        )
        log.info(f"X_train shape: {X_train.shape}  train_primary: {train_primary.shape}")
    else:
        log.warning(
            "X_train.npy or record_ids_train.npy not found — "
            "skipping Stages 3, 4, 5 (run step02_preprocessing.py first)."
        )

    # ── Stage 2 — Class distribution figure ──────────────────────────────────
    log.info("=" * 60)
    log.info("STAGE 2 — Class distribution figure")
    log.info("=" * 60)
    stage2_class_figure(final_classes, counts, results_dir, log)

    # ── Stage 3 — ECG examples figure ────────────────────────────────────────
    log.info("=" * 60)
    log.info("STAGE 3 — ECG examples figure")
    log.info("=" * 60)
    if X_train is not None:
        stage3_ecg_figure(
            X_train, train_primary, final_classes, record_ids,
            fs, results_dir, rng, log,
        )
    else:
        log.info("Skipped (no X_train.npy)")

    # ── Stage 4 — Morphology stats ───────────────────────────────────────────
    log.info("=" * 60)
    log.info("STAGE 4 — Morphology statistics")
    log.info("=" * 60)
    morphology_stats: dict = {}
    if X_train is not None:
        morphology_stats = stage4_morphology(
            X_train, train_primary, final_classes, fs, processed_dir, rng, log,
            n_per_class=200,
        )
    else:
        log.info("Skipped (no X_train.npy)")

    # ── Stage 5 — HRV stats ───────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STAGE 5 — HRV statistics (NORM)")
    log.info("=" * 60)
    if X_train is not None:
        stage5_hrv(X_train, train_primary, fs, processed_dir, rng, log, n_records=200)
    else:
        log.info("Skipped (no X_train.npy)")

    # ── Stage 6 — Validation table ────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STAGE 6 — Validation table")
    log.info("=" * 60)
    if X_train is not None:
        stage6_validation(X_train, final_classes, counts, morphology_stats, log)
    else:
        # Run reduced validation (counts only, no signal checks)
        log.info("Signal checks skipped (no X_train). Class count checks:")
        all_ok = True
        for cls in final_classes:
            for split in ("val", "test"):
                n = counts[cls].get(split, 0)
                ok = n > 0
                all_ok = all_ok and ok
                status = "✓" if ok else "✗"
                log.info(f"  {status} {cls} in {split}: {n}")

    # ── Stage 7 — A3-subband reference distribution ─────────────────────────
    log.info("=" * 60)
    log.info("STAGE 7 — A3-subband reference distribution")
    log.info("=" * 60)
    if X_train is not None:
        stage7_a3_subband_stats(
            X_train, train_primary, final_classes, processed_dir, rng, log,
            n_per_class=200,
        )
    else:
        log.info("Skipped (no X_train.npy)")

    log.info("=" * 60)
    print(f"✓ EDA complete. Final classes: {final_classes}. Morphology stats saved.")


if __name__ == "__main__":
    main()
