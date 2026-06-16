"""
step02_preprocessing.py — publication-quality ECG preprocessing pipeline.

Pipeline stages:
  1. Load PTB-XL metadata and derive 7-class labels from SCP codes
  2. Load raw wfdb signals, apply bandpass + baseline-wander correction
  3. Split using official PTB-XL stratified folds (no random split)
  4. Per-lead z-score normalisation fit on training set only (no data leakage)
  5. Save compressed NumPy arrays + preprocessing_stats.json
  6. Spot-check: reload, assert, plot 12-lead comparison (MI vs NORM)

Outputs (all in outputs/processed/):
  X_{train,val,test}.npy               — float32, shape (N, 1000, 12)
  y_{train,val,test}_multilabel.npy    — int8,    shape (N, 7)
  y_{train,val,test}_single.npy        — int16,   shape (N,)
  record_ids_{train,val,test}.npy      — int32,   shape (N,)
  preprocessing_stats.json
  preprocessing_spot_check.png

Usage:
    python step02_preprocessing.py
"""

from __future__ import annotations

import ast
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, medfilt
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, get_logger, set_seed

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# PTB-XL SCP code → our 7-class diagnostic superclass.
# Derived from scp_statements.csv (diagnostic_class column) plus explicit AFIB
# handling.  Any SCP code not listed here is mapped to OTHER.
_SCP_TO_SUPERCLASS: dict[str, str] = {
    # --- NORM ---
    "NORM": "NORM",
    # --- MI ---
    "AMI": "MI", "IMI": "MI", "LMI": "MI", "PMI": "MI",
    "ILMI": "MI", "ALMI": "MI", "INJAS": "MI", "INJAL": "MI",
    "INJIL": "MI", "INJIN": "MI", "INJLA": "MI",
    "IPLMI": "MI", "IPMI": "MI",
    "ISCAL": "MI", "ISCAN": "MI", "ISCAS": "MI",
    "ISCIL": "MI", "ISCIN": "MI", "ISCLA": "MI", "ISC_": "MI",
    # --- STTC (ST/T-Change) ---
    "STD_": "STTC", "STDD": "STTC", "STTC": "STTC",
    "VCLVH": "STTC", "NDT": "STTC", "NST_": "STTC",
    "DIG": "STTC", "LNGQT": "STTC",
    # --- CD (Conduction Disturbance) ---
    "LAFB": "CD", "LPFB": "CD", "IRBBB": "CD", "IVCD": "CD",
    "LBBB": "CD", "RBBB": "CD", "ILBBB": "CD",
    "CLBBB": "CD", "CRBBB": "CD", "PACE": "CD",
    "AVB": "CD", "1AVB": "CD", "2AVB": "CD", "3AVB": "CD",
    "WPW": "CD", "PSVT": "CD", "SVTAC": "CD",
    # --- HYP (Hypertrophy) ---
    "LVH": "HYP", "RVH": "HYP", "SEHYP": "HYP",
    "LAE": "HYP", "RAE": "HYP", "HVOLT": "HYP", "LVOLT": "HYP",
    # --- AFIB ---
    "AFIB": "AFIB", "AFLT": "AFIB",
}

# ──────────────────────────────────────────────────────────────────────────────
# Signal processing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _butter_bandpass(
    signal: np.ndarray,
    low: float,
    high: float,
    fs: float,
    order: int,
) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter along axis 0 (time)."""
    nyq = fs / 2.0
    low_n, high_n = low / nyq, high / nyq
    # Clamp to avoid scipy numerical issues at Nyquist
    high_n = min(high_n, 0.999)
    b, a = butter(order, [low_n, high_n], btype="band")
    # filtfilt over each lead independently
    return filtfilt(b, a, signal, axis=0).astype(np.float32)


def _baseline_correct(signal: np.ndarray, window_samples: int) -> np.ndarray:
    """Remove baseline wander by subtracting a per-lead median filter.

    Operates on shape (T, 12).  Window must be odd for medfilt.
    """
    # medfilt requires odd kernel size
    ks = window_samples if window_samples % 2 == 1 else window_samples + 1
    out = np.empty_like(signal, dtype=np.float32)
    for lead in range(signal.shape[1]):
        baseline = medfilt(signal[:, lead].astype(np.float64), kernel_size=ks)
        out[:, lead] = (signal[:, lead] - baseline).astype(np.float32)
    return out


def _process_record(
    record_path: str,
    signal_length: int,
    low: float,
    high: float,
    fs: float,
    order: int,
    baseline_samples: int,
) -> Optional[np.ndarray]:
    """Load and filter a single PTB-XL record.  Returns shape (T, 12) or None."""
    try:
        rec = wfdb.rdrecord(record_path, sampfrom=0, sampto=signal_length)
        sig = rec.p_signal.astype(np.float32)   # (T, 12)

        # Replace NaN with interpolated values before filtering
        if np.isnan(sig).any():
            for c in range(sig.shape[1]):
                nans = np.isnan(sig[:, c])
                if nans.all():
                    sig[:, c] = 0.0
                elif nans.any():
                    idx = np.arange(len(sig[:, c]))
                    sig[:, c] = np.interp(idx, idx[~nans], sig[~nans, c])

        sig = _baseline_correct(sig, baseline_samples)
        sig = _butter_bandpass(sig, low, high, fs, order)
        return sig
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Label helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_scp_codes(raw: str) -> dict[str, float]:
    """Parse the scp_codes string column (Python dict repr or JSON)."""
    try:
        return ast.literal_eval(str(raw))
    except (ValueError, SyntaxError):
        try:
            return json.loads(str(raw).replace("'", '"'))
        except json.JSONDecodeError:
            return {}


def _assign_labels(
    scp_dict: dict[str, float],
    confidence_threshold: float,
    classes: list[str],
) -> list[str]:
    """Map SCP codes → list of our 7-class labels that exceed the threshold."""
    assigned: set[str] = set()
    for code, confidence in scp_dict.items():
        if confidence < confidence_threshold:
            continue
        superclass = _SCP_TO_SUPERCLASS.get(code.upper())
        if superclass and superclass in classes:
            assigned.add(superclass)
    # Fallback: if nothing mapped, label as OTHER
    if not assigned and scp_dict:
        assigned.add("OTHER")
    return sorted(assigned)


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _compute_train_stats(
    X_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-lead mean and std over the training set.

    X_train shape: (N, T, 12)
    Returns arrays of shape (12,).
    """
    # Reshape to (N*T, 12) for vectorised stats
    flat = X_train.reshape(-1, X_train.shape[2])
    mu = flat.mean(axis=0)
    sigma = flat.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)   # prevent div-by-zero
    return mu.astype(np.float32), sigma.astype(np.float32)


def _normalise(
    X: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    clip: tuple[float, float],
) -> np.ndarray:
    """Z-score normalise and clip. Operates in-place and returns float32."""
    X = (X - mu) / sigma          # broadcast over (N, T, 12)
    X = np.clip(X, clip[0], clip[1])
    return X.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Spot-check plot
# ──────────────────────────────────────────────────────────────────────────────

def _spot_check_plot(
    X: np.ndarray,
    y_single: np.ndarray,
    class_names: list[str],
    record_ids: np.ndarray,
    fs: float,
    out_path: str,
    log,
) -> None:
    """Plot 12-lead ECGs for one MI and one NORM record side by side."""
    mi_idx = np.where(y_single == class_names.index("MI"))[0]
    norm_idx = np.where(y_single == class_names.index("NORM"))[0]

    if len(mi_idx) == 0 or len(norm_idx) == 0:
        log.warning("Cannot create spot-check plot — MI or NORM absent in training set.")
        return

    fig, axes = plt.subplots(12, 2, figsize=(16, 24), sharey=False)
    fig.suptitle("Preprocessing Spot-Check: MI (left) vs NORM (right)", fontsize=14)

    lead_names = [
        "I", "II", "III", "aVR", "aVL", "aVF",
        "V1", "V2", "V3", "V4", "V5", "V6",
    ]
    t = np.arange(X.shape[1]) / fs

    for col, (idx, label, color) in enumerate(
        [(mi_idx[0], "MI", "#d62728"), (norm_idx[0], "NORM", "#2ca02c")]
    ):
        rec_id = record_ids[idx]
        for lead in range(12):
            ax = axes[lead, col]
            ax.plot(t, X[idx, :, lead], color=color, linewidth=0.6)
            ax.set_ylabel(lead_names[lead], fontsize=7)
            ax.tick_params(labelsize=6)
            if lead == 0:
                ax.set_title(f"{label}  (ecg_id={rec_id})", color=color)
            if lead == 11:
                ax.set_xlabel("Time (s)", fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    log.info(f"Spot-check plot saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    log = get_logger("step02_preprocessing", cfg=cfg)
    set_seed(cfg.seeds[0])

    ptbxl_root = Path(cfg.paths.data.ptbxl)
    out_dir = Path(cfg.paths.outputs.processed)
    out_dir.mkdir(parents=True, exist_ok=True)

    fs = float(cfg.ptbxl.sampling_rate)
    signal_length = int(cfg.ptbxl.signal_length)
    classes: list[str] = list(cfg.ptbxl.classes)
    train_folds: list[int] = list(cfg.ptbxl.train_fold)
    val_folds: list[int] = list(cfg.ptbxl.val_fold)
    test_folds: list[int] = list(cfg.ptbxl.test_fold)

    pp = cfg.preprocessing
    bp_low = float(pp.bandpass_low)
    bp_high = float(pp.bandpass_high)
    filt_order = int(pp.filter_order)
    baseline_samples = int(pp.baseline_window_sec * fs)
    clip_lo, clip_hi = float(pp.clip_range[0]), float(pp.clip_range[1])
    batch_size = int(pp.batch_size)
    ml_conf = float(pp.multilabel_confidence)

    log.info("=" * 60)
    log.info("STAGE 1 — Load metadata and build labels")
    log.info("=" * 60)

    db_path = ptbxl_root / "ptbxl_database.csv"
    if not db_path.exists():
        log.error(
            f"PTB-XL database CSV not found at {db_path}. "
            "Run step01_data_download.py first."
        )
        sys.exit(1)

    df = pd.read_csv(db_path, index_col="ecg_id")
    log.info(f"Loaded {len(df):,} records from ptbxl_database.csv")

    # Try to load label_mapping from step01; fall back to built-in _SCP_TO_SUPERCLASS
    label_map_path = ptbxl_root / "label_mapping.json"
    if label_map_path.exists():
        with open(label_map_path) as f:
            label_mapping = json.load(f)
        # Flatten class→[scp_codes] to scp_code→class lookup
        step01_lookup: dict[str, str] = {}
        for cls, codes in label_mapping.items():
            for code in codes:
                step01_lookup[code.upper()] = cls
        scp_lookup = {**_SCP_TO_SUPERCLASS, **step01_lookup}
        log.info(f"Merged label_mapping.json ({len(step01_lookup)} codes)")
    else:
        # Also try scp_statements.csv shipped with PTB-XL
        scp_csv = ptbxl_root / "scp_statements.csv"
        scp_lookup = dict(_SCP_TO_SUPERCLASS)
        if scp_csv.exists():
            scp_df = pd.read_csv(scp_csv, index_col=0)
            if "diagnostic_class" in scp_df.columns:
                for code, row in scp_df.iterrows():
                    dc = str(row["diagnostic_class"]).strip().upper()
                    if dc in classes and str(code).upper() not in scp_lookup:
                        scp_lookup[str(code).upper()] = dc
                log.info(f"Augmented lookup with scp_statements.csv ({len(scp_df)} entries)")
        else:
            log.warning("label_mapping.json and scp_statements.csv not found — using built-in SCP map")

    # Parse labels
    log.info("Parsing scp_codes and assigning diagnostic superclasses …")
    label_lists: list[list[str]] = []
    for ecg_id, row in tqdm(df.iterrows(), total=len(df), desc="  parsing labels", ncols=80):
        scp_dict = _parse_scp_codes(row["scp_codes"])
        # Use the resolved lookup
        assigned: set[str] = set()
        for code, conf in scp_dict.items():
            if conf < ml_conf:
                continue
            superclass = scp_lookup.get(code.upper())
            if superclass and superclass in classes:
                assigned.add(superclass)
        if not assigned and scp_dict:
            assigned.add("OTHER")
        label_lists.append(sorted(assigned))

    df["labels"] = label_lists

    # Primary label = class with highest SCP confidence (for single-label column)
    primary_labels: list[str] = []
    for ecg_id, row in df.iterrows():
        scp_dict = _parse_scp_codes(row["scp_codes"])
        best_code, best_conf = "", -1.0
        for code, conf in scp_dict.items():
            if conf > best_conf and scp_lookup.get(code.upper()) in classes:
                best_code, best_conf = code, conf
        cls = scp_lookup.get(best_code.upper(), "OTHER") if best_code else "OTHER"
        primary_labels.append(cls)
    df["primary_label"] = primary_labels

    class_dist = df["primary_label"].value_counts().to_dict()
    log.info("Class distribution (primary label):")
    for cls in classes:
        log.info(f"  {cls:>8s}: {class_dist.get(cls, 0):>5,}")

    # ──────────────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STAGE 2+3 — Load signals + apply fold-based split")
    log.info("=" * 60)

    # Split index sets by official folds
    train_mask = df["strat_fold"].isin(train_folds)
    val_mask   = df["strat_fold"].isin(val_folds)
    test_mask  = df["strat_fold"].isin(test_folds)

    df_train = df[train_mask]
    df_val   = df[val_mask]
    df_test  = df[test_mask]

    log.info(
        f"Official folds → train: {len(df_train):,} | "
        f"val: {len(df_val):,} | test: {len(df_test):,}"
    )

    filter_params = {
        "bandpass_low_hz": bp_low,
        "bandpass_high_hz": bp_high,
        "filter_order": filt_order,
        "filter_type": "butterworth_zerophase",
        "baseline_window_sec": float(pp.baseline_window_sec),
        "baseline_method": "median_filter",
        "sampling_rate_hz": fs,
        "signal_length_samples": signal_length,
    }

    def _load_split(
        split_df: pd.DataFrame,
        split_name: str,
    ) -> tuple[np.ndarray, np.ndarray, list[list[str]], np.ndarray]:
        """Load and filter all records in a split; return (X, record_ids, label_lists, primary)."""
        n = len(split_df)
        X_buf = np.zeros((n, signal_length, 12), dtype=np.float32)
        ids_buf = np.zeros(n, dtype=np.int32)
        lbl_buf: list[list[str]] = []
        primary_buf: list[str] = []
        valid_mask = np.ones(n, dtype=bool)

        for batch_start in tqdm(
            range(0, n, batch_size),
            desc=f"  loading {split_name}",
            ncols=80,
        ):
            batch_idx = range(batch_start, min(batch_start + batch_size, n))
            for local_i, global_i in enumerate(batch_idx):
                row = split_df.iloc[global_i]
                rel_path = str(row.get("filename_lr", "")).strip()
                if not rel_path:
                    valid_mask[global_i] = False
                    lbl_buf.append([])
                    primary_buf.append("OTHER")
                    continue

                record_path = str(ptbxl_root / rel_path)
                sig = _process_record(
                    record_path,
                    signal_length,
                    bp_low,
                    bp_high,
                    fs,
                    filt_order,
                    baseline_samples,
                )
                if sig is None:
                    valid_mask[global_i] = False
                    lbl_buf.append([])
                    primary_buf.append("OTHER")
                    log.warning(f"Failed to load record: {record_path}")
                else:
                    X_buf[global_i] = sig[:signal_length]  # safety crop
                lbl_buf.append(row["labels"])
                primary_buf.append(row["primary_label"])
                ids_buf[global_i] = int(split_df.index[global_i])

        # Drop failed records
        n_failed = (~valid_mask).sum()
        if n_failed:
            log.warning(f"  Dropped {n_failed} unreadable records from {split_name}")
        X_buf = X_buf[valid_mask]
        ids_buf = ids_buf[valid_mask]
        lbl_buf_clean = [lbl_buf[i] for i in range(n) if valid_mask[i]]
        prim_clean = [primary_buf[i] for i in range(n) if valid_mask[i]]

        primary_arr = np.array(
            [classes.index(p) if p in classes else classes.index("OTHER") for p in prim_clean],
            dtype=np.int16,
        )
        return X_buf, ids_buf, lbl_buf_clean, primary_arr

    X_train_raw, ids_train, lbl_train, y_train_single = _load_split(df_train, "train")
    X_val_raw,   ids_val,   lbl_val,   y_val_single   = _load_split(df_val,   "val")
    X_test_raw,  ids_test,  lbl_test,  y_test_single  = _load_split(df_test,  "test")

    log.info(
        f"Loaded shapes — train: {X_train_raw.shape} | "
        f"val: {X_val_raw.shape} | test: {X_test_raw.shape}"
    )

    # ──────────────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STAGE 4 — Normalise (fit on train only) + encode labels")
    log.info("=" * 60)

    mu, sigma = _compute_train_stats(X_train_raw)
    log.info(f"Per-lead train mean (first 4 leads): {mu[:4].round(4)}")
    log.info(f"Per-lead train std  (first 4 leads): {sigma[:4].round(4)}")

    clip_range = (clip_lo, clip_hi)
    X_train = _normalise(X_train_raw, mu, sigma, clip_range)
    X_val   = _normalise(X_val_raw,   mu, sigma, clip_range)
    X_test  = _normalise(X_test_raw,  mu, sigma, clip_range)
    del X_train_raw, X_val_raw, X_test_raw   # free memory

    # Multi-label binarisation
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([classes])                          # fit with all possible classes

    y_train_ml = mlb.transform(lbl_train).astype(np.int8)
    y_val_ml   = mlb.transform(lbl_val).astype(np.int8)
    y_test_ml  = mlb.transform(lbl_test).astype(np.int8)

    log.info(f"Multilabel shapes — train: {y_train_ml.shape} | val: {y_val_ml.shape}")
    log.info(f"Class order in multilabel matrix: {list(mlb.classes_)}")

    # ──────────────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STAGE 5 — Save arrays")
    log.info("=" * 60)

    def _save(name: str, arr: np.ndarray) -> None:
        path = out_dir / name
        np.save(str(path), arr)
        log.info(f"  Saved {name:45s}  shape={arr.shape}  dtype={arr.dtype}")

    _save("X_train.npy",            X_train)
    _save("X_val.npy",              X_val)
    _save("X_test.npy",             X_test)
    _save("y_train_multilabel.npy", y_train_ml)
    _save("y_val_multilabel.npy",   y_val_ml)
    _save("y_test_multilabel.npy",  y_test_ml)
    _save("y_train_single.npy",     y_train_single)
    _save("y_val_single.npy",       y_val_single)
    _save("y_test_single.npy",      y_test_single)
    _save("record_ids_train.npy",   ids_train)
    _save("record_ids_val.npy",     ids_val)
    _save("record_ids_test.npy",    ids_test)

    stats = {
        "per_lead_mean": mu.tolist(),
        "per_lead_std":  sigma.tolist(),
        "lead_names":    list(cfg.ptbxl.lead_names),
        "class_order":   classes,
        "class_distribution": class_dist,
        "n_train": int(len(X_train)),
        "n_val":   int(len(X_val)),
        "n_test":  int(len(X_test)),
        "filter_params":       filter_params,
        "clip_range":          [clip_lo, clip_hi],
        "multilabel_confidence_threshold": ml_conf,
        "train_folds": train_folds,
        "val_folds":   val_folds,
        "test_folds":  test_folds,
        "seed": int(cfg.seeds[0]),
    }
    stats_path = out_dir / "preprocessing_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"  Saved preprocessing_stats.json")

    # ──────────────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STAGE 6 — Spot check")
    log.info("=" * 60)

    # Reload to verify round-trip integrity
    X_train_reload = np.load(out_dir / "X_train.npy")
    np.testing.assert_allclose(
        X_train_reload, X_train, rtol=0, atol=0,
        err_msg="Round-trip mismatch: saved vs reloaded X_train",
    )
    log.info("Round-trip assert passed (X_train save/reload exact match)")

    # Shape assertions
    assert X_train.shape[-1] == 12,    f"Expected 12 leads, got {X_train.shape[-1]}"
    assert X_train.shape[-2] == signal_length, \
        f"Expected {signal_length} samples, got {X_train.shape[-2]}"

    # NaN / Inf checks
    for split_name, arr in [
        ("X_train", X_train), ("X_val", X_val), ("X_test", X_test),
        ("y_train_ml", y_train_ml), ("y_val_ml", y_val_ml),
    ]:
        assert not np.isnan(arr).any(),  f"NaN found in {split_name}"
        assert not np.isinf(arr).any(),  f"Inf found in {split_name}"
    log.info("NaN / Inf checks passed on all saved arrays")

    # Summary statistics
    log.info("X_train statistics:")
    log.info(f"  mean={X_train.mean():.4f}  std={X_train.std():.4f}  "
             f"min={X_train.min():.4f}  max={X_train.max():.4f}")

    # Spot-check plot
    spot_check_path = str(out_dir / "preprocessing_spot_check.png")
    _spot_check_plot(
        X_train, y_train_single, classes, ids_train,
        fs, spot_check_path, log,
    )

    log.info("=" * 60)
    log.info("✓ Preprocessing complete. Arrays saved to outputs/processed/")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
