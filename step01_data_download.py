"""
step01_data_download.py — dataset acquisition and label mapping for ECG research.

Datasets:
  • PTB-XL  (primary)  — 21,837 12-lead ECGs at 100/500 Hz, PhysioNet
  • MIT-BIH  (secondary) — 48 2-lead arrhythmia recordings at 360 Hz, PhysioNet

Outputs:
  data/ptbxl/            — raw PTB-XL download (wfdb format)
  data/mitbih/           — raw MIT-BIH download (wfdb format)
  outputs/processed/label_mapping.json   — SCP code → 7-class superclass map

Usage:
    python step01_data_download.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import wfdb

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, get_logger, set_seed

# ──────────────────────────────────────────────────────────────────────────────
# Exhaustive SCP-code → diagnostic superclass mapping
# Source: PTB-XL paper (Wagner et al. 2020, Table 1) + scp_statements.csv
# ──────────────────────────────────────────────────────────────────────────────

_SCP_SUPERCLASS_MAP: dict[str, str] = {
    # ── NORM ──────────────────────────────────────────────────────────────────
    "NORM": "NORM",

    # ── MI ────────────────────────────────────────────────────────────────────
    # Acute / recent
    "AMI":   "MI",   # Acute MI
    "IMI":   "MI",   # Inferior MI
    "LMI":   "MI",   # Lateral MI
    "PMI":   "MI",   # Posterior MI
    "ALMI":  "MI",   # Anterior-lateral MI
    "ASMI":  "MI",   # Anterior-septal MI
    "ILMI":  "MI",   # Inferior-lateral MI
    "IPMI":  "MI",   # Inferior-posterior MI
    "IPLMI": "MI",   # Inferior-posterior-lateral MI
    # Injury patterns
    "INJAL": "MI",   # Injury: anterior-lateral
    "INJAS": "MI",   # Injury: anterior-septal
    "INJIL": "MI",   # Injury: inferior-lateral
    "INJIN": "MI",   # Injury: inferior
    "INJLA": "MI",   # Injury: lateral
    # Ischaemia patterns
    "ISCAL": "MI",   # Ischaemia: anterior-lateral
    "ISCAN": "MI",   # Ischaemia: anterior
    "ISCAS": "MI",   # Ischaemia: anterior-septal
    "ISCIL": "MI",   # Ischaemia: inferior-lateral
    "ISCIN": "MI",   # Ischaemia: inferior
    "ISCLA": "MI",   # Ischaemia: lateral
    "ISC_":  "MI",   # Non-specific ischaemia

    # ── STTC (ST/T-wave Change) ───────────────────────────────────────────────
    "STTC":  "STTC",
    "NST_":  "STTC",  # Non-specific ST-change
    "NDT":   "STTC",  # Non-specific T-wave change
    "DIG":   "STTC",  # Digitalis effect
    "LNGQT": "STTC",  # Long QT interval
    "STD_":  "STTC",  # ST depression (non-specific)
    "STDD":  "STTC",  # ST depression with diffuse changes
    "VCLVH": "STTC",  # Voltage criteria for LVH + repolarisation

    # ── CD (Conduction Disturbance) ───────────────────────────────────────────
    # Bundle branch blocks
    "LBBB":  "CD",
    "RBBB":  "CD",
    "CLBBB": "CD",   # Complete LBBB
    "CRBBB": "CD",   # Complete RBBB
    "ILBBB": "CD",   # Incomplete LBBB
    "IRBBB": "CD",   # Incomplete RBBB
    # Fascicular blocks
    "LAFB":  "CD",   # Left anterior fascicular block
    "LPFB":  "CD",   # Left posterior fascicular block
    # AV blocks
    "AVB":   "CD",
    "1AVB":  "CD",   # 1st degree AV block
    "2AVB":  "CD",   # 2nd degree AV block
    "3AVB":  "CD",   # 3rd degree AV block
    # Pre-excitation / SVT
    "WPW":   "CD",   # Wolff-Parkinson-White
    "PSVT":  "CD",   # Paroxysmal SVT
    "SVTAC": "CD",   # SVT — atrio-ventricular conduction
    # Intraventricular
    "IVCD":  "CD",   # Intraventricular conduction delay
    # Pacemaker
    "PACE":  "CD",   # Pacemaker rhythm

    # ── HYP (Hypertrophy) ────────────────────────────────────────────────────
    "LVH":   "HYP",  # Left ventricular hypertrophy
    "RVH":   "HYP",  # Right ventricular hypertrophy
    "SEHYP": "HYP",  # Septal hypertrophy
    "LAE":   "HYP",  # Left atrial enlargement
    "RAE":   "HYP",  # Right atrial enlargement
    "HVOLT": "HYP",  # High voltage (LVH criteria)
    "LVOLT": "HYP",  # Low voltage

    # ── AFIB ─────────────────────────────────────────────────────────────────
    "AFIB":  "AFIB",
    "AFLT":  "AFIB",  # Atrial flutter
}


def _build_label_mapping_from_csv(
    scp_csv: Path,
    log,
) -> dict[str, str]:
    """Augment _SCP_SUPERCLASS_MAP with scp_statements.csv.

    PTB-XL's scp_statements.csv has a ``diagnostic_class`` column containing
    the five primary superclasses (NORM, MI, STTC, CD, HYP).  Any SCP codes
    present in the CSV but not in our hand-curated map are added here; codes
    already mapped keep their existing assignment so AFIB is preserved.
    """
    valid_classes = {"NORM", "MI", "STTC", "CD", "HYP", "AFIB", "OTHER"}
    mapping: dict[str, str] = dict(_SCP_SUPERCLASS_MAP)

    scp_df = pd.read_csv(scp_csv, index_col=0)
    col = "diagnostic_class" if "diagnostic_class" in scp_df.columns else None
    if col is None:
        log.warning("scp_statements.csv has no 'diagnostic_class' column — using built-in map")
        return mapping

    added = 0
    for code, row in scp_df.iterrows():
        code_upper = str(code).strip().upper()
        dc = str(row[col]).strip().upper()
        if code_upper not in mapping and dc in valid_classes:
            mapping[code_upper] = dc
            added += 1

    log.info(f"scp_statements.csv added {added} previously unknown SCP codes to the map")
    return mapping


def _invert_mapping(flat: dict[str, str]) -> dict[str, list[str]]:
    """Turn code→class into class→[codes] for label_mapping.json."""
    inv: dict[str, list[str]] = {}
    for code, cls in sorted(flat.items()):
        inv.setdefault(cls, []).append(code)
    return inv


# ──────────────────────────────────────────────────────────────────────────────
# Disk-usage helper
# ──────────────────────────────────────────────────────────────────────────────

def _dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / 1024 / 1024


def _count_dat_files(path: Path) -> int:
    return len(list(path.rglob("*.dat")))


# ──────────────────────────────────────────────────────────────────────────────
# Download helpers
# ──────────────────────────────────────────────────────────────────────────────

def _download_ptbxl(dest: Path, log) -> None:
    """Download PTB-XL via wfdb.dl_database; skips if already present."""
    sentinel = dest / "ptbxl_database.csv"
    if sentinel.exists():
        log.info(f"PTB-XL already present at {dest} — skipping download.")
        return

    log.info(f"Downloading PTB-XL → {dest}  (this may take 5-10 minutes …)")
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        wfdb.dl_database("ptb-xl", str(dest))
    except Exception as exc:
        log.error(f"wfdb.dl_database failed: {exc}")
        log.error(
            "Manual fallback: wget -r -N -c -np "
            "https://physionet.org/files/ptb-xl/1.0.3/ -P data/ptbxl/"
        )
        raise
    elapsed = time.time() - t0
    log.info(f"PTB-XL download complete in {elapsed/60:.1f} min")


def _download_mitbih(dest: Path, log) -> None:
    """Download MIT-BIH via wfdb.dl_database; skips if already present."""
    # MIT-BIH has no single manifest CSV so we check for a known record
    sentinel = dest / "100.dat"
    if sentinel.exists():
        log.info(f"MIT-BIH already present at {dest} — skipping download.")
        return

    log.info(f"Downloading MIT-BIH → {dest}  (this may take 2-3 minutes …)")
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        wfdb.dl_database("mitdb", str(dest))
    except Exception as exc:
        log.error(f"wfdb.dl_database failed: {exc}")
        log.error(
            "Manual fallback: wget -r -N -c -np "
            "https://physionet.org/files/mitdb/1.0.0/ -P data/mitbih/"
        )
        raise
    elapsed = time.time() - t0
    log.info(f"MIT-BIH download complete in {elapsed/60:.1f} min")


# ──────────────────────────────────────────────────────────────────────────────
# Verification helpers
# ──────────────────────────────────────────────────────────────────────────────

def _verify_ptbxl(ptbxl_dir: Path, log) -> dict:
    """Verify PTB-XL integrity; return summary dict."""
    result = {"status": "FAIL", "records": 0, "size_mb": 0.0, "issues": []}

    db_csv = ptbxl_dir / "ptbxl_database.csv"
    scp_csv = ptbxl_dir / "scp_statements.csv"

    if not db_csv.exists():
        result["issues"].append("ptbxl_database.csv not found")
        return result
    if not scp_csv.exists():
        result["issues"].append("scp_statements.csv not found")

    df = pd.read_csv(db_csv)
    n = len(df)
    result["records"] = n
    if n < 21000:
        result["issues"].append(f"Expected ~21837 rows in ptbxl_database.csv, got {n}")

    # Spot-check 3 random records using filename_lr (100 Hz)
    if "filename_lr" in df.columns:
        sample_rows = df.sample(min(3, len(df)), random_state=42)
        unreadable: list[str] = []
        for _, row in sample_rows.iterrows():
            path = str(ptbxl_dir / str(row["filename_lr"]).strip())
            try:
                rec = wfdb.rdrecord(path, sampfrom=0, sampto=100)
                if rec.p_signal is None:
                    unreadable.append(path)
            except Exception as exc:
                unreadable.append(f"{path} ({exc})")
        if unreadable:
            result["issues"].append(f"Unreadable records: {unreadable}")
        else:
            log.info(f"PTB-XL spot-check: 3/3 random records readable")
    else:
        result["issues"].append("ptbxl_database.csv missing 'filename_lr' column")

    result["size_mb"] = _dir_size_mb(ptbxl_dir)
    result["status"] = "OK" if not result["issues"] else "WARN"
    return result


def _verify_mitbih(mitbih_dir: Path, log) -> dict:
    """Verify MIT-BIH integrity; return summary dict."""
    result = {"status": "FAIL", "records": 0, "size_mb": 0.0, "issues": []}

    dat_files = list(mitbih_dir.rglob("*.dat"))
    n = len(dat_files)
    result["records"] = n

    if n < 40:
        result["issues"].append(f"Expected ≥ 48 .dat files, found {n}")
        return result

    # Spot-check 1 record
    candidate = dat_files[0]
    record_path = str(candidate.with_suffix(""))
    try:
        rec = wfdb.rdrecord(record_path, sampfrom=0, sampto=360)
        if rec.p_signal is None:
            result["issues"].append(f"Null signal from {record_path}")
        else:
            log.info(f"MIT-BIH spot-check: '{candidate.name}' readable — shape {rec.p_signal.shape}")
    except Exception as exc:
        result["issues"].append(f"Could not read {record_path}: {exc}")

    result["size_mb"] = _dir_size_mb(mitbih_dir)
    result["status"] = "OK" if not result["issues"] else "WARN"
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────────────

def _print_summary(rows: list[tuple[str, int, float, str]]) -> None:
    """Print a formatted download-summary table."""
    col_w = [20, 10, 14, 8]
    header = f"{'Dataset':<{col_w[0]}} {'Records':>{col_w[1]}} {'Size (MB)':>{col_w[2]}} {'Status':>{col_w[3]}}"
    sep = "-" * (sum(col_w) + len(col_w) - 1)
    print()
    print(sep)
    print(header)
    print(sep)
    for dataset, records, size_mb, status in rows:
        print(
            f"{dataset:<{col_w[0]}} {records:>{col_w[1]},} {size_mb:>{col_w[2]}.1f} {status:>{col_w[3]}}"
        )
    print(sep)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    log = get_logger("step01_data_download", cfg=cfg)
    set_seed(cfg.seeds[0])

    ptbxl_dir = Path(cfg.paths.data.ptbxl)
    mitbih_dir = Path(cfg.paths.data.mitbih)
    processed_dir = Path(cfg.paths.outputs.processed)

    # Ensure all output directories exist
    for d in [ptbxl_dir, mitbih_dir, processed_dir]:
        d.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("STEP 1 — Download datasets")
    log.info("=" * 60)

    _download_ptbxl(ptbxl_dir, log)
    _download_mitbih(mitbih_dir, log)

    log.info("=" * 60)
    log.info("STEP 2 — Verify downloads")
    log.info("=" * 60)

    ptbxl_summary  = _verify_ptbxl(ptbxl_dir, log)
    mitbih_summary = _verify_mitbih(mitbih_dir, log)

    for dataset, summary in [("PTB-XL", ptbxl_summary), ("MIT-BIH", mitbih_summary)]:
        if summary["issues"]:
            for issue in summary["issues"]:
                log.warning(f"{dataset}: {issue}")
        else:
            log.info(f"{dataset}: verification passed")

    log.info("=" * 60)
    log.info("STEP 3 — Build label mapping")
    log.info("=" * 60)

    scp_csv = ptbxl_dir / "scp_statements.csv"
    if scp_csv.exists():
        flat_map = _build_label_mapping_from_csv(scp_csv, log)
    else:
        log.warning("scp_statements.csv not found — using built-in SCP map only")
        flat_map = dict(_SCP_SUPERCLASS_MAP)

    # Log per-class code counts
    inverted = _invert_mapping(flat_map)
    for cls in list(cfg.ptbxl.classes):
        codes = inverted.get(cls, [])
        log.info(f"  {cls:>8s}: {len(codes):>3} SCP codes  — {codes[:6]}{'…' if len(codes)>6 else ''}")

    mapping_path = processed_dir / "label_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(inverted, f, indent=2, sort_keys=True)
    log.info(f"Saved label_mapping.json → {mapping_path}")

    log.info("=" * 60)
    log.info("STEP 4 — Summary")
    log.info("=" * 60)

    _print_summary([
        ("PTB-XL",  ptbxl_summary["records"],  ptbxl_summary["size_mb"],  ptbxl_summary["status"]),
        ("MIT-BIH", mitbih_summary["records"],  mitbih_summary["size_mb"], mitbih_summary["status"]),
    ])

    if ptbxl_summary["status"] == "FAIL":
        log.error("PTB-XL verification FAILED — check warnings above before running step02.")
        sys.exit(1)

    print("✓ Data download complete. Ready for step02_preprocessing.py")


if __name__ == "__main__":
    main()
