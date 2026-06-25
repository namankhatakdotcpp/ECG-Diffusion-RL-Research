"""
step01_data_load_and_visualise.py — PTB-XL data loading, verification & visualisation.

PURPOSE
-------
Walk the full data pipeline end-to-end before any preprocessing happens:
download PTB-XL, load metadata, extract ECG paths, load signals via WFDB,
verify signal shape, visualise ECG leads, build a 100-record sample dataset,
and inspect the SCP label system.

STAGES
------
  1. Download PTB-XL (skip if already present)
  2. Load metadata
  3. Extract ECG paths and verify the first 10 exist
  4. Load a single ECG signal via WFDB (demonstration)
  5. Verify signal shape (1000, 12), no NaN/Inf
  6. Visualise ECG leads (Lead I single + all 12 leads)
  7. Load 100 records with tqdm, handling failures gracefully
  8. Build a 100-record sample dataset array + metadata
  9. Inspect the SCP label system (code frequency + diagnostic superclass map)

OUTPUTS
-------
  outputs/processed/metadata_head.csv      — first 20 metadata rows
  outputs/processed/sample_100_ecgs.npy    — (N<=100, 1000, 12) sample array
  outputs/processed/sample_100_meta.csv    — ecg_id, primary_code, strat_fold
  outputs/processed/scp_code_summary.csv   — SCP code frequency across all records
  outputs/processed/label_mapping.json     — {class_name: [scp_codes]} superclass map
  outputs/results/fig_lead_I_single.png    — single-lead quick view
  outputs/results/fig_all_12_leads.png     — full 12-lead clinical layout

Usage:
    python step01_data_load_and_visualise.py
"""

from __future__ import annotations

import ast
import json
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, get_logger, set_seed

# ──────────────────────────────────────────────────────────────────────────────
# Exhaustive SCP-code → diagnostic superclass mapping
# Source: PTB-XL paper (Wagner et al. 2020, Table 1) + scp_statements.csv
# ──────────────────────────────────────────────────────────────────────────────

_SCP_SUPERCLASS_MAP: dict[str, str] = {
    "NORM": "NORM",
    "AMI": "MI", "IMI": "MI", "LMI": "MI", "PMI": "MI",
    "ALMI": "MI", "ASMI": "MI", "ILMI": "MI", "IPMI": "MI", "IPLMI": "MI",
    "INJAL": "MI", "INJAS": "MI", "INJIL": "MI", "INJIN": "MI", "INJLA": "MI",
    "ISCAL": "MI", "ISCAN": "MI", "ISCAS": "MI", "ISCIL": "MI",
    "ISCIN": "MI", "ISCLA": "MI", "ISC_": "MI",
    "STTC": "STTC", "NST_": "STTC", "NDT": "STTC", "DIG": "STTC",
    "LNGQT": "STTC", "STD_": "STTC", "STDD": "STTC", "VCLVH": "STTC",
    "LBBB": "CD", "RBBB": "CD", "CLBBB": "CD", "CRBBB": "CD",
    "ILBBB": "CD", "IRBBB": "CD", "LAFB": "CD", "LPFB": "CD",
    "AVB": "CD", "1AVB": "CD", "2AVB": "CD", "3AVB": "CD",
    "WPW": "CD", "PSVT": "CD", "SVTAC": "CD", "IVCD": "CD", "PACE": "CD",
    "LVH": "HYP", "RVH": "HYP", "SEHYP": "HYP",
    "LAE": "HYP", "RAE": "HYP", "HVOLT": "HYP", "LVOLT": "HYP",
    "AFIB": "AFIB", "AFLT": "AFIB",
}

LEAD_COLORS = [
    "#1B3A6B", "#0E7C86", "#C9872A", "#1E8449",
    "#8E44AD", "#E74C3C", "#2980B9", "#F39C12",
    "#1ABC9C", "#E67E22", "#9B59B6", "#34495E",
]


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 — Download
# ──────────────────────────────────────────────────────────────────────────────

# NOTE: PhysioNet wget/wfdb download replaced with local ZIP extraction.
# Download is unreliable (timeouts). Manually download the ZIP from:
# https://physionet.org/content/ptb-xl/1.0.3/
# and place it in the project root before running.

ZIP_NAME = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"


def _ptbxl_layout_ok(dest: Path) -> bool:
    """True only if BOTH the metadata CSV and the records100 waveform tree
    are present — the CSV alone is not sufficient evidence of a complete
    extraction (it can exist on its own if someone copied just that file).
    """
    return (dest / "ptbxl_database.csv").exists() and (dest / "records100").is_dir()


def _download_ptbxl(dest: Path, log) -> None:
    if _ptbxl_layout_ok(dest):
        log.info("✓ PTB-XL already extracted. Skipping.")
        return

    search_dirs = [
        Path("."),
        Path(".."),
        Path.home(),
    ]
    zip_path = None
    for d in search_dirs:
        candidate = d / ZIP_NAME
        if candidate.exists():
            zip_path = candidate.resolve()
            break

    if zip_path is None:
        print(
            f"\n[ERROR] PTB-XL ZIP not found, and {dest} is missing or incomplete\n"
            f"  (needs both ptbxl_database.csv and a records100/ folder).\n"
            f"  Expected filename : {ZIP_NAME}\n"
            f"  Searched in       : ./, ../, ~/\n"
            f"\n"
            f"  Download from: https://physionet.org/content/ptb-xl/1.0.3/\n"
            f"  Then place the ZIP in the project root and re-run.\n"
        )
        sys.exit(1)

    # Extract into a scratch staging dir first — PhysioNet's ZIP wraps
    # everything in its own top-level folder name (NOT "ptbxl/"), so we
    # can't assume the internal layout. Find wherever ptbxl_database.csv
    # actually landed, then flatten that directory's contents into `dest`.
    staging = dest.parent / "_ptbxl_extract_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    log.info(f"Found ZIP at {zip_path} — extracting (staging) …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(path=staging)

    csv_matches = list(staging.rglob("ptbxl_database.csv"))
    if not csv_matches:
        shutil.rmtree(staging)
        print(
            f"\n[ERROR] Extracted {zip_path} but found no ptbxl_database.csv anywhere "
            f"inside it.\n  The ZIP may be corrupt or incomplete — re-download it.\n"
        )
        sys.exit(1)

    data_root = csv_matches[0].parent
    log.info(f"Found dataset root inside ZIP at {data_root} — moving into {dest} …")
    dest.mkdir(parents=True, exist_ok=True)
    for item in data_root.iterdir():
        target = dest / item.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(item), str(target))

    shutil.rmtree(staging)

    if not _ptbxl_layout_ok(dest):
        print(
            f"\n[ERROR] Extraction finished but {dest} still doesn't have both "
            f"ptbxl_database.csv and records100/ — the ZIP may be missing the "
            f"waveform files. Re-download from physionet.org and retry.\n"
        )
        sys.exit(1)

    log.info("✓ Extraction complete and verified (ptbxl_database.csv + records100/ present).")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 9 helpers — SCP superclass mapping (supplementary to PROMPT 1's spec;
# kept from the original step01 implementation since step02/03 read it)
# ──────────────────────────────────────────────────────────────────────────────

def _build_label_mapping_from_csv(scp_csv: Path, log) -> dict[str, str]:
    """Augment _SCP_SUPERCLASS_MAP with scp_statements.csv's diagnostic_class column."""
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
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    log = get_logger("step01_data_load_and_visualise", cfg=cfg)
    set_seed(cfg.seeds[0])

    ptbxl_dir     = Path(cfg.paths.data.ptbxl)
    processed_dir = Path(cfg.paths.outputs.processed)
    results_dir   = Path(cfg.paths.outputs.results)
    for d in (ptbxl_dir, processed_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 1 — Download
    # ════════════════════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("STAGE 1 — Download PTB-XL")
    log.info("=" * 60)
    _download_ptbxl(ptbxl_dir, log)

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 2 — Load metadata
    # ════════════════════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("STAGE 2 — Load metadata")
    log.info("=" * 60)

    metadata = pd.read_csv(ptbxl_dir / "ptbxl_database.csv")
    log.info(f"Metadata loaded: {len(metadata)} records")
    log.info(f"Columns: {list(metadata.columns)}")
    print(metadata.head())

    metadata.head(20).to_csv(processed_dir / "metadata_head.csv", index=False)
    log.info(f"Saved metadata_head.csv → {processed_dir / 'metadata_head.csv'}")

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 3 — Extract ECG paths
    # ════════════════════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("STAGE 3 — Extract ECG paths")
    log.info("=" * 60)

    metadata["filepath"] = metadata["filename_lr"].apply(lambda f: str(ptbxl_dir / f))

    for path in metadata["filepath"].head(10):
        assert Path(path + ".dat").exists(), f"Missing: {path}.dat"
        assert Path(path + ".hea").exists(), f"Missing: {path}.hea"
    log.info("✓ ECG paths verified for first 10 records")
    log.info(f"Example path: {metadata['filepath'].iloc[0]}")

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 4 — Load a single ECG signal via WFDB
    # ════════════════════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("STAGE 4 — Load a single ECG via WFDB")
    log.info("=" * 60)

    record = wfdb.rdrecord(metadata["filepath"].iloc[0])
    signal = record.p_signal

    log.info(f"Signal shape: {signal.shape}")
    log.info(f"Sampling frequency: {record.fs} Hz")
    log.info(f"Lead names: {record.sig_name}")
    log.info(f"Signal duration: {signal.shape[0] / record.fs:.1f} seconds")
    log.info(f"Min value: {signal.min():.4f}  Max: {signal.max():.4f}")

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 5 — Verify signal shape
    # ════════════════════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("STAGE 5 — Verify signal shape")
    log.info("=" * 60)

    assert signal.shape == (1000, 12), f"Expected (1000, 12), got {signal.shape}"
    assert not np.isnan(signal).any(), "NaN values in signal"
    assert not np.isinf(signal).any(), "Inf values in signal"
    log.info("✓ Signal shape verified: (1000, 12)")
    log.info("✓ No NaN or Inf values")

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 6 — Visualise ECG leads
    # ════════════════════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("STAGE 6 — Visualise ECG leads")
    log.info("=" * 60)

    time_axis = np.arange(1000) / 100

    # Figure A — Lead I single
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(time_axis, signal[:, 0], color="#1B3A6B", linewidth=0.8)
    ax.set_title(f"PTB-XL Record #{metadata['ecg_id'].iloc[0]} — Lead I")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude (mV)")
    ax.grid(True, alpha=0.3)
    fig.savefig(str(results_dir / "fig_lead_I_single.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure B — All 12 leads
    fig, axes = plt.subplots(12, 1, figsize=(16, 20), sharex=True)
    lead_names = record.sig_name
    for i, (ax, name, color) in enumerate(zip(axes, lead_names, LEAD_COLORS)):
        ax.plot(time_axis, signal[:, i], color=color, linewidth=0.7)
        ax.set_ylabel(name, fontsize=9, rotation=0, labelpad=25)
        ax.grid(True, alpha=0.2)
        ax.set_yticks([])
    axes[-1].set_xlabel("Time (seconds)", fontsize=11)
    fig.suptitle(
        f"PTB-XL Record #{metadata['ecg_id'].iloc[0]} — All 12 Leads",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(str(results_dir / "fig_all_12_leads.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info("✓ ECG visualisation saved (fig_lead_I_single.png, fig_all_12_leads.png)")

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 7 — Load multiple ECGs (100 records)
    # ════════════════════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("STAGE 7 — Load 100 records")
    log.info("=" * 60)

    all_ecgs: list[np.ndarray] = []
    all_ids:  list[int] = []
    failed:   list[int] = []

    for idx, row in tqdm(metadata.head(100).iterrows(), total=100, desc="Loading ECGs"):
        try:
            rec = wfdb.rdrecord(row["filepath"])
            sig = rec.p_signal

            if sig.shape != (1000, 12):
                failed.append(idx)
                continue
            if np.isnan(sig).any() or np.isinf(sig).any():
                failed.append(idx)
                continue

            all_ecgs.append(sig)
            all_ids.append(row["ecg_id"])
        except Exception:
            failed.append(idx)

    log.info(f"✓ Loaded {len(all_ecgs)}/100 records successfully")
    log.info(f"  Failed: {len(failed)} records (IDs: {failed})")

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 8 — Dataset builder
    # ════════════════════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("STAGE 8 — Dataset builder")
    log.info("=" * 60)

    all_ecgs_array = np.stack(all_ecgs, axis=0)
    log.info(f"Dataset array shape: {all_ecgs_array.shape}")
    log.info(f"Memory usage: {all_ecgs_array.nbytes / 1e6:.1f} MB")

    loaded_meta = metadata[metadata["ecg_id"].isin(all_ids)].copy()
    loaded_meta["scp_dict"] = loaded_meta["scp_codes"].apply(ast.literal_eval)
    loaded_meta["primary_code"] = loaded_meta["scp_dict"].apply(
        lambda d: max(d, key=d.get) if d else "UNKNOWN"
    )

    log.info("Class distribution of the 100 records:")
    print(loaded_meta["primary_code"].value_counts())

    np.save(str(processed_dir / "sample_100_ecgs.npy"), all_ecgs_array)
    loaded_meta[["ecg_id", "primary_code", "strat_fold"]].to_csv(
        processed_dir / "sample_100_meta.csv", index=False
    )
    log.info("✓ Dataset builder complete. Saved sample_100_ecgs.npy")

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 9 — Inspect label system
    # ════════════════════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("STAGE 9 — Inspect label system")
    log.info("=" * 60)

    scp_csv = ptbxl_dir / "scp_statements.csv"
    if scp_csv.exists():
        scp_df = pd.read_csv(scp_csv, index_col=0)
        log.info(f"Total SCP codes in PTB-XL: {len(scp_df)}")
        if "diagnostic" in scp_df.columns:
            diag = scp_df[scp_df["diagnostic"] == 1]
            cols = [c for c in ("description", "diagnostic_class") if c in diag.columns]
            print(diag[cols].to_string())
    else:
        log.warning("scp_statements.csv not found — skipping code description printout")
        scp_df = pd.DataFrame()

    # Code counts in the 100-record sample
    code_counts = Counter()
    for d in loaded_meta["scp_dict"]:
        code_counts.update(d.keys())
    log.info(f"Unique SCP codes in 100 records: {len(code_counts)}")
    log.info(f"Top 10: {code_counts.most_common(10)}")

    # Full SCP code frequency across ALL records (not just the 100-sample)
    full_code_counts = Counter()
    for raw in metadata["scp_codes"]:
        try:
            d = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            continue
        full_code_counts.update(d.keys())

    summary_rows = [
        {"scp_code": code, "count": count}
        for code, count in sorted(full_code_counts.items(), key=lambda kv: -kv[1])
    ]
    pd.DataFrame(summary_rows).to_csv(processed_dir / "scp_code_summary.csv", index=False)
    log.info(f"Saved scp_code_summary.csv ({len(summary_rows)} unique codes across all records)")

    # Supplementary: SCP-code → diagnostic-superclass mapping (used by step02/03)
    if scp_csv.exists():
        flat_map = _build_label_mapping_from_csv(scp_csv, log)
    else:
        log.warning("scp_statements.csv not found — using built-in SCP map only")
        flat_map = dict(_SCP_SUPERCLASS_MAP)

    inverted = _invert_mapping(flat_map)
    mapping_path = processed_dir / "label_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(inverted, f, indent=2, sort_keys=True)
    log.info(f"Saved label_mapping.json → {mapping_path}")

    # ════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ════════════════════════════════════════════════════════════════════════
    summary_rows_print = [
        ("Total records",     f"{len(metadata):,}"),
        ("Records loaded",    f"{len(all_ecgs)}/100"),
        ("Signal shape",      "(1000, 12)"),
        ("Sampling rate",     "100 Hz"),
        ("Duration",          "10 seconds"),
        ("Leads",             "12"),
        ("Unique SCP codes",  str(len(full_code_counts))),
        ("Figures saved",     "2"),
    ]
    col_w = max(len(k) for k, _ in summary_rows_print) + 2
    print()
    print("┌" + "─" * (col_w + 24) + "┐")
    print("│  PTB-XL Data Loading Summary".ljust(col_w + 24) + "│")
    print("├" + "─" * (col_w + 24) + "┤")
    for k, v in summary_rows_print:
        print(f"│  {k:<{col_w}}│  {v:<19}│")
    print("└" + "─" * (col_w + 24) + "┘")
    print()

    print("✓ Step 1 complete. 100-record dataset built. Ready for step02_preprocessing.py")


if __name__ == "__main__":
    main()
