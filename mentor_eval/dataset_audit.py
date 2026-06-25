"""
mentor_eval/dataset_audit.py — scan PTB-XL for corrupted/unreadable signals
and metadata mismatches.

Checks per record (ecg_id):
  1. filename_lr resolves to an existing .dat/.hea pair
  2. wfdb.rdrecord() succeeds (file is actually readable)
  3. signal shape is (1000, 12), no NaN/Inf
  4. scp_codes parses to a non-empty dict (no missing diagnostic codes)

Writes:
  outputs/mentor_review/dataset_audit/audit_report.csv
  outputs/mentor_review/dataset_audit/audit_summary.txt

Usage:
    python -m mentor_eval.dataset_audit [--ptbxl-dir PATH] [--out-dir PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_config, get_logger
from mentor_eval.class_mapping import parse_scp_codes


def audit_dataset(ptbxl_dir: Path, out_dir: Path, log) -> pd.DataFrame:
    db_path = ptbxl_dir / "ptbxl_database.csv"
    if not db_path.exists():
        raise FileNotFoundError(f"ptbxl_database.csv not found at {db_path}")

    db = pd.read_csv(db_path)
    log.info(f"Auditing {len(db)} records from {db_path} …")

    rows = []
    for _, rec in tqdm(db.iterrows(), total=len(db), desc="audit"):
        eid = int(rec["ecg_id"])
        rel_path = str(rec["filename_lr"])
        dat_path = ptbxl_dir / f"{rel_path}.dat"
        hea_path = ptbxl_dir / f"{rel_path}.hea"

        if not dat_path.exists() or not hea_path.exists():
            rows.append({
                "ecg_id": eid, "reason": "missing_file",
                "detail": f"missing .dat/.hea at {rel_path}",
            })
            continue

        try:
            record = wfdb.rdrecord(str(ptbxl_dir / rel_path))
            sig = record.p_signal
        except Exception as exc:
            rows.append({
                "ecg_id": eid, "reason": "unreadable",
                "detail": f"wfdb.rdrecord failed: {exc}",
            })
            continue

        if sig.shape != (1000, 12):
            rows.append({
                "ecg_id": eid, "reason": "bad_shape",
                "detail": f"shape={sig.shape}, expected (1000, 12)",
            })
            continue

        if not np.isfinite(sig).all():
            n_bad = int((~np.isfinite(sig)).sum())
            rows.append({
                "ecg_id": eid, "reason": "nan_or_inf",
                "detail": f"{n_bad} non-finite samples",
            })
            continue

        scp = parse_scp_codes(rec.get("scp_codes", "{}"))
        if not scp:
            rows.append({
                "ecg_id": eid, "reason": "missing_scp_codes",
                "detail": "scp_codes parsed to an empty dict",
            })
            continue

    report = pd.DataFrame(rows, columns=["ecg_id", "reason", "detail"])
    out_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_dir / "audit_report.csv", index=False)

    n_total = len(db)
    n_bad   = len(report)
    summary_lines = [
        f"PTB-XL dataset audit — {n_total} total records",
        f"Flagged records: {n_bad} ({100 * n_bad / n_total:.2f}%)",
        f"Clean records:   {n_total - n_bad} ({100 * (n_total - n_bad) / n_total:.2f}%)",
        "",
        "Breakdown by reason:",
    ]
    if n_bad:
        for reason, count in report["reason"].value_counts().items():
            summary_lines.append(f"  {reason:<20} {count}")
    else:
        summary_lines.append("  (none)")

    summary = "\n".join(summary_lines)
    (out_dir / "audit_summary.txt").write_text(summary + "\n")
    log.info(summary)
    print(summary)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit PTB-XL for corrupted/mislabeled records.")
    parser.add_argument("--ptbxl-dir", type=str, default=None, help="Override path to data/ptbxl/")
    parser.add_argument("--out-dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("dataset_audit", cfg=cfg)

    ptbxl_dir = Path(args.ptbxl_dir) if args.ptbxl_dir else Path(cfg.paths.data.ptbxl)
    out_dir   = Path(args.out_dir) if args.out_dir else Path(cfg.paths.outputs.results).parent / "mentor_review" / "dataset_audit"

    audit_dataset(ptbxl_dir, out_dir, log)
    print(f"✓ Audit complete. Report → {out_dir / 'audit_report.csv'}")


if __name__ == "__main__":
    main()
