"""
mentor_eval/stage3_subband_master_comparison.py — aggregates all Stage 3
candidates' subband_similarity_metrics.csv (mentor_eval.subband_similarity_
metrics, run per-candidate by run_stage3_queue.py's eval step) into one
long-format table: one row per (Candidate, Disease, Subband).

Complementary to stage3_master_comparison.py's disease-level table, not a
replacement -- the disease-level table answers "how similar overall", this
one answers "which frequency band is failing" (A3=slow-wave/morphology,
D3, D2, D1=high-frequency QRS; see mentor_eval/subband_features.py for the
bior4.4 / level-3 decomposition this project uses, recalibrated from
Sharma et al.'s J=6/1000Hz to J=3 for this project's 100Hz data).

No Hausdorff column here -- subband_similarity_metrics.py does not compute
it per-band (only Mahalanobis/Bhattacharyya/cosine), unlike the disease-level
table. AFIB is not a missing/N/A row here: subband_similarity_metrics.py's
own BOX_CLASSES = ["Normal", "STEMI", "NSTEMI"] never includes it, so it is
structurally absent from the source data, not filtered out by this script.

Reuses VARIANT_BY_RUN_ID / RESULTS_ROOT from run_stage3_queue.py, same as
stage3_master_comparison.py.

Safe to re-run at any point: a candidate with no subband_similarity_metrics.csv
yet is reported as its own row ("not yet evaluated"), never fabricated.

Usage:
    python -m mentor_eval.stage3_subband_master_comparison
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STAGE3_CODE_DIR = (
    REPO_ROOT / "Roadmap" / "Stage_3_Architecture_Improvements" / "Code" / "stage3_candidates"
)
if str(STAGE3_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE3_CODE_DIR))

from run_stage3_queue import VARIANT_BY_RUN_ID, RESULTS_ROOT  # noqa: E402
from mentor_eval.subband_features import SUBBAND_NAMES  # noqa: E402
from mentor_eval.subband_similarity_metrics import BOX_CLASSES  # noqa: E402

REPORTS_DIR = REPO_ROOT / "Roadmap" / "Stage_3_Architecture_Improvements" / "Reports"


def _candidate_rows(run_id: str) -> list[dict]:
    csv_path = RESULTS_ROOT / run_id / "mentor_eval" / "subband_similarity_metrics.csv"
    if not csv_path.exists():
        return [
            {
                "Candidate": run_id, "Disease": disease, "Subband": band,
                "Cosine": None, "Mahalanobis": None, "Bhattacharyya": None,
                "Status": "not yet evaluated -- no subband_similarity_metrics.csv found",
            }
            for disease in BOX_CLASSES for band in SUBBAND_NAMES
        ]

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        return [
            {
                "Candidate": run_id, "Disease": disease, "Subband": band,
                "Cosine": None, "Mahalanobis": None, "Bhattacharyya": None,
                "Status": f"error reading subband_similarity_metrics.csv: {exc}",
            }
            for disease in BOX_CLASSES for band in SUBBAND_NAMES
        ]

    rows = []
    for disease in BOX_CLASSES:
        for band in SUBBAND_NAMES:
            match = df[(df["class"] == disease) & (df["subband"] == band)]
            if match.empty:
                rows.append({
                    "Candidate": run_id, "Disease": disease, "Subband": band,
                    "Cosine": None, "Mahalanobis": None, "Bhattacharyya": None,
                    "Status": "missing row in source CSV",
                })
                continue
            r = match.iloc[0]
            flag = r.get("flag")
            rows.append({
                "Candidate": run_id, "Disease": disease, "Subband": band,
                "Cosine": r.get("cosine_similarity"),
                "Mahalanobis": r.get("mahalanobis"),
                "Bhattacharyya": r.get("bhattacharyya"),
                "Status": flag if isinstance(flag, str) and flag else "done",
            })
    return rows


def build_rows() -> list[dict]:
    rows = []
    for run_id in sorted(VARIANT_BY_RUN_ID):
        rows.extend(_candidate_rows(run_id))
    return rows


def _to_markdown(df: pd.DataFrame) -> str:
    header = list(df.columns)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for _, row in df.iterrows():
        cells = ["--" if pd.isna(v) or v is None else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    df = pd.DataFrame(build_rows())

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "Stage3_Subband_Master_Comparison.csv"
    md_path = REPORTS_DIR / "Stage3_Subband_Master_Comparison.md"

    df.to_csv(csv_path, index=False)

    n_done = (df["Status"] == "done").sum()
    md_lines = [
        "# Stage 3 -- Subband-wise Master Comparison\n",
        f"*{n_done}/{len(df)} (candidate, disease, subband) rows evaluated. "
        "Complements Stage3_Disease_Master_Comparison.md (overall similarity); "
        "this table shows which frequency band -- A3 (slow-wave/morphology), "
        "D3, D2, D1 (high-frequency QRS) -- diverges most from real ECGs, "
        "per mentor_eval/subband_features.py's bior4.4/level-3 decomposition. "
        "No Hausdorff column (not computed per-band). AFIB is not a row here "
        "-- it has no entry in subband_similarity_metrics.py's own class list, "
        "not filtered out by this script.*\n",
        _to_markdown(df),
    ]
    md_path.write_text("\n".join(md_lines) + "\n")

    print(df.to_string(index=False))
    print(f"\nWrote:\n  {csv_path}\n  {md_path}")


if __name__ == "__main__":
    main()
