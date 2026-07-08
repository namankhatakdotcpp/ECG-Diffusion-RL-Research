"""
mentor_eval/stage3_master_comparison.py — aggregates all Stage 3 candidates'
disease_similarity_table.csv (mentor_eval.disease_similarity_table, run
per-candidate by run_stage3_queue.py's eval step) into one wide-format
master comparison table.

Reuses VARIANT_BY_RUN_ID / RESULTS_ROOT from run_stage3_queue.py (single
source of truth for which run_id maps to which variant, per
compare_candidates.py's own precedent) rather than re-declaring the
candidate list here.

Safe to re-run at any point, including while candidates are still
training/evaluating: a candidate with no disease_similarity_table.csv yet
is reported as "not yet evaluated", never fabricated or silently
skipped from the output. AFIB is never included as a data column -- it
has no trained diffusion class project-wide (see class_mapping.py /
MENTOR_CLASSES), so a numeric AFIB column would misrepresent a
structural N/A as a missing data point.

Usage:
    python -m mentor_eval.stage3_master_comparison
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
from mentor_eval.disease_similarity_table import FOURTH_METRIC_LABEL  # noqa: E402

REPORTS_DIR = REPO_ROOT / "Roadmap" / "Stage_3_Architecture_Improvements" / "Reports"

# Only the three diseases the mentor's table template asks for cross-candidate
# (Normal/STEMI/NSTEMI). AFIB and Healthy Sinus are in the per-candidate table
# but excluded here -- AFIB is structurally N/A for every candidate (nothing
# to compare across candidates), Healthy Sinus is pending class-taxonomy
# clarification (see disease_similarity_table.py's own caveat).
DISEASES = ["Normal", "STEMI", "NSTEMI"]

METRIC_SUFFIX = {
    "Cosine Similarity": "Cos",
    "Mahalanobis Distance": "Maha",
    "Hausdorff Distance": "Haus",
    FOURTH_METRIC_LABEL: "Bhat",
}


def _candidate_row(run_id: str) -> dict:
    row = {"Candidate": run_id}
    for disease in DISEASES:
        for suffix in METRIC_SUFFIX.values():
            row[f"{disease} {suffix}"] = None
    row["Status"] = "not yet evaluated -- no disease_similarity_table.csv found"

    csv_path = RESULTS_ROOT / run_id / "mentor_eval" / "disease_similarity_table.csv"
    if not csv_path.exists():
        return row

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        row["Status"] = f"error reading disease_similarity_table.csv: {exc}"
        return row

    for disease in DISEASES:
        matches = df[df["Disease"] == disease]
        if matches.empty:
            continue
        r = matches.iloc[0]
        for metric_col, suffix in METRIC_SUFFIX.items():
            row[f"{disease} {suffix}"] = r.get(metric_col)

    row["Status"] = "done"
    return row


def build_rows() -> list[dict]:
    return [_candidate_row(run_id) for run_id in sorted(VARIANT_BY_RUN_ID)]


def _to_markdown(df: pd.DataFrame) -> str:
    header = list(df.columns)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for _, row in df.iterrows():
        cells = ["--" if pd.isna(v) or v is None else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    rows = build_rows()
    df = pd.DataFrame(rows)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "Stage3_Disease_Master_Comparison.csv"
    md_path = REPORTS_DIR / "Stage3_Disease_Master_Comparison.md"

    df.to_csv(csv_path, index=False)

    n_done = (df["Status"] == "done").sum()
    md_lines = [
        "# Stage 3 -- Disease-wise Master Comparison\n",
        f"*{n_done}/{len(df)} candidates evaluated. AFIB excluded from this table -- "
        "no trained diffusion class project-wide, not a missing data point "
        "(see mentor_eval/disease_similarity_table.py). Rows for un-evaluated "
        "candidates are left blank, not fabricated.*\n",
        _to_markdown(df),
    ]
    md_path.write_text("\n".join(md_lines) + "\n")

    print(df.to_string(index=False))
    print(f"\nWrote:\n  {csv_path}\n  {md_path}")


if __name__ == "__main__":
    main()
