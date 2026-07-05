"""
Roadmap/_infra/export_latex_tables.py -- LaTeX table generator for the
paper, parameterized to accept a results table (list of dicts, or a
DataFrame) as input rather than reading/computing metrics itself.

Deliberately does NOT read Stage3_Comparison.csv directly at import
time or hardcode any real numbers -- render_stage3_table() takes rows
as an argument, so it can be pointed at real data once available, or at
synthetic data for a smoke test, without code changes either way. This
is generation CODE, not a final asset: running main() against today's
actual (still-empty) Stage3_Comparison.csv produces a table that
correctly says every candidate is "not yet evaluated" -- it is not
meant to be treated as a camera-ready paper table until real Wave 1/2
numbers exist.

Usage:
    python export_latex_tables.py                 # renders from the real
                                                    # Stage3_Comparison.csv
                                                    # (whatever it currently
                                                    # contains)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Optional

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
STAGE3_REPORTS_DIR = REPO_ROOT / "Roadmap" / "Stage_3_Architecture_Improvements" / "Reports"
STAGE3_CSV_PATH = STAGE3_REPORTS_DIR / "Stage3_Comparison.csv"
OUT_DIR = REPO_ROOT / "Roadmap" / "_infra" / "paper_tables"


def _escape_latex(value: object) -> str:
    if value is None or value == "":
        return "--"
    s = str(value)
    for ch, esc in (("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        s = s.replace(ch, esc)
    return s


def render_stage3_table(rows: list[dict], columns: Optional[list[str]] = None, caption: str = "") -> str:
    """rows: list of dicts (e.g. from csv.DictReader or compare_candidates.py's
    own row objects). columns: subset/order of keys to render -- defaults to
    every key present in the first row. Returns a LaTeX booktabs table as a
    string; writes nothing itself (main() handles file output) so this is
    unit-testable in isolation."""
    if not rows:
        return "% No rows to render -- nothing fabricated.\n"

    columns = columns or list(rows[0].keys())
    col_spec = "l" * len(columns)

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    if caption:
        lines.append(rf"\caption{{{_escape_latex(caption)}}}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(_escape_latex(c) for c in columns) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(_escape_latex(row.get(c)) for c in columns) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    if not STAGE3_CSV_PATH.exists():
        print(f"[BLOCKED] {STAGE3_CSV_PATH} not found -- run compare_candidates.py first. Nothing fabricated.")
        sys.exit(1)

    with open(STAGE3_CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    columns = ["Candidate", "Variant", "Params (M)", "Generated Accuracy",
               "Generated Macro-F1", "Generated Macro-AUC", "Optimizer Config", "Status"]
    tex = render_stage3_table(
        rows, columns=columns,
        caption="Stage 3 candidate comparison (auto-generated from Stage3_Comparison.csv).",
    )

    all_not_evaluated = all(r.get("Status", "").startswith("not yet evaluated") for r in rows)
    if all_not_evaluated:
        tex = (
            "% WARNING: every candidate below is 'not yet evaluated' -- this table\n"
            "% reflects the current (pre-training-completion) state, NOT final results.\n"
            "% Do not include in a paper draft until Stage3_Comparison.csv has real values.\n"
        ) + tex

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "stage3_comparison_table.tex"
    out_path.write_text(tex)
    print(f"Wrote {out_path}")
    if all_not_evaluated:
        print("[NOTE] Source data has no evaluated candidates yet -- table is a structural placeholder, not final.")


if __name__ == "__main__":
    main()
