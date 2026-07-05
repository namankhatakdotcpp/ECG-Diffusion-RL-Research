"""
test_export_latex_tables.py -- regression tests for export_latex_tables.py.

Run:
    pytest Roadmap/_infra/test_export_latex_tables.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_latex_tables import render_stage3_table, _escape_latex


def test_empty_rows_produces_comment_not_fabricated_table():
    result = render_stage3_table([])
    assert "No rows to render" in result
    assert r"\begin{table}" not in result


def test_underscore_escaped_exactly_once():
    """Regression guard for the exact bug found during development: a
    caption or cell containing '_' must come out as a single '\\_' in
    the rendered LaTeX, not double-escaped to '\\\\_'."""
    rows = [{"Candidate": "S3-001", "Variant": "baseline"}]
    result = render_stage3_table(rows, caption="Uses Stage3_Comparison.csv")
    assert r"Stage3\_Comparison.csv" in result
    assert r"Stage3\\_Comparison.csv" not in result


def test_none_and_empty_string_render_as_dashes():
    rows = [{"Candidate": "S3-001", "Generated Accuracy": None, "Status": ""}]
    result = render_stage3_table(rows)
    assert "-- & --" in result


def test_column_subset_and_order_respected():
    rows = [{"A": "1", "B": "2", "C": "3"}]
    result = render_stage3_table(rows, columns=["C", "A"])
    lines = result.splitlines()
    header_line = [l for l in lines if "&" in l and r"\\" in l][0]
    assert header_line.startswith("C & A")


def test_special_chars_escaped():
    assert _escape_latex("50%") == r"50\%"
    assert _escape_latex("a_b") == r"a\_b"
    assert _escape_latex("a&b") == r"a\&b"
    assert _escape_latex(None) == "--"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
