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

# Any of these LITERAL 2-backslash sequences appearing in rendered LaTeX
# output means _escape_latex ran twice on the same value somewhere (the
# exact bug class caught during development, at the caption call site).
# NOTE: these are plain substrings, not regexes -- "\\_" here is TWO
# literal backslash characters followed by "_" (Python escaping: each
# "\\" is one literal backslash). A naive regex r"\\_" would match a
# single correctly-escaped "\_" too (one backslash is itself a valid
# regex escape for a backslash) -- caught this exact false-positive
# while writing this test, which is itself a small instance of the same
# lesson: verify a check against a known-good input before trusting it.
_DOUBLE_ESCAPE_SEQUENCES = ["\\\\_", "\\\\%", "\\\\&", "\\\\#"]


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


def test_no_double_escaping_anywhere_in_rendered_table():
    """Repo-wide regression guard for the double-escaping bug CLASS, not
    just the single caption call site where it was originally caught.
    Special characters are injected into every place render_stage3_table
    accepts a string: the caption, a column header, and multiple cell
    values -- confirms none of _escape_latex's call sites (used 3x inside
    render_stage3_table: caption, header, rows) can produce a doubled
    backslash sequence."""
    rows = [
        {"Weird_Column%Name": "value_with_underscore", "B": "50% complete", "C": "a&b#c"},
        {"Weird_Column%Name": "another_value", "B": "100%", "C": None},
    ]
    result = render_stage3_table(
        rows,
        columns=["Weird_Column%Name", "B", "C"],
        caption="Uses Stage3_Comparison.csv at 50% completion, a&b#c",
    )
    for seq in _DOUBLE_ESCAPE_SEQUENCES:
        assert seq not in result, f"found double-escaped sequence {seq!r} in rendered output:\n{result}"


def test_escape_latex_itself_is_not_idempotent_by_design():
    """Documents the actual invariant: _escape_latex is NOT safe to call
    twice on its own output (calling it on an already-escaped string WILL
    double-escape) -- the fix for the original bug was "don't pre-escape
    before calling it", not "make escaping idempotent". This test pins
    that documented behavior so a future refactor doesn't silently change
    the contract without the double-escape tests above being updated to
    match."""
    once = _escape_latex("a_b")
    twice = _escape_latex(once)
    assert once == r"a\_b"
    assert twice == r"a\\_b"
    assert once != twice


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
