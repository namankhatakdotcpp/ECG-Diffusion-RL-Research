"""
test_make_paper_figures.py -- smoke tests confirming each figure function
runs without error and writes a real (non-empty) PNG. Does not check
pixel content -- these are structural/summary figures, not something a
numeric assertion can meaningfully validate; the point of this test is
regression-catching an exception or a 0-byte file, not visual QA.

Run:
    pytest Roadmap/_infra/test_make_paper_figures.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_paper_figures as mpf


@pytest.mark.parametrize("fn_name", [
    "figure1_pipeline",
    "figure2_architecture_comparison",
    "figure3_stage2_findings",
    "figure6_clinical_eval_pipeline",
    "figure7_roadmap_timeline",
])
def test_figure_function_writes_nonempty_png(fn_name, tmp_path, monkeypatch):
    monkeypatch.setattr(mpf, "OUT_DIR", tmp_path)
    fn = getattr(mpf, fn_name)
    out_path = fn()
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert out_path.parent == tmp_path


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
