"""
Stage 3 -- two-iteration optimizer smoke test, all 5 candidates.

Runs smoke_test.run_optimizer_smoke_test for every variant and reports
PASS/FAIL per candidate. Per the Wave 1 execution directive: a FAIL on
any candidate is a STOP CONDITION -- that candidate must not proceed to
GPU training until fixed.

Usage:
    python run_optimizer_smoke_tests.py
"""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

REPO_ROOT = THIS_DIR.parents[3]
STAGE2_CODE_DIR = REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Code"
if str(STAGE2_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_CODE_DIR))

from common.io import load_config, get_logger  # noqa: E402
from smoke_test import run_optimizer_smoke_test  # noqa: E402

VARIANTS = [
    ("S3-001", "baseline"),
    ("S3-002", "layerscale"),
    ("S3-003", "late_gain"),
    ("S3-004", "residual_scaling"),
    ("S3-005", "hybrid"),
]


def main() -> None:
    cfg = load_config()
    log = get_logger("run_optimizer_smoke_tests", cfg=cfg)

    results = {}
    for run_id, variant in VARIANTS:
        results[run_id] = run_optimizer_smoke_test(cfg, log, variant=variant, run_id=run_id, n_iters=2)

    print("\n" + "=" * 50)
    print("Two-iteration optimizer smoke test -- summary")
    print("=" * 50)
    for run_id, ok in results.items():
        print(f"  [{'x' if ok else ' '}] {run_id}: {'PASS' if ok else 'FAIL'}")

    if not all(results.values()):
        failed = [rid for rid, ok in results.items() if not ok]
        print(f"\nSTOP CONDITION: {failed} failed the optimizer smoke test.")
        sys.exit(1)


if __name__ == "__main__":
    main()
