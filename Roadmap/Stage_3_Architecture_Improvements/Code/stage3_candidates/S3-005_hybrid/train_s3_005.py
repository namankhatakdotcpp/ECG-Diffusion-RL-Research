"""
S3-005 -- Hybrid (LayerScale on every block, plus an extra learnable
boost scalar on blocks 5-6 specifically). Combines the network-wide
mechanism (S3-002) with the targeted late-block emphasis (S3-003),
rather than picking one -- per Stage3_Roadmap.md's candidate table,
motivated by "multiple findings" jointly (Items 1, 2A, 2B, 3, 5).

Usage:
    python train_s3_005.py                # full training (Phase 2, GPU)
    python train_s3_005.py --smoke-test    # CPU shape/gradient check only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CANDIDATES_DIR = Path(__file__).resolve().parents[1]
if str(CANDIDATES_DIR) not in sys.path:
    sys.path.insert(0, str(CANDIDATES_DIR))

from common_train import train_variant, REPO_ROOT  # noqa: E402

STAGE2_CODE_DIR = REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Code"
if str(STAGE2_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_CODE_DIR))
from common.io import load_config, get_logger  # noqa: E402

VARIANT = "hybrid"
RUN_ID = "S3-005"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                         help="CPU forward/backward shape check only, no data pipeline, no training")
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger(f"train_{RUN_ID}", cfg=cfg)

    if args.smoke_test:
        from smoke_test import run_smoke_test
        run_smoke_test(cfg, log, VARIANT, RUN_ID)
        return

    train_variant(cfg, log, variant=VARIANT, run_id=RUN_ID)


if __name__ == "__main__":
    main()
