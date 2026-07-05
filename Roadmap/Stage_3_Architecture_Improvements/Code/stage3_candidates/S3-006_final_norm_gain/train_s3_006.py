"""
S3-006 -- final_norm/unproj gain (learnable per-channel gain applied at
the final_norm -> unproj boundary, plain TransformerBlock everywhere
else). Motivated by Phase 0 Task 0.2's borderline finding (retention_
ratio_conditioning 0.41 vs. 0.5 threshold) implicating this specific
boundary, tracked separately from the block-level gain candidates
(S3-002..005) per Stage3_Phase0_Report.md's Decision Gate A. Wave 3
priority (lower than S3-002..005) given the borderline margin, but
implemented now so it is ready to queue immediately once Wave 1/2
finish.

Usage:
    python train_s3_006.py                # full training (Phase 2, GPU)
    python train_s3_006.py --smoke-test    # CPU shape/gradient check only
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

VARIANT = "final_norm_gain"
RUN_ID = "S3-006"


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
