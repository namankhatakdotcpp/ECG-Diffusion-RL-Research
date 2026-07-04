"""
S3-004 -- Residual scaling (learnable SCALAR, not per-channel, gain on
every block's two residual branches). Motivated by Item 3's own
block-level (not channel-level) granularity -- tests whether a coarser
scaling mechanism is sufficient, before committing to LayerScale's
finer-grained (and higher-parameter-count) per-channel version.

Usage:
    python train_s3_004.py                # full training (Phase 2, GPU)
    python train_s3_004.py --smoke-test    # CPU shape/gradient check only
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

VARIANT = "residual_scaling"
RUN_ID = "S3-004"


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
