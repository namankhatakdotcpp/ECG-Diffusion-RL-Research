"""
S3-001 -- Baseline (no architecture change, control).

Re-verifies the current checkpoint's own numbers under Stage 3's
training entry point (same architecture as step04_transformer_
diffusion.py's ECGTransformerDiffusion, via model_variants.build_variant_model
with variant="baseline") -- included so Phase 4's comparison table has
a same-pipeline control, not just the pre-existing outputs/models/
diffusion_best.pt trained under a slightly different historical code
path.

Usage:
    python train_s3_001.py                # full training (Phase 2, GPU)
    python train_s3_001.py --smoke-test    # CPU shape/gradient check only
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

VARIANT = "baseline"
RUN_ID = "S3-001"


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
