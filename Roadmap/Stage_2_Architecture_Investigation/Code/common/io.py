"""
Stage 2 Tier 0 -- shared checkpoint/config loading.

LIFTED (copied, not moved) from the boilerplate repeated at the top of
`main()` in both stage2_tier0_item1_layerwise_magnitude_direction/
layerwise_direction_probe.py and stage2_tier0_item2_localized_gain/
item2_gain_sweep.py -- both originals untouched. This does not change
which checkpoint is used (still `outputs/models/diffusion_best.pt` by
default, per Item 1/2A's own convention) -- just centralizes the lookup.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ITEM1_CODE_DIR = (
    REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Code"
    / "stage2_tier0_item1_layerwise_magnitude_direction"
)
if str(ITEM1_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(ITEM1_CODE_DIR))

from utils import load_config, get_logger  # noqa: E402
from mentor_eval.checkpoint_utils import load_checkpoint  # noqa: E402


def load_model_checkpoint(cfg, ckpt_path: Path | None = None):
    """Returns the LoadedCheckpoint (or None if not found), from
    outputs/models/diffusion_best.pt by default -- the same checkpoint
    every Item 1/2A run has used."""
    path = ckpt_path if ckpt_path else Path(cfg.paths.outputs.models) / "diffusion_best.pt"
    return load_checkpoint(path, cfg)
