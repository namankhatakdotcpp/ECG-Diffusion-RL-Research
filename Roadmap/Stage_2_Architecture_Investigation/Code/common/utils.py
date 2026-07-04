"""
Stage 2 Tier 0 -- shared constants for the gain-sweep experiments (Item 2A
localized, Item 2B uniform, and any later item reusing the same grid).

LIFTED (copied, not moved) from
stage2_tier0_item2_localized_gain/item2_gain_sweep.py -- original
untouched. These values are locked by the pre-registration
(Tier0_Findings.md, Item 2 v3, commit e84c54c) and must not be edited
silently; any change is a new pre-registration revision, not a patch here.
"""

from __future__ import annotations

GAIN_GRID = [1.0, 1.25, 1.5, 2.0, 3.0, 5.0]  # locked, Item 2 v3 Sec. 7
TIMESTEPS = [100, 500, 900]  # matches Item 1's own baseline + sensitivity reruns
K_DRAWS = 20


def class_pairs(n_classes: int) -> list[tuple[int, int]]:
    return [(0, cls_b) for cls_b in range(1, n_classes)]
