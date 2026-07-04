"""
Stage 2 / Tier 0 Item 6 -- confidence-interval addendum.

Reads attention_entropy_raw.json (already produced by
item6_attention_entropy.py) and computes a 95% CI on entropy_diff per
block, across the n=15 pooled (pair, timestep) cells -- the same n=15
evidentiary unit Item 1 used for its own pooled statistics. No rerun of
the sweep; pure post-processing, per review that flagged the original
report's point-estimate-only claim as needing a variance check before
"blocks 5/6 exceed threshold" could be trusted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common.io import REPO_ROOT as _REPO_ROOT  # noqa: E402 (confirms common/ importable, unused otherwise)

OUT_DIR = (
    REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Outputs"
    / "stage2_tier0_item6_attention_entropy"
)


def main() -> None:
    with open(OUT_DIR / "attention_entropy_raw.json") as f:
        d = json.load(f)

    n_layers = d["n_layers"]
    cell_diffs = [[] for _ in range(n_layers)]
    for pair, tdict in d["pairs"].items():
        for t_val, vals in tdict.items():
            ea = vals["entropy_class_A"]
            eb = vals["entropy_class_B"]
            for layer in range(n_layers):
                cell_diffs[layer].append(abs(ea[layer] - eb[layer]))

    results = []
    for layer in range(n_layers):
        arr = np.array(cell_diffs[layer])
        mean = float(arr.mean())
        std = float(arr.std(ddof=1))
        se = std / np.sqrt(len(arr))
        ci95 = 1.96 * se
        results.append({
            "block": layer + 1, "n_cells": len(arr), "mean": mean, "std": std,
            "se": se, "ci95_low": mean - ci95, "ci95_high": mean + ci95,
            "crosses_threshold_0.05": bool((mean - ci95) < 0.05 < (mean + ci95)),
        })
        print(f"block {layer+1}: n={len(arr)} mean={mean:.4f} std={std:.4f} "
              f"SE={se:.4f} 95% CI=[{mean-ci95:.4f}, {mean+ci95:.4f}] "
              f"crosses_0.05={results[-1]['crosses_threshold_0.05']}")

    with open(OUT_DIR / "attention_entropy_confidence_intervals.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote: {OUT_DIR / 'attention_entropy_confidence_intervals.json'}")


if __name__ == "__main__":
    main()
