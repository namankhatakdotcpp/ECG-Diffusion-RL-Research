"""
Stage 2 / Tier 0 Item 4 -- bootstrap CI on the percentile-rank estimate.

Reads gradient_probe_raw.json (already produced by item4_gradient_probe.py
on the GPU) -- pure post-processing, no rerun. Resamples the N_DRAWS
per-tensor gradient-norm draws WITH REPLACEMENT >=1000 times, recomputes
class_emb.weight's percentile rank among the other tensors each time,
and reports the point estimate, bootstrap SD, and 95% CI.

This directly answers the question the raw percentile_rank number alone
cannot: how much would this rank estimate move under fresh sampling
noise, given only N_DRAWS=30 independent batches went into it.
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

OUT_DIR = (
    REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Outputs"
    / "stage2_tier0_item4_gradient_competitiveness"
)

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42


def bootstrap_percentile_rank(primary_grad_norms: dict, n_bootstrap: int = N_BOOTSTRAP) -> dict:
    """primary_grad_norms: {tensor_name: [g_draw0, g_draw1, ...]} -- same
    structure as gradient_probe_raw.json's primary_grad_norms_per_draw."""
    names = sorted(primary_grad_norms.keys())
    n_draws = len(primary_grad_norms["class_emb.weight"])
    rng = np.random.RandomState(BOOTSTRAP_SEED)

    # Stack into (n_draws, n_tensors) so a single resampled draw-index set
    # applies consistently across all tensors (preserves any cross-tensor
    # correlation within a draw, e.g. a batch that's globally "easy").
    matrix = np.array([primary_grad_norms[name] for name in names]).T  # (n_draws, n_tensors)
    class_emb_idx = names.index("class_emb.weight")
    other_idx = [i for i in range(len(names)) if i != class_emb_idx]

    ranks = []
    for _ in range(n_bootstrap):
        sample_idx = rng.randint(0, n_draws, size=n_draws)
        resampled = matrix[sample_idx]  # (n_draws, n_tensors)
        pooled_mean = resampled.mean(axis=0)  # (n_tensors,)
        class_emb_mean = pooled_mean[class_emb_idx]
        other_means = pooled_mean[other_idx]
        rank = float(np.mean(other_means < class_emb_mean)) * 100.0
        ranks.append(rank)

    ranks = np.array(ranks)
    point_estimate_mean = float(matrix.mean(axis=0)[class_emb_idx])
    return {
        "n_draws": n_draws,
        "n_bootstrap": n_bootstrap,
        "point_estimate_percentile_rank": float(
            100.0 * np.mean(matrix.mean(axis=0)[other_idx] < point_estimate_mean)
        ),
        "bootstrap_mean": float(ranks.mean()),
        "bootstrap_sd": float(ranks.std(ddof=1)),
        "bootstrap_ci95_low": float(np.percentile(ranks, 2.5)),
        "bootstrap_ci95_high": float(np.percentile(ranks, 97.5)),
    }


def main() -> None:
    raw_path = OUT_DIR / "gradient_probe_raw.json"
    if not raw_path.exists():
        print(f"[BLOCKED] {raw_path} not found -- run item4_gradient_probe.py on the GPU "
              f"and transfer gradient_probe_raw.json here first. Cannot bootstrap without "
              f"the actual per-draw data.")
        return

    with open(raw_path) as f:
        raw = json.load(f)

    result = bootstrap_percentile_rank(raw["primary_grad_norms_per_draw"])
    print(json.dumps(result, indent=2))

    with open(OUT_DIR / "gradient_probe_bootstrap_ci.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote: {OUT_DIR / 'gradient_probe_bootstrap_ci.json'}")


if __name__ == "__main__":
    main()
