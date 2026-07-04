"""
Stage 2 / Tier 0 Item 6 -- Attention Entropy and Attention-Map Inspection.

Implements the locked pre-registration
(Roadmap/Stage_2_Architecture_Investigation/Reports/Item6_PreRegistration.md).
No cross-attention exists in this model (confirmed source read) -- this
inspects the existing self-attention's entropy under class conditioning
(injected only via adaLN modulation of attention's input, never as an
attention query/key/value itself).

H(head, query) = -sum_key(p * log(p + eps)), averaged over heads and
query positions per block, per class, per draw -- NOT computed from
attention weights pre-averaged across heads (a different, non-
interchangeable quantity, since entropy is non-linear).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common.io import load_config, get_logger, load_model_checkpoint  # noqa: E402
from common.hooks import register_attention_input_hooks  # noqa: E402
from common.plotting import plot_attention_entropy_vs_block  # noqa: E402
from common.utils import class_pairs, K_DRAWS, TIMESTEPS  # noqa: E402

OUT_DIR = (
    REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Outputs"
    / "stage2_tier0_item6_attention_entropy"
)
FIG_DIR = (
    REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Figures"
    / "stage2_tier0_item6_attention_entropy"
)
REPORT_DIR = REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Reports"

ENTROPY_DIFF_THRESHOLD = 0.05
EPS = 1e-12


def attention_entropy_per_block(model, attn_inputs: dict) -> list:
    """Replays block.attn(h,h,h, need_weights=True, average_attn_weights=False)
    for each block using the captured input, returns one entropy scalar
    per block (averaged over heads and query positions)."""
    entropies = []
    with torch.no_grad():
        for k, block in enumerate(model.blocks):
            h = attn_inputs[k]
            _, attn_weights = block.attn(
                h, h, h, need_weights=True, average_attn_weights=False
            )
            # attn_weights: (batch, num_heads, seq_len, seq_len) -- softmax over last dim (keys)
            p = attn_weights.clamp(min=EPS)
            H = -(p * p.log()).sum(dim=-1)  # (batch, num_heads, seq_len) -- entropy per (head, query)
            entropies.append(float(H.mean().item()))  # averaged over heads and query positions
    return entropies


def main() -> None:
    cfg = load_config()
    log = get_logger("item6_attention_entropy", cfg=cfg)
    torch.manual_seed(0)

    loaded = load_model_checkpoint(cfg)
    if loaded is None:
        print("[BLOCKED] Checkpoint not found. Run Experiment 1 first.")
        return

    model = loaded.model
    device = loaded.device
    n_classes = loaded.n_classes
    n_layers = len(model.blocks)
    n_leads = 12
    seq_len = int(cfg.ptbxl.signal_length)
    model.eval()  # inspection of frozen attention behavior, no dropout

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    pairs = class_pairs(n_classes)
    log.info(f"Item 6 sweep: pairs={pairs}, timesteps={TIMESTEPS}, k_draws={K_DRAWS}, "
             f"n_layers={n_layers}, model.eval() mode")

    entropy_a_cells = [[] for _ in range(n_layers)]
    entropy_b_cells = [[] for _ in range(n_layers)]
    raw = {"pairs": {}}

    for (y_a_val, y_b_val) in pairs:
        for t_val in TIMESTEPS:
            entropy_a_draws = [[] for _ in range(n_layers)]
            entropy_b_draws = [[] for _ in range(n_layers)]

            for draw in range(K_DRAWS):
                torch.manual_seed(1000 + draw)
                x_t = torch.randn(1, n_leads, seq_len, device=device)
                t = torch.full((1,), t_val, device=device, dtype=torch.long)
                y_a = torch.full((1,), y_a_val, device=device, dtype=torch.long)
                y_b = torch.full((1,), y_b_val, device=device, dtype=torch.long)

                handles, attn_inputs = register_attention_input_hooks(model)
                with torch.no_grad():
                    model(x_t, t, y_a)
                for h in handles:
                    h.remove()
                ent_a = attention_entropy_per_block(model, attn_inputs)

                handles, attn_inputs = register_attention_input_hooks(model)
                with torch.no_grad():
                    model(x_t, t, y_b)
                for h in handles:
                    h.remove()
                ent_b = attention_entropy_per_block(model, attn_inputs)

                for layer in range(n_layers):
                    entropy_a_draws[layer].append(ent_a[layer])
                    entropy_b_draws[layer].append(ent_b[layer])

            mean_a = [float(np.mean(v)) for v in entropy_a_draws]
            mean_b = [float(np.mean(v)) for v in entropy_b_draws]
            for layer in range(n_layers):
                entropy_a_cells[layer].append(mean_a[layer])
                entropy_b_cells[layer].append(mean_b[layer])

            raw["pairs"].setdefault(f"0->{y_b_val}", {})[str(t_val)] = {
                "entropy_class_A": mean_a, "entropy_class_B": mean_b,
            }
            log.info(f"Pair (0->{y_b_val}), t={t_val}: H(A)={[round(x,4) for x in mean_a]} "
                     f"H(B)={[round(x,4) for x in mean_b]}")

    entropy_a_pooled = [float(np.mean(v)) for v in entropy_a_cells]
    entropy_b_pooled = [float(np.mean(v)) for v in entropy_b_cells]
    entropy_diff_pooled = [abs(a - b) for a, b in zip(entropy_a_pooled, entropy_b_pooled)]
    max_entropy = float(np.log(seq_len))  # theoretical max entropy for a uniform 600-key distribution

    df = pd.DataFrame({
        "block": list(range(1, n_layers + 1)),
        "entropy_class_A": entropy_a_pooled,
        "entropy_class_B": entropy_b_pooled,
        "entropy_diff": entropy_diff_pooled,
    })
    df.to_csv(OUT_DIR / "attention_entropy.csv", index=False)
    fig_path = plot_attention_entropy_vs_block(df, FIG_DIR, threshold=ENTROPY_DIFF_THRESHOLD)

    with open(OUT_DIR / "attention_entropy_raw.json", "w") as f:
        json.dump({
            "n_layers": n_layers, "k_draws": K_DRAWS, "timesteps": TIMESTEPS,
            "max_theoretical_entropy_log_seqlen": max_entropy,
            "pooled_entropy_class_A": entropy_a_pooled,
            "pooled_entropy_class_B": entropy_b_pooled,
            "pooled_entropy_diff": entropy_diff_pooled,
            "pairs": raw["pairs"],
        }, f, indent=2)

    mean_diff = float(np.mean(entropy_diff_pooled))
    verdict = ("VERIFIED -- attention is class-blind (near-identical entropy across labels)"
               if mean_diff < ENTROPY_DIFF_THRESHOLD else
               "FALSIFIED -- attention entropy measurably differs by class label")

    print(json.dumps({
        "per_block": df.to_dict(orient="records"),
        "mean_entropy_diff_pooled": mean_diff,
        "max_theoretical_entropy": max_entropy,
        "threshold": ENTROPY_DIFF_THRESHOLD,
        "verdict": verdict,
    }, indent=2, default=str))
    print(f"\nMean |entropy diff| across blocks (pooled): {mean_diff:.5f} "
          f"(threshold {ENTROPY_DIFF_THRESHOLD}, max theoretical entropy {max_entropy:.4f}) -- {verdict}")
    print(f"Wrote: {fig_path}")


if __name__ == "__main__":
    main()
