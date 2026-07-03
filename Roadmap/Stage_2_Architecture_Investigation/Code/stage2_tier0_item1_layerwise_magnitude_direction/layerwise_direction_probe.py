"""
Stage 2 / Tier 0 Item 1 — Layer-wise Direction Probe.

DERIVED COPY, not an independent implementation. Originally written and
first executed in Stage 1 as Experiment 3.5
(Roadmap/Stage_1_Diagnosis/Code/Experiment_3_Directional_Probe/
layerwise_direction_probe.py), where it remains, unmodified from this
point forward, as the historical Stage 1 artifact. This copy lives here
so Stage 2's Tier 0 Item 1 has a self-contained Code/ subfolder per the
master prompt's deliverables spec, and so the `--timestep-frac` flag
(added specifically for Item 1's methodology review) is attached to a
script that is clearly Stage 2-owned rather than blurring Stage 1/Stage 2
ownership by editing the Stage 1 original in place.

OUT_DIR/FIG_DIR below still point at Stage 1's Outputs/Figures
directories, not Stage 2's -- intentional, not an oversight: all of this
experiment's artifacts (baseline t=500 + the t=100/t=900 sensitivity
reruns) were already produced there before this copy existed, and
splitting one experiment's outputs across two stage directories would be
worse than the ownership ambiguity this copy is meant to resolve. Tier 0
items 2-9, which have no Stage 1 lineage, write natively into Stage 2's
own Outputs/Figures.

Experiment 3's directional_conditioning_probe.py answers "does the FINAL
generated ECG move toward the correct class" using MentorClassifier's
embedding space. It can't say WHERE inside the diffusion model's own 6
Transformer blocks (config.diffusion.n_transformer_layers — this codebase's
backbone has 6 blocks, not 12) the class-conditioning signal appears,
strengthens, or gets washed out. This experiment answers that, entirely
inside the diffusion model's own residual stream — no MentorClassifier, no
DDIM sampling, no generation needed, so it is cheap (pure forward passes)
and can be run immediately after Experiment 1's checkpoint exists.

Method
------
Register a forward hook on every TransformerBlock in model.blocks, mean-
pooling each block's output tokens (B, 600, model_dim) -> (B, model_dim) to
get one "layer feature" vector per sample per layer.

For K independent random draws of the initial noise x_t (same fixed
timestep t per draw), run the model TWICE per draw — once with class label
A=0, once with class label B — and record each layer's feature both times.
This gives, per layer k and per draw i:

    delta_k^(i) = feat_k(x_t^(i), t, B) - feat_k(x_t^(i), t, A)

Two quantities per layer, aggregated over all K draws:

  magnitude_k    = mean_i( ||delta_k^(i)|| ) / mean_i( ||feat_k(x_t^(i), t, A)|| )
                   (normalized so layers with naturally larger activation
                   scale aren't automatically read as "more conditioned")

  direction_consistency_k = mean_i( cosine_similarity(delta_k^(i), mean_j(delta_k^(j))) )
                   in [-1, 1]. This is the key novel measurement: if the
                   class label produces a STABLE, repeatable direction of
                   change in a layer's representation regardless of the
                   random noise draw, consistency will be high (near 1).
                   If the layer's response to the class label is
                   essentially noise-shaped — magnitude might be nonzero,
                   but which direction it points changes randomly draw to
                   draw — consistency will be near 0. A layer with high
                   magnitude but low consistency is the layer where
                   conditioning has an effect but not a MEANINGFUL one; a
                   layer with declining consistency relative to earlier
                   layers is where a real conditioning signal starts being
                   diluted by the residual stream.

Repeated for every class pair (0, cls) for cls in 1..n_classes-1, then
averaged across pairs for the headline per-layer plot (per-pair detail kept
in the raw JSON).

Writes to Roadmap/Stage_1_Diagnosis/Outputs/Experiment_3_Directional_Probe/:
  layerwise_probe_{tag}.csv     — layer, magnitude, direction_consistency (averaged over class pairs)
  layerwise_probe_raw_{tag}.json — per-pair, per-layer detail
Writes to Roadmap/Stage_1_Diagnosis/Figures/Experiment_3_Directional_Probe/:
  layerwise_magnitude_{tag}.png
  layerwise_direction_consistency_{tag}.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from utils import load_config, get_logger
from mentor_eval.checkpoint_utils import load_checkpoint

OUT_DIR = REPO_ROOT / "Roadmap" / "Stage_1_Diagnosis" / "Outputs" / "Experiment_3_Directional_Probe"
FIG_DIR = REPO_ROOT / "Roadmap" / "Stage_1_Diagnosis" / "Figures" / "Experiment_3_Directional_Probe"
K_DRAWS = 20
PROBE_TIMESTEP_FRAC = 0.5  # probe at T/2 by default — mid-trajectory, where both time and class signal are active


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def _register_layer_hooks(model) -> tuple[list, dict]:
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def _make_hook(layer_idx: int):
        def _hook(module, inp, out):
            captured[layer_idx] = out.detach().mean(dim=1).cpu()  # (B, model_dim), mean-pooled over tokens
        return _hook

    for i, block in enumerate(model.blocks):
        handles.append(block.register_forward_hook(_make_hook(i)))
    return handles, captured


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--tag", type=str, default="baseline")
    parser.add_argument("--k-draws", type=int, default=K_DRAWS)
    parser.add_argument(
        "--timestep-frac", type=float, default=PROBE_TIMESTEP_FRAC,
        help=(
            "Fraction of T to probe at (default matches the original "
            "single-timestep behavior, PROBE_TIMESTEP_FRAC=0.5 i.e. t=500). "
            "Added for Stage 2 Tier 0 Item 1's methodology review -- confirms "
            "or refutes whether a layer-wise magnitude/direction finding is "
            "timestep-specific rather than assuming a single t=500 probe "
            "generalizes. Does not change default behavior."
        ),
    )
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("layerwise_direction_probe", cfg=cfg)
    torch.manual_seed(0)

    ckpt_path = Path(args.ckpt) if args.ckpt else Path(cfg.paths.outputs.models) / "diffusion_best.pt"
    loaded = load_checkpoint(ckpt_path, cfg)
    if loaded is None:
        print(f"[BLOCKED] Checkpoint not found at {ckpt_path}. Run Experiment 1 first.")
        return

    model = loaded.model
    device = loaded.device
    n_classes = loaded.n_classes
    n_layers = len(model.blocks)
    log.info(f"Model has {n_layers} TransformerBlocks. Probing at t={int(cfg.diffusion.T * args.timestep_frac)}.")

    handles, captured = _register_layer_hooks(model)

    n_leads = 12
    seq_len = int(cfg.ptbxl.signal_length)
    t_val = int(int(cfg.diffusion.T) * args.timestep_frac)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    raw = {"n_layers": n_layers, "probe_timestep": t_val, "k_draws": args.k_draws, "pairs": {}}
    per_layer_mag_all_pairs = np.zeros((n_classes - 1, n_layers))
    per_layer_cons_all_pairs = np.zeros((n_classes - 1, n_layers))

    with torch.no_grad():
        for pair_idx, cls_b in enumerate(range(1, n_classes)):
            deltas_by_layer = [[] for _ in range(n_layers)]
            base_norm_by_layer = [[] for _ in range(n_layers)]

            for draw in range(args.k_draws):
                torch.manual_seed(1000 + draw)
                x_t = torch.randn(1, n_leads, seq_len, device=device)
                t = torch.full((1,), t_val, device=device, dtype=torch.long)

                y_a = torch.full((1,), 0, device=device, dtype=torch.long)
                model(x_t, t, y_a)
                feat_a = {k: v.clone() for k, v in captured.items()}

                y_b = torch.full((1,), cls_b, device=device, dtype=torch.long)
                model(x_t, t, y_b)
                feat_b = {k: v.clone() for k, v in captured.items()}

                for layer in range(n_layers):
                    fa = feat_a[layer][0].numpy()
                    fb = feat_b[layer][0].numpy()
                    deltas_by_layer[layer].append(fb - fa)
                    base_norm_by_layer[layer].append(np.linalg.norm(fa))

            per_draw_mag_by_layer = []
            per_draw_cons_by_layer = []
            for layer in range(n_layers):
                deltas = np.stack(deltas_by_layer[layer])  # (K, model_dim)
                mean_delta = deltas.mean(axis=0)
                mean_base_norm = float(np.mean(base_norm_by_layer[layer]))
                magnitude = float(np.mean(np.linalg.norm(deltas, axis=1))) / (mean_base_norm + 1e-8)
                consistency = float(np.mean([cosine_sim(d, mean_delta) for d in deltas]))
                per_layer_mag_all_pairs[pair_idx, layer] = magnitude
                per_layer_cons_all_pairs[pair_idx, layer] = consistency
                # Per-draw values (not just the aggregated mean) -- needed for a real
                # within-pair variance estimate, raised explicitly in Stage 2 review.
                per_draw_norms = np.linalg.norm(deltas, axis=1) / (mean_base_norm + 1e-8)
                per_draw_cons = np.array([cosine_sim(d, mean_delta) for d in deltas])
                per_draw_mag_by_layer.append(per_draw_norms.tolist())
                per_draw_cons_by_layer.append(per_draw_cons.tolist())

            raw["pairs"][f"0->{cls_b}"] = {
                "magnitude_per_layer": per_layer_mag_all_pairs[pair_idx].tolist(),
                "direction_consistency_per_layer": per_layer_cons_all_pairs[pair_idx].tolist(),
                # Shape (n_layers, k_draws) -- one magnitude/consistency value per
                # individual noise draw, per layer. draws are independent Gaussian
                # noise seeds at FIXED (class-pair, timestep) -- not different real
                # ECG samples, since x_t here is pure noise, not real signal.
                "magnitude_per_layer_per_draw": per_draw_mag_by_layer,
                "direction_consistency_per_layer_per_draw": per_draw_cons_by_layer,
            }
            log.info(f"Pair (0 -> {cls_b}): magnitude={per_layer_mag_all_pairs[pair_idx].round(4).tolist()} "
                     f"consistency={per_layer_cons_all_pairs[pair_idx].round(4).tolist()}")

    for h in handles:
        h.remove()

    avg_mag = per_layer_mag_all_pairs.mean(axis=0)
    avg_cons = per_layer_cons_all_pairs.mean(axis=0)
    df = pd.DataFrame({
        "layer": list(range(1, n_layers + 1)),
        "magnitude": avg_mag,
        "direction_consistency": avg_cons,
    })
    df.to_csv(OUT_DIR / f"layerwise_probe_{args.tag}.csv", index=False)
    with open(OUT_DIR / f"layerwise_probe_raw_{args.tag}.json", "w") as f:
        json.dump(raw, f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["layer"], df["magnitude"], marker="o", color="steelblue")
    ax.set_xlabel("Transformer block index (1 = earliest)")
    ax.set_ylabel("Normalized conditioning magnitude")
    ax.set_title(f"Layer-wise conditioning magnitude ({args.tag})")
    ax.set_xticks(df["layer"])
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"layerwise_magnitude_{args.tag}.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["layer"], df["direction_consistency"], marker="o", color="crimson")
    ax.axhline(0.0, linestyle="--", color="gray", label="No consistent direction (noise-like)")
    ax.set_xlabel("Transformer block index (1 = earliest)")
    ax.set_ylabel("Direction consistency (cosine, avg. over noise draws)")
    ax.set_title(f"Layer-wise conditioning direction consistency ({args.tag})\n"
                 "High = stable class-conditioning direction; low = noise-like effect")
    ax.set_xticks(df["layer"])
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"layerwise_direction_consistency_{args.tag}.png", dpi=200)
    plt.close(fig)

    log.info(f"Done. See {OUT_DIR} and {FIG_DIR}")
    print(f"✓ Layer-wise direction probe complete (tag={args.tag}). See {OUT_DIR} and {FIG_DIR}")


if __name__ == "__main__":
    main()
