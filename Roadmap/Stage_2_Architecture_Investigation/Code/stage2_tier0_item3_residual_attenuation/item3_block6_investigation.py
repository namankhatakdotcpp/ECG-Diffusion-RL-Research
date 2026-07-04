"""
Stage 2 / Tier 0 Item 3 -- follow-up investigation into the block 6 spike
observed in the main sweep (R_k combined pooled: block 3 valley = 0.0947,
block 6 spike = 0.7616, ~8x larger). Does NOT lock decision criteria,
write Item3_Report.md, or commit -- diagnostic only, per explicit
instruction.

Confirmed by source read before running anything
(step04_transformer_diffusion.py:276-279): `tokens = self.final_norm(tokens)`
then `out = self.unproj(tokens)` -- FinalNorm's output IS unproj's input,
the same tensor, no transform between them. So "FinalNorm output norm"
and "unpatchify input norm" are trivially identical by construction, not
two measurements -- only one new hook (on `model.final_norm`) is needed,
not two.

Reuses the exact same 5x3x20 design as the main sweep (not a new,
independently-chosen design) so these numbers are pooled consistently
with the already-reported R_k table. One new capture point
(`model.final_norm`) added to the existing register_layer_hooks +
register_block0_input_hook combination -- everything else unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parents[1]  # Roadmap/.../Code/
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common.io import REPO_ROOT, load_config, get_logger, load_model_checkpoint  # noqa: E402
from common.hooks import register_layer_hooks, register_block0_input_hook  # noqa: E402
from common.utils import class_pairs, K_DRAWS, TIMESTEPS  # noqa: E402

OUT_DIR = (
    REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Outputs"
    / "stage2_tier0_item3_residual_attenuation"
)

VALLEY_BLOCK_IDX = 2  # block 3 (1-indexed) -- the observed minimum, 0-indexed


def register_final_norm_hook(model):
    captured = {}

    def _hook(module, inp, out):
        captured["final_norm_output"] = out.detach().mean(dim=1).cpu()

    handle = model.final_norm.register_forward_hook(_hook)
    return handle, captured


def run_single_pass(model, x_t, t, y):
    handles, out_captured = register_layer_hooks(model)
    pre_handle, in0_captured = register_block0_input_hook(model)
    fn_handle, fn_captured = register_final_norm_hook(model)
    try:
        with torch.no_grad():
            model(x_t, t, y)
        block_outputs = {k: v.clone() for k, v in out_captured.items()}
        block0_input = in0_captured["block0_input"].clone()
        final_norm_output = fn_captured["final_norm_output"].clone()
    finally:
        fn_handle.remove()
        pre_handle.remove()
        for h in handles:
            h.remove()

    n_layers = len(block_outputs)
    block_inputs = {0: block0_input}
    for k in range(1, n_layers):
        block_inputs[k] = block_outputs[k - 1]
    return block_inputs, block_outputs, final_norm_output


def main() -> None:
    cfg = load_config()
    log = get_logger("item3_block6_investigation", cfg=cfg)
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

    pairs = class_pairs(n_classes)
    log.info(f"Block-6 investigation sweep: pairs={pairs}, timesteps={TIMESTEPS}, "
             f"k_draws={K_DRAWS} (same design as main sweep, reused for pooling consistency)")

    n_layers_idx6 = n_layers - 1  # block 6, 0-indexed

    # Norms collected per draw, per class, pooled at the end -- same pooling convention as main sweep.
    block6_in_norms, block6_out_norms, block6_residual_norms = [], [], []
    final_norm_out_norms = []
    valley_in_norms, valley_out_norms, valley_residual_norms = [], [], []

    for (y_a_val, y_b_val) in pairs:
        for t_val in TIMESTEPS:
            for draw in range(K_DRAWS):
                torch.manual_seed(1000 + draw)
                x_t = torch.randn(1, n_leads, seq_len, device=device)
                t = torch.full((1,), t_val, device=device, dtype=torch.long)
                y_a = torch.full((1,), y_a_val, device=device, dtype=torch.long)
                y_b = torch.full((1,), y_b_val, device=device, dtype=torch.long)

                for y in (y_a, y_b):
                    block_in, block_out, final_norm_out = run_single_pass(model, x_t, t, y)

                    b6_in = block_in[n_layers_idx6][0].numpy()
                    b6_out = block_out[n_layers_idx6][0].numpy()
                    block6_in_norms.append(np.linalg.norm(b6_in))
                    block6_out_norms.append(np.linalg.norm(b6_out))
                    block6_residual_norms.append(np.linalg.norm(b6_out - b6_in))
                    final_norm_out_norms.append(np.linalg.norm(final_norm_out[0].numpy()))

                    v_in = block_in[VALLEY_BLOCK_IDX][0].numpy()
                    v_out = block_out[VALLEY_BLOCK_IDX][0].numpy()
                    valley_in_norms.append(np.linalg.norm(v_in))
                    valley_out_norms.append(np.linalg.norm(v_out))
                    valley_residual_norms.append(np.linalg.norm(v_out - v_in))

    block6_out_mean = float(np.mean(block6_out_norms))
    final_norm_out_mean = float(np.mean(final_norm_out_norms))
    valley_out_mean = float(np.mean(valley_out_norms))

    # Ratio: FinalNorm output norm / block6 output norm (compression at block 6)
    ratio_block6 = final_norm_out_mean / block6_out_mean
    # Comparison baseline: what does LayerNorm-equivalent normalization do to the valley block's
    # OWN output norm, relative to itself run through final_norm (a hypothetical "if this were
    # the last block" comparison) -- NOT literally applicable (final_norm only runs after block 6
    # in the real architecture), so instead report the valley block's own residual/output ratio
    # as the same-pipeline comparison point requested: how large is block 3's residual update
    # relative to its own output, vs. block 6's residual update relative to its own output.
    block6_residual_to_output = float(np.mean(block6_residual_norms)) / block6_out_mean
    valley_residual_to_output = float(np.mean(valley_residual_norms)) / valley_out_mean

    result = {
        "block6_input_norm_mean": float(np.mean(block6_in_norms)),
        "block6_output_norm_mean": block6_out_mean,
        "block6_residual_update_norm_mean": float(np.mean(block6_residual_norms)),
        "final_norm_output_norm_mean": final_norm_out_mean,
        "unpatchify_input_norm_mean": final_norm_out_mean,  # identical by construction, source-confirmed
        "valley_block_index_1indexed": VALLEY_BLOCK_IDX + 1,
        "valley_input_norm_mean": float(np.mean(valley_in_norms)),
        "valley_output_norm_mean": valley_out_mean,
        "valley_residual_update_norm_mean": float(np.mean(valley_residual_norms)),
        "ratio_final_norm_output_to_block6_output": ratio_block6,
        "ratio_block6_residual_to_block6_output": block6_residual_to_output,
        "ratio_valley_residual_to_valley_output": valley_residual_to_output,
    }

    with open(OUT_DIR / "block6_investigation.json", "w") as f:
        json.dump(result, f, indent=2)

    log.info(json.dumps(result, indent=2))

    # Outcome classification (a/b/c), per the requested three-way check.
    if block6_residual_to_output <= valley_residual_to_output:
        outcome = "(a) block 6 is not architecturally special once normalized -- its residual " \
                  "update is not larger, relative to its own output, than the valley block's"
    elif ratio_block6 < 1.0:
        outcome = "(b) block 6's output is compressed by FinalNorm, but its residual contribution " \
                  "(relative to its own output) is still larger than the valley block's -- it " \
                  "matters more even after compression"
    else:
        outcome = "(c) block 6 is barely compressed by FinalNorm -- the final block genuinely " \
                  "performs the bulk of late-stage representation refinement"

    print(json.dumps(result, indent=2))
    print(f"\nOUTCOME CLASSIFICATION: {outcome}")


if __name__ == "__main__":
    main()
