"""
mentor_eval/conditioning_sensitivity_probe.py — retrain-free test of whether
the diffusion model's predicted noise actually changes when the class label
changes, holding x_t and t fixed.

Method:
  Fix a batch of random x_t and a timestep t. Pass the same (x_t, t) through
  the model with class_label = class_A vs class_B. Measure ||eps_A - eps_B||
  normalized by eps scale. If conditioning is working, this ratio should be
  meaningfully above 0. If it's near 0, the class embedding has no effect on
  the output regardless of training.

  Also measures control_diff: the norm difference from passing the SAME class
  twice. This should be exactly 0 (model is deterministic in eval mode) —
  confirms x_t identicalness and serves as a numerical zero reference.

  Finally inspects the class_emb.weight row norms and time_mlp weight norms
  to diagnose whether the class embedding magnitude is anomalously small
  relative to the time embedding, which would explain why conditioning is
  overwhelmed.

Writes: outputs/conditioning_analysis/sensitivity_probe.csv

Blocked gracefully (prints [BLOCKED] message, exits 0) if checkpoint is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time
from utils import load_config, get_logger
from mentor_eval.checkpoint_utils import load_checkpoint

OUT_DIR = Path("outputs/conditioning_analysis")


def main() -> None:
    cfg = load_config()
    log = get_logger("conditioning_sensitivity_probe", cfg=cfg)
    torch.manual_seed(0)

    # Same checkpoint path used by all other mentor_eval scripts
    ckpt_path = Path(cfg.paths.outputs.models) / "diffusion_best.pt"
    loaded = load_checkpoint(ckpt_path, cfg)
    if loaded is None:
        print(
            f"\n[BLOCKED] Checkpoint not found at {ckpt_path}.\n"
            "  Re-run on GPU server after step04 training completes.\n"
        )
        return

    model   = loaded.model   # already in eval mode from load_checkpoint
    classes = loaded.class_names
    class_to_idx = {c: i for i, c in enumerate(classes)}
    device  = loaded.device

    # n_leads is hardcoded 12 in ECGTransformerDiffusion.__init__ (not a config field)
    n_leads = 12
    # seq_len from config — same field used in step04_transformer_diffusion.py:185
    seq_len = int(cfg.ptbxl.signal_length)
    batch   = 8

    x_t = torch.randn(batch, n_leads, seq_len, device=device)

    T = int(cfg.diffusion.T)
    probe_timesteps = [T - 1, int(T * 0.75), int(T * 0.5), int(T * 0.25), 0]

    log.info(f"Loaded checkpoint: epoch={loaded.epoch}  classes={classes}")
    log.info(f"Probe shape: x_t={tuple(x_t.shape)}  timesteps={probe_timesteps}")
    log.info(f"T={T}  n_leads={n_leads}  seq_len={seq_len}")

    results = []
    with torch.no_grad():
        for t_val in probe_timesteps:
            t   = torch.full((batch,), t_val, device=device, dtype=torch.long)
            y_a = torch.full((batch,), class_to_idx[classes[0]], device=device, dtype=torch.long)

            # Two identical passes with the same class — difference must be 0
            eps_a1 = model(x_t, t, y_a)
            eps_a2 = model(x_t, t, y_a)
            control_diff = (eps_a1 - eps_a2).flatten(1).norm(dim=1).mean().item()
            eps_scale    = eps_a1.flatten(1).norm(dim=1).mean().item()

            row = {"t": t_val, "control_diff": control_diff, "eps_scale": eps_scale}

            for cls in classes[1:]:
                y_b   = torch.full((batch,), class_to_idx[cls], device=device, dtype=torch.long)
                eps_b = model(x_t, t, y_b)
                diff  = (eps_a1 - eps_b).flatten(1).norm(dim=1).mean().item()
                row[f"{classes[0]}_vs_{cls}_norm"] = diff / (eps_scale + 1e-8)

            results.append(row)
            log.info(
                f"t={t_val:4d}: " +
                ", ".join(f"{k}={v:.6f}" for k, v in row.items() if k != "t")
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "sensitivity_probe.csv"
    if csv_path.exists():
        old_ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(csv_path.stat().st_mtime))
        csv_path.rename(OUT_DIR / f"sensitivity_probe_{old_ts}.csv")
    pd.DataFrame(results).to_csv(csv_path, index=False)
    log.info(f"Saved → {csv_path}")

    # ── Class embedding vs time MLP weight magnitudes ──────────────────────────
    log.info("=== class embedding magnitude vs. time MLP weights ===")
    class_emb_param = None
    time_mlp_norms  = []

    for name, p in model.named_parameters():
        ln = name.lower()
        if "class_emb" in ln:
            # class_emb.weight: (n_classes, model_dim)
            class_emb_param = (name, p)
        elif "time_mlp" in ln and p.dim() > 1:
            # time_mlp linear weight matrices
            time_mlp_norms.append((name, p.detach().norm(dim=-1).mean().item()))

    if class_emb_param:
        name, p = class_emb_param
        row_norms = p.detach().norm(dim=-1)   # (n_classes,) — one norm per class
        log.info(
            f"{name}: per-class embedding norms = "
            f"{[round(x, 4) for x in row_norms.tolist()]}  "
            f"mean = {row_norms.mean().item():.4f}  "
            f"std  = {row_norms.std().item():.4f}"
        )
    else:
        log.warning("class_emb.weight not found in named_parameters() — check model definition.")

    if time_mlp_norms:
        for name, norm in time_mlp_norms:
            log.info(f"{name}: mean row norm = {norm:.4f}")
    else:
        log.warning("No time_mlp parameters found — check model definition.")

    # Diagnostic interpretation hint
    log.info("=== interpretation ===")
    log.info(
        "If all {cls}_vs_{other}_norm values are near 0, the class embedding "
        "has no measurable effect on the output — conditioning is collapsed "
        "at the weight level (not a sampling-time issue)."
    )
    log.info(
        "If class_emb per-class norms are much smaller than time_mlp row norms, "
        "the class signal is being drowned out by the time embedding in the "
        "'cond = t_emb + c_emb' sum."
    )


if __name__ == "__main__":
    main()
