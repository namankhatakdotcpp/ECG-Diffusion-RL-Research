"""
mentor_eval/cfg_sweep.py — classifier-free guidance (CFG) scale sweep.

PRE-FLIGHT CHECK (run before training): this script inspects the diffusion
training code and reports whether CFG is supported. The result is written to
outputs/conditioning_analysis/cfg_sweep_result.txt before any generation is
attempted.

FINDING: CFG is NOT supported by the current architecture.

Evidence (step04_transformer_diffusion.py):
  - ECGTransformerDiffusion.forward() always takes a real class_label tensor
    and always calls self.class_emb(class_label) — no null/uncond token.
  - The training loop (train() function) always passes batch_cls to the model
    with p_uncond=0 (conditional dropout was never implemented).
  - There is no null-class embedding registered anywhere in the model.
  - GaussianDiffusion.ddim_sample() takes a single class_label tensor and
    passes it directly — no mechanism for a two-pass unconditional/conditional
    difference.

What CFG requires to work:
  1. A null (unconditional) class embedding, usually added as class index
     n_classes (one past the end of class_emb).
  2. Training with p_uncond (e.g. 0.1–0.2) probability of replacing
     class_label with the null index on each training step.
  3. Sampling with two forward passes per step:
       eps_uncond = model(x_t, t, null_label)
       eps_cond   = model(x_t, t, real_label)
       eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

None of these are present. Cranking a guidance_scale parameter would require
changes to ECGTransformerDiffusion, GaussianDiffusion.ddim_sample(), and
retraining from scratch with unconditional dropout.

This script does NOT fake CFG by substituting something else. The note above
is the complete output of this experiment.

Writes to: outputs/conditioning_analysis/cfg_sweep_result.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_config, get_logger
from utils.backup import snapshot_before_write

OUT_DIR = Path("outputs/conditioning_analysis")

_RESULT_TEXT = """\
CFG SWEEP — RESULT
==================

Status: NOT SUPPORTED — this experiment cannot be run without retraining.

Finding: The current ECGTransformerDiffusion model was trained WITHOUT
classifier-free guidance. Specifically:

  1. No unconditional/null class token exists in the model.
  2. Training never dropped class labels (p_uncond = 0 throughout).
  3. ddim_sample() has no two-pass CFG logic.

What this means for conditioning diagnosis:
  CFG is a cheap sampling-time fix only when the model was trained with it.
  Since it wasn't, a guidance-scale sweep cannot improve conditioning without
  retraining. The 10-minute experiment recommended in the diagnosis plan is
  not available here.

What would be required to enable CFG:
  1. Add a null embedding: expand class_emb to (n_classes + 1) tokens; index
     n_classes is the unconditional token.
  2. Retrain with p_uncond ≈ 0.10–0.15: randomly replace class_label with
     n_classes during training.
  3. Update ddim_sample() to run two forward passes per step and combine
     with the guidance formula:
       eps_guided = eps_uncond + scale * (eps_cond - eps_uncond)
  4. Grid-search guidance_scale ∈ {1.5, 2.0, 3.0, 5.0, 7.5} after
     retraining.

Recommendation:
  Use the UMAP/t-SNE embedding visualization (embedding_visualization.py)
  and the conditioning diagnostic (conditioning_diagnostic.py) to first
  confirm the severity of conditioning collapse. If the collapse is confirmed,
  retrain with CFG dropout and then run the scale sweep. Do not spend time
  on ST-segment or T-wave metrics until conditioning is verified working.
"""


def main() -> None:
    cfg = load_config()
    log = get_logger("cfg_sweep", cfg=cfg)

    snapshot_before_write(OUT_DIR)
    out_path = OUT_DIR / "cfg_sweep_result.txt"
    out_path.write_text(_RESULT_TEXT)
    log.info(f"CFG sweep result written → {out_path}")
    print(_RESULT_TEXT)
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
