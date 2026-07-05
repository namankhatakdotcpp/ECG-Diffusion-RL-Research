# Architecture.md — ECG Diffusion + RL Research Project

Status snapshot as of 2026-07-02. This document describes the project as the
code currently implements it, verified by reading source directly — not by
trusting prior markdown summaries. Where a prior report disagreed with the
code, the code wins and the discrepancy is called out explicitly.

**Not updated in place since 2026-07-02** -- Stage 2 and Stage 3 have since
added `mentor_eval/` (evaluation pipeline, not described below) and
`Roadmap/Stage_3_Architecture_Improvements/Code/stage3_candidates/`
(6 architecture variants built on top of, not replacing, `step04`'s
`ECGTransformerDiffusion` described here -- see `model_variants.py`'s
`ECGTransformerDiffusionVariant` and `Roadmap/Stage_3_Architecture_Improvements/Stage3_Status.md`
for what's current). The step01-09 pipeline description below is still
accurate; it just isn't the whole picture anymore.

## 1. Project structure

```
ECG/
├── config.yaml                       — single source of truth for all params/paths
├── step01_data_load_and_visualise.py — download/verify/visualise PTB-XL
├── step02_preprocessing.py           — filter, normalise, fold split, save arrays
├── step03_eda_and_class_mapping.py   — decide final classes, morphology/HRV stats
├── step04_transformer_diffusion.py   — ECGTransformerDiffusion model + training + CFG
├── step05_baseline_eval.py           — DTW/MMD/FED/Morph/TSTR — Table 1
├── step06_reward_function.py         — ClinicalReward: morph+hrv+real+diag
├── step07_rl_finetuning.py           — PPO/GRPO RL fine-tuning
├── step08_final_evaluation.py        — baseline vs RL head-to-head — Table 2 + figures
├── step09_ablation_study.py          — reward-component ablation — Table 3
├── mentor_eval/                      — diagnostic/eval layer, does not retrain the model
│   ├── class_mapping.py              — SCP code → class mappings (two schemes, see §4)
│   ├── conditioning_sensitivity_probe.py — forward-pass class-effect probe (magnitude only)
│   ├── conditioning_diagnostic.py    — generate→classify confusion table
│   ├── cfg_sweep.py                  — CFG guidance-scale sweep
│   ├── verify_cfg_routing.py         — checks CFG plumbing isn't dead/zero
│   ├── classification_validation.py  — MentorClassifier (train + eval)
│   ├── embedding_visualization.py    — UMAP/t-SNE of MentorClassifier features
│   ├── dataset_audit.py              — raw PTB-XL integrity check
│   └── ... (subband/zoomed/figures utilities)
├── utils/
│   ├── config.py  — load_config()
│   ├── metrics.py — dtw_distance, mmd_rbf, per_class_f1, tstr_score
│   ├── seed.py, logger.py, backup.py
├── data/ptbxl/     — raw PTB-XL (21,799 records)
├── outputs/processed/ — preprocessed splits (already built, see §2)
├── outputs/models/    — checkpoints (EMPTY on this machine, see §6)
└── logs/              — training logs (EMPTY on this machine, see §6)
```

## 2. Data pipeline

- **Source:** PTB-XL only (21,799 records, sampling_rate=100Hz, 12 leads,
  1000 samples/record = 10s). `dataset_audit.py` confirms 0 flagged/corrupt
  records.
- **Preprocessing** (`step02`): Butterworth bandpass 0.5–40Hz, baseline
  correction, z-score normalisation, clip to [-4, 4], official PTB-XL
  stratified folds (train=1-8, val=9, test=10).
- **Class scheme actually used for training** (from `step03`, stored in
  `outputs/processed/class_names.json`): **6 classes** —
  `NORM, MI, STTC, CD, HYP, OTHER`. AFIB (103 records) falls below
  `min_class_samples=200` and is folded into `OTHER`.
- **Already-processed split sizes on this machine** (`outputs/processed/*.npy`):
  train 17,418 / val 2,183 / test 2,198. Per-class train counts: NORM 7,386,
  MI 3,374, STTC 2,651, CD 2,630, HYP 1,036, OTHER 254.
- **Separate, incompatible 4-class "mentor" scheme** used only for
  evaluation/diagnosis (`mentor_eval/class_mapping.py`): `Normal, STEMI,
  NSTEMI, AFIB`, mapped from different SCP-code groupings. This is a review
  proxy, not what the diffusion model is trained on — see §4.

## 3. Model pipeline (`step04_transformer_diffusion.py`)

**`ECGTransformerDiffusion`** — patch-based transformer denoiser:
- `PatchEmbed1D`: each 12-lead, 1000-sample ECG → 600 tokens (12 leads × 50
  patches of size 20), each patch linearly projected to `model_dim=256`,
  plus a learnable per-lead embedding and a static sinusoidal position
  embedding.
- 6 pre-norm Transformer blocks (BERT-style), 8 heads, FFN dim 1024.
- Conditioning: sinusoidal timestep embedding + class embedding
  (`n_classes + 1` rows — the extra row is the CFG null/unconditional
  token, index `n_classes`). Two conditioning paths coexist:
  - `cond = t_emb + c_emb` summed and broadcast-added to every token
    (residual path, unchanged from an earlier PR).
  - `cond_film = concat(t_emb, c_emb)` fed into each block's `adaLN` layer,
    which produces FiLM-style shift/scale parameters for both the
    attention and FFN sub-layers (decoupled per commit `4d3f8cf`).
- Output: linear unprojection back to 12×1000, zero-initialised so training
  starts near-identity noise prediction.
- **Objective:** ε-prediction (DDPM), cosine beta schedule, T=1000.
- **Sampling:** DDIM, 50 deterministic steps (η=0). `ddim_sample` supports an
  optional `guidance_scale`: when set, it runs one batched forward pass with
  `[real_labels; null_labels]` concatenated and combines
  `eps = eps_uncond + scale * (eps_cond - eps_uncond)`.
- **CFG training:** per-sample Bernoulli dropout (`p_uncond`, config default
  0.10) replaces the true class label with the null index during training
  — this is what makes the null token meaningful at sampling time.
- **Optimiser:** AdamW, class_emb excluded from weight decay (asserted at
  code level, not just a config comment).
- **EMA:** decay 0.9999, with a context manager to swap in EMA weights for
  validation/generation.

**Discrepancy found:** `outputs/conditioning_analysis/cfg_sweep_result.txt`
states the model "was trained WITHOUT classifier-free guidance" (no null
token, `p_uncond=0`, no two-pass sampling). None of that matches the current
code above. That report was written before commits `3a2b035` (per-block
FiLM), `4d3f8cf` (decoupled AdaLN conditioning), `2d47374` (CFG training +
sampling), and `231703f` (CFG routing verification script). **The report is
stale and must not be treated as a current finding** — it describes a
version of the model that no longer exists in this repo. This is exactly the
kind of report/code conflict Stage 1 is supposed to catch.

## 4. Conditioning / diagnostic pipeline (`mentor_eval/`)

Layered, non-destructive analysis tools built on top of the diffusion model
and a separately-trained **MentorClassifier** (4 conv blocks, 12→32→64→128
channels, `AdaptiveAvgPool1d` → `Linear(128, 4)`, trained on real PTB-XL only,
30 epochs). Three distinct diagnostic layers exist, each answering a
different question:

1. **`conditioning_sensitivity_probe.py`** — cheapest, no generation needed.
   Fixes `x_t`/`t`, swaps only the class label, measures
   `||eps_A - eps_B||` normalised by overall epsilon scale, plus a
   same-class control (should be ≈0). Also inspects raw `class_emb` row
   norms vs. `time_mlp` weight magnitudes to check if class signal is
   drowned out in the `cond = t_emb + c_emb` sum. **This measures magnitude
   only** — a nonzero diff proves the label changes the output somehow, not
   that it changes it in the *correct semantic direction*. This is the gap
   Experiment 3 (directional probe) is designed to close.
2. **`conditioning_diagnostic.py`** — generates N=50 samples per diffusion
   class, classifies them with MentorClassifier, builds a
   diffusion-class × mentor-class confusion table. Diagonal = conditioning
   works; single dominant column = collapse. AFIB is unreachable here (see
   below).
3. **`embedding_visualization.py`** — extracts the 128-dim penultimate
   MentorClassifier feature (forward hook on the pooled conv output),
   projects with UMAP and t-SNE, and overlays real (circles) vs. generated
   (triangles) points per class. Visual proxy for "does generated data live
   in the correct manifold region."
4. **`cfg_sweep.py`** — sweeps `guidance_scale ∈ {None, 1, 2, 3, 5}`,
   re-running the classifier confusion + a "top-class collapse fraction"
   metric at each scale. Its last recorded output
   (`cfg_sweep_result.txt`) is stale (§3) and must be re-run against the
   current checkpoint once one exists.
5. **`verify_cfg_routing.py`** — sanity check independent of generation
   quality: is the CFG null-token path structurally "live" (output
   projection weights off zero-init, and conditioned vs. null passes
   actually differ by more than numerical noise)? Three-way verdict:
   routing live / output head still zero (too early in training) / routing
   bug (head live but no signal difference).

**Critical asymmetry — AFIB cannot be generated at all:**
`class_mapping.py`'s `MENTOR_TO_TRAINED_CLASS` maps `"AFIB": None` — the
diffusion model has no dedicated AFIB class (it was merged into `OTHER`
during step03 due to only 103 records). Every mentor-side evaluation that
loops over the 4 mentor classes explicitly skips AFIB generation and logs a
warning. This means **AFIB conditioning cannot be evaluated as "does the
model generate correct AFIB" — only whether real AFIB samples behave
strangely under the classifier** (this is exactly Experiment 4's OOD/reject-
class question, and it is answerable only on *real* AFIB ECGs, not
generated ones).

## 5. Training pipeline

- `step04`: diffusion training, 200 epochs configured, batch size 32,
  weighted random sampler (inverse class frequency) so rare classes are
  seen as often as common ones. Checkpoints every `save_every=25` epochs
  plus a running "best val loss" checkpoint. CSV log of
  epoch/step/train_loss/val_loss/lr/gpu_mem.
- `step06`/`step07`: clinical reward (morph 0.4 + hrv 0.4 + real 0.15 +
  diag 0.05) and PPO/GRPO RL fine-tuning on top of the frozen diffusion
  checkpoint — **out of scope for Stage 1**, which is diagnosis-only, but
  relevant context: RL fine-tuning depends entirely on a working baseline
  diffusion checkpoint existing first.

**Device resolution bug relevant to compute planning:**
`step04_transformer_diffusion.py:681`:
```python
def _resolve_device(cfg) -> str:
    return ("cuda" if torch.cuda.is_available() else "cpu") if str(cfg.device) == "auto" else str(cfg.device)
```
`config.yaml` sets `device: auto`. This function never checks
`torch.backends.mps.is_available()`. On this machine, `torch.cuda.is_available()
== False` and `torch.backends.mps.is_available() == True` — so with
`device: auto`, training silently runs on **CPU**, not the available Apple
Silicon GPU. This is a real, current-code finding (not a stale report) and
materially changes the time cost of every training run in Stage 1
Experiments 1 and 2.

## 6. Evaluation pipeline

- `step05_baseline_eval.py` (1162 lines): DTW nearest-neighbour distance,
  MMD (RBF kernel), FED (encoder-based Fréchet-style distance, trains a
  small CNN encoder), morphological validity (neurokit2-based), and TSTR
  (train-on-synthetic-test-on-real) — this is "Table 1" in the paper plan.
- `step08_final_evaluation.py` (789 lines): baseline vs. RL head-to-head,
  500 synthetic ECGs/class/seed, Wilcoxon significance tests, LaTeX table +
  figures 5–7 — downstream of RL, out of scope for Stage 1.
- `utils/metrics.py`: shared metric primitives (`dtw_distance`, `mmd_rbf`,
  `per_class_f1`, `tstr_score` = synthetic-macro-F1 / real-macro-F1 ratio).

**Current machine state — no baseline exists to reproduce:**
`outputs/models/`, `logs/`, and `outputs/generated/` contain only
`.gitkeep` placeholders. `outputs/mentor_review/SUMMARY.md` explicitly
marks sections 4 (real-vs-generated comparison), 6 (training progression),
7 (loss curves), 8 (similarity metrics), and the generated-data half of
section 9 (classification validation) as `_Blocked_`, each stating it needs
`diffusion_best.pt`, which "lives on the GPU server, not this machine." The
only concrete numbers that exist anywhere are a **real-data-only** sanity
check: accuracy=0.844, macro F1=0.743, macro AUC=0.958 for MentorClassifier
trained and tested on real PTB-XL (no generated data involved at all).

**Implication:** There is no previously-recorded baseline_report with
generated-sample accuracy/macro-F1/collapse% numbers anywhere in this repo.
Experiment 1 (Stage 1) cannot "reproduce" such numbers — it must establish
them for the first time on this machine, from the current code.

## 7. Current weaknesses (as of this review)

1. **No trained checkpoint or training log exists on this machine** — the
   entire mentor-review evaluation layer has been running against a state
   that doesn't exist locally; every "blocked" finding needs a real
   training run before it can be checked at all.
2. **Stale diagnostic report actively contradicts current code**
   (`cfg_sweep_result.txt` says CFG isn't implemented; it now is). Anyone
   reading that file without checking the code would draw the wrong
   conclusion about the state of the project.
3. **Device auto-resolution ignores MPS**, silently downgrading to CPU on
   this Mac and making "hours on GPU" estimates in the README inapplicable
   here without a config/code fix.
4. **AFIB is structurally unreachable by the generator** — any conditioning
   metric that implicitly assumes all mentor classes are generatable will
   silently exclude AFIB, which must be accounted for in every Stage 1
   report rather than treated as a data gap to "fix" by conditioning
   changes.
5. **Sensitivity probe measures only magnitude, not direction** — a
   nonzero class effect on `eps` does not establish that the effect points
   toward the correct class's data manifold. This is the stated reason
   Experiment 3 exists.
6. **Two non-interchangeable class taxonomies** (6-class trained scheme vs.
   4-class mentor review scheme) create a permanent mapping/exclusion step
   in every cross-pipeline evaluation — a persistent source of subtle bugs
   if a new experiment assumes the two are 1:1.
