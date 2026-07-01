# Stage 1 Decisions

## 2026-07-02 — Architecture review complete; experiments not yet run

**Decision:** Do not attempt to "reproduce" numeric results from
`outputs/mentor_review/SUMMARY.md` or `outputs/sharma_inspired_analysis/SUMMARY.md`
as if they were a validated baseline. Both were explicitly written on a
machine with no trained checkpoint — most of their sections are marked
`_Blocked_`. There is no prior baseline_report with real generated-sample
metrics (accuracy/macro F1/collapse %) anywhere in this repo. Experiment 1
must establish a first real baseline, not "reproduce" one.

**Evidence:**
- `outputs/models/` contains only `.gitkeep` — no checkpoints on this machine.
- `logs/` contains only `.gitkeep` — no training log.
- `outputs/mentor_review/SUMMARY.md` sections 4, 6, 7, 8, and the generated-data
  half of section 9 are explicitly marked `_Blocked_`, needing
  `diffusion_best.pt` "on the GPU server, not this machine."
- `outputs/conditioning_analysis/cfg_sweep_result.txt` states CFG was never
  implemented in the model. This directly contradicts current
  `step04_transformer_diffusion.py`, which implements a null class token
  (`null_class_index = n_classes`), per-sample CFG dropout (`p_uncond`,
  default 0.10), and two-pass guided DDIM sampling
  (`GaussianDiffusion.ddim_sample`, `guidance_scale` param). Per the standing
  instruction to trust code over stale reports: **CFG is implemented in the
  current code**; the sweep report is stale (predates commits `3a2b035`
  through `231703f`) and must be regenerated, not cited as evidence CFG is
  missing.

**Also flagged (not yet a blocking decision):**
- `config.yaml`'s `device: auto` resolves to `"cuda" if torch.cuda.is_available() else "cpu"`
  (`step04_transformer_diffusion.py:681`) — it never checks `torch.backends.mps.is_available()`.
  On this machine (no CUDA, MPS available) this means **training silently
  runs on CPU** unless `device` is explicitly overridden. This materially
  affects the compute-time estimate for Experiment 1 and especially
  Experiment 2 (6 dataset sizes × training runs).

**Open question for the user before starting Experiment 1 (see chat):**
compute budget / time budget for training runs on this machine, and whether
to fix the MPS device-resolution gap before the first training run (a one-line
change, but it changes what "reproducing the baseline" method actually is).

## 2026-07-02 — Compute plan resolved; MPS bug fixed; Experiments 1-3 handed to GPU server

**Decision:** Fixed `_resolve_device()` in `step04_transformer_diffusion.py`
to check `torch.backends.mps.is_available()` (was cuda-or-cpu only) — a
correctness fix independent of the compute-source decision, harmless on
CUDA machines. Benchmarked this machine directly rather than guessing:
CPU training measured at 23,057.6 ms/step (batch=32) → ~29 days for a
200-epoch run; MPS OOMs on the very first step at batch=32 (8.43 GiB
requested vs. 9.07 GiB max on this 8GB-unified-memory M3). Asked the user
how to proceed; they confirmed a GPU server is available. Experiments 1
(baseline), 2 (dataset scaling), and 3 (directional probe) are fully coded
and documented under `Code/` but require that GPU server to execute —
their `Reports/*.md` are stubs pending real numbers, not fabricated
results.

**Decision:** Experiment 4 (MentorClassifier verification) needs no
diffusion checkpoint or GPU, so it was run directly on this machine rather
than waiting on the GPU-server round-trip. Result: AFIB does behave as an
out-of-distribution reject class, but only in a moderate-noise regime
(sigma≈0.25-0.5, 3.2-3.6x the chance rate of absorbing misclassifications);
at extreme noise the same role shifts to NSTEMI instead. See
`Reports/classifier_validation_report.md` for full evidence. **This means
AFIB-adjacent results in Experiments 1-3 (once run) must be interpreted
with this caveat already in hand, not treated as fresh conditioning
evidence.**
