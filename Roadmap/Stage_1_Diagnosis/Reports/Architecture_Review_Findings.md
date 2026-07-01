# Architecture Review Findings — Pre-Experiment Summary

Full detail lives in [`Architecture.md`](../../../Architecture.md) at the
project root (Task 2 of the master prompt). This file summarizes only the
findings that change how Stage 1 experiments must be run.

## Findings that block/reshape Experiment 1

1. **No baseline exists to reproduce.** `outputs/models/`, `logs/`, and
   `outputs/generated/` are empty on this machine. `outputs/mentor_review/SUMMARY.md`
   marks the generated-sample sections `_Blocked_` pending a checkpoint that
   "lives on the GPU server, not this machine." The only real number on
   record anywhere is a real-data-only classifier sanity check (acc=0.844,
   macro F1=0.743, macro AUC=0.958) — no generated-data accuracy/F1/collapse%
   has ever been recorded in this repo. Experiment 1 must **establish** a
   baseline, not verify one against a prior number.
2. **`cfg_sweep_result.txt` is stale and contradicts current code.** It
   states CFG isn't implemented; `step04_transformer_diffusion.py` in this
   repo has a null class token, CFG training dropout, and two-pass guided
   DDIM sampling. Per project rules ("if a report and the code disagree,
   trust the code and document the discrepancy") — trust the code. The
   sweep must be re-run once a checkpoint exists; do not cite the old
   `.txt` as evidence CFG is missing.
3. **`config.yaml`'s `device: auto` never checks MPS** — only `cuda` vs.
   `cpu` (`step04_transformer_diffusion.py:681`). On this Mac, that means
   silent CPU training. This is a real, current-code bug (not stale), and
   it changes the wall-clock cost of every Stage 1 training run.
4. **AFIB cannot be generated** (`class_mapping.py`:
   `MENTOR_TO_TRAINED_CLASS["AFIB"] = None` — merged into `OTHER` at step03
   due to only 103 records below `min_class_samples=200`). Every Stage 1
   report must treat AFIB results as "real-ECG-only" evidence, never as a
   generation-conditioning result.
5. **Sensitivity probe (`conditioning_sensitivity_probe.py`) measures
   magnitude only** (`||eps_A - eps_B||`), not semantic direction — this is
   the documented reason Experiment 3 (directional probe) must be built as
   new code rather than reusing the existing probe as-is.

## Decision required before running Experiment 1

See `Decisions.md` and the chat: user sign-off needed on (a) whether to fix
the MPS device-resolution gap before training, and (b) compute/time budget,
since no GPU (CUDA) is available and full 200-epoch training historically
took "hours" even with a GPU per the README.
