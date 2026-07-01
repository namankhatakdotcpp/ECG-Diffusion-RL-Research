# Experiment 1 — Baseline Reproduction

**Status: NOT YET RUN.** This machine cannot train the diffusion model
(CPU: ~23s/step, ~29 days for 200 epochs; MPS: OOMs at batch=32 on 8GB
unified memory — see `Roadmap/Stage_1_Diagnosis/Decisions.md`). Per the
user's decision, this experiment will be executed on their GPU server using
`Roadmap/Stage_1_Diagnosis/Code/Experiment_1_Baseline/run_experiment_1.sh`.
This file will be filled in once results are handed back — see that
script's README for exactly what to return.

## What this experiment will establish (not "reproduce" — see below)

No prior report in this repo contains real generated-sample metrics
(accuracy/macro F1/collapse %) for the diffusion model — see
`Architecture.md` §6 and `Decisions.md`. The only pre-existing number is a
real-data-only classifier sanity check: accuracy=0.844, macro F1=0.743,
macro AUC=0.958 (MentorClassifier trained+tested on real PTB-XL, no
generated data involved). That number is the one true "reproduction" check
in this experiment — see the STOP condition below.

## Planned sections (to be filled in after the GPU run)

- Training: final train/val loss, whether loss curves are stable (no
  divergence/collapse), wall-clock time, epochs actually completed.
- Generation: qualitative check that `diffusion_val_ep*.png` samples look
  like ECGs (not noise) for every class.
- Accuracy / macro F1 / collapse %: from
  `outputs/mentor_review/classification_validation/classifier_generated_eval.json`.
- CFG sweep: from `outputs/conditioning_analysis/cfg_sweep_result.txt` +
  `cfg_sweep_metrics.csv` — this time run against a checkpoint that
  actually has CFG training baked in (unlike the stale prior sweep, see
  `Architecture.md` §3).
- Sensitivity probe: from `outputs/conditioning_analysis/sensitivity_probe.csv`
  — magnitude-only class effect, is it nonzero and comparable across
  classes, or dominated by the time embedding?
- CFG routing verification: pass/fail verdict from `verify_cfg_routing.py`.

## STOP condition (per the master research protocol)

If the real-data classifier stage doesn't reproduce accuracy≈0.844,
macro F1≈0.743, macro AUC≈0.958 (this stage doesn't depend on the diffusion
model at all — it's pure real-data classification), **stop and investigate**
before trusting anything else in Stage 1. A mismatch there means the data
pipeline itself has drifted (fold split, preprocessing, or PTB-XL version),
not a conditioning problem.
