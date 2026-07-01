# Experiment 1 — Baseline Reproduction

## Why this experiment doesn't "reproduce" anything

There is no prior baseline_report anywhere in this repo with real
generated-sample accuracy/macro-F1/collapse% numbers — see
`Roadmap/Stage_1_Diagnosis/Decisions.md`. `outputs/mentor_review/SUMMARY.md`
marks every section that needs generated samples as `_Blocked_`. So this
experiment establishes the first real baseline for the current codebase,
using the current config.yaml (200 epochs, unchanged) — it is not checking
new numbers against old ones, except for the one real-data-only number that
does exist: MentorClassifier on real PTB-XL scored accuracy=0.844,
macro F1=0.743, macro AUC=0.958. That number should reproduce almost
exactly, since it doesn't depend on the diffusion model at all — if it
doesn't, something about the data pipeline changed and that must be
investigated before trusting anything downstream.

## How to run

```bash
bash Roadmap/Stage_1_Diagnosis/Code/Experiment_1_Baseline/run_experiment_1.sh
```

Requires: PTB-XL already downloaded/preprocessed (this repo's
`outputs/processed/*.npy` are already present and can be copied over rather
than regenerated, if convenient — they were built from the official
strat_fold split and are seed-independent).

## What to hand back

Copy these paths back (or point Claude at them if continuing on the GPU
server directly):

- `outputs/models/diffusion_best.pt`, `diffusion_architecture.json`
- `logs/diffusion_training_log.csv`
- `outputs/conditioning_analysis/*` (sensitivity_probe.csv, cfg_sweep_result.txt,
  cfg_sweep_metrics.csv, conditioning_confusion.csv, conditioning_heatmap.png,
  embedding_umap.png, embedding_tsne.png, embedding_features.csv)
- `outputs/mentor_review/classification_validation/*`
  (classifier_real_eval.json, classifier_generated_eval.json,
  confusion_matrix_real.png, confusion_matrix_generated.png)

Claude will write `baseline_report.md` (see the stub in
`Roadmap/Stage_1_Diagnosis/Reports/`) once these are available — do not
fill in numbers by hand; they'll be filled in during analysis to keep a
single source of truth for how each number was derived.

## STOP condition

If accuracy/macro F1/AUC on the **real-data** classifier stage differs
meaningfully from 0.844 / 0.743 / 0.958, stop and flag it — that stage
doesn't depend on the diffusion model at all, so a mismatch means the data
pipeline (fold split, preprocessing, or PTB-XL version) has drifted, and
nothing downstream should be trusted until that's resolved.
