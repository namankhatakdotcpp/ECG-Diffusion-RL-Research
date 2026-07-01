#!/usr/bin/env bash
# Stage 1 / Experiment 1 — Baseline Reproduction
#
# Run this on the GPU server, from the repository root (same directory as
# config.yaml). It trains the current diffusion model exactly as configured
# (200 epochs, config.yaml unchanged) and runs every existing diagnostic that
# needs a trained checkpoint, in the order that lets each step build on the
# previous one's output.
#
# This machine (the one preparing this script) has no CUDA GPU and only 8GB
# of unified memory — training and generation could not be run or verified
# here. Nothing below has been executed; treat first-run output carefully
# and report anything that errors rather than silently skipping it.
#
# After this finishes, copy back to the laptop (or hand to Claude directly
# on the GPU server):
#   outputs/models/diffusion_best.pt
#   outputs/models/diffusion_architecture.json
#   logs/diffusion_training_log.csv
#   outputs/conditioning_analysis/   (sensitivity_probe.csv, cfg_sweep_result.txt,
#                                      cfg_sweep_metrics.csv, conditioning_confusion.csv,
#                                      embedding_*.png/csv)
#   outputs/mentor_review/classification_validation/  (classifier_*_eval.json, confusion_matrix_*.png)
#
# Deterministic seed: config.yaml's seeds[0] = 42 (set by step04's set_seed call).

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "=== [1/7] Train baseline diffusion model (step04) ==="
python step04_transformer_diffusion.py

echo "=== [2/7] Sensitivity probe (magnitude-only class effect) ==="
python -m mentor_eval.conditioning_sensitivity_probe

echo "=== [3/7] CFG routing verification (is the null-token path live?) ==="
python -m mentor_eval.verify_cfg_routing

echo "=== [4/7] Conditioning diagnostic (generate -> classify confusion table) ==="
python -m mentor_eval.conditioning_diagnostic

echo "=== [5/7] CFG guidance-scale sweep ==="
python -m mentor_eval.cfg_sweep

echo "=== [6/7] MentorClassifier validation (real + generated data) ==="
python -m mentor_eval.classification_validation

echo "=== [7/7] Embedding visualization (UMAP/t-SNE, real vs generated) ==="
python -m mentor_eval.embedding_visualization

echo "=== Regenerating mentor_eval SUMMARY.md from disk state ==="
python -m mentor_eval.run_all

echo "Done. See Roadmap/Stage_1_Diagnosis/Code/Experiment_1_Baseline/README.md for what to hand back."
