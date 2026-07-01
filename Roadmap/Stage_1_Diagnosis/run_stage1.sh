#!/usr/bin/env bash
# Stage 1 — full diagnosis pipeline, single command.
#
# Runs Experiments 1, 1.5, 2 (with 2.5 curves built in), 3, 3.5, 4 (skipped
# if already complete), and 4.5, in dependency order, then assembles a
# results digest. Run this on the GPU server from the repository root.
#
# Env vars:
#   FORCE_RETRAIN=1   Re-run Experiment 1's training even if
#                     outputs/models/diffusion_best.pt already exists.
#
# Any extra arguments are passed through to Experiment 2
# (run_dataset_scaling.py), e.g.:
#   bash run_stage1.sh --epochs 100 --curve-every 20

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE1_CODE="Roadmap/Stage_1_Diagnosis/Code"
BEST_CKPT="outputs/models/diffusion_best.pt"

echo "================================================================"
echo " Stage 1 — Diagnosis pipeline"
echo "================================================================"

# ── Experiment 1: Baseline Reproduction ─────────────────────────────────────
if [ -f "$BEST_CKPT" ] && [ "${FORCE_RETRAIN:-0}" != "1" ]; then
  echo "[1/8] Experiment 1: $BEST_CKPT already exists — skipping training."
  echo "       Set FORCE_RETRAIN=1 to re-run from scratch."
else
  echo "[1/8] Experiment 1: Baseline Reproduction"
  bash "$STAGE1_CODE/Experiment_1_Baseline/run_experiment_1.sh"
fi

# ── Experiment 1.5: Checkpoint Verification ─────────────────────────────────
echo "[2/8] Experiment 1.5: Checkpoint Verification (conditioning vs epoch)"
python "$STAGE1_CODE/Experiment_1_Baseline/checkpoint_verification.py"

# ── Experiment 2 (+ 2.5): Dataset Scaling with training curves ─────────────
echo "[3/8] Experiment 2 (+2.5): Dataset Scaling"
python "$STAGE1_CODE/Experiment_2_Dataset_Scaling/run_dataset_scaling.py" "$@"

# ── Experiment 3: Directional Conditioning Analysis ─────────────────────────
echo "[4/8] Experiment 3: Directional Conditioning Analysis (baseline checkpoint)"
python "$STAGE1_CODE/Experiment_3_Directional_Probe/directional_conditioning_probe.py" \
    --ckpt "$BEST_CKPT" --tag baseline

for d in Roadmap/Stage_1_Diagnosis/Outputs/Experiment_2_Dataset_Scaling/checkpoints/size_*; do
  [ -d "$d" ] || continue
  size_tag="$(basename "$d")"
  echo "       Experiment 3: Directional Conditioning Analysis ($size_tag)"
  python "$STAGE1_CODE/Experiment_3_Directional_Probe/directional_conditioning_probe.py" \
      --ckpt "$d/diffusion_best.pt" --tag "$size_tag"
done

# ── Experiment 3.5: Layer-wise Direction Probe ──────────────────────────────
echo "[5/8] Experiment 3.5: Layer-wise Direction Probe"
python "$STAGE1_CODE/Experiment_3_Directional_Probe/layerwise_direction_probe.py" \
    --ckpt "$BEST_CKPT" --tag baseline

# ── Experiment 4: MentorClassifier Verification (skip if already complete) ──
EXP4_MARKER="Roadmap/Stage_1_Diagnosis/Outputs/Experiment_4_Classifier_Verification/noise_robustness.csv"
if [ -f "$EXP4_MARKER" ]; then
  echo "[6/8] Experiment 4: already complete ($EXP4_MARKER exists) — skipping."
else
  echo "[6/8] Experiment 4: MentorClassifier Verification"
  python "$STAGE1_CODE/Experiment_4_Classifier_Verification/classifier_verification.py"
fi

# ── Experiment 4.5: Feature Drift Visualization ─────────────────────────────
# Always re-run: it auto-detects the checkpoint and adds the generated-sample
# overlay once available, with no flags needed.
echo "[7/8] Experiment 4.5: Feature Drift Visualization"
python "$STAGE1_CODE/Experiment_4_Classifier_Verification/feature_drift_visualization.py"

# ── Collect + summarize ──────────────────────────────────────────────────────
echo "[8/8] Assembling Stage 1 results digest"
python "Roadmap/Stage_1_Diagnosis/collect_stage1_results.py"

echo "================================================================"
echo " Stage 1 pipeline complete."
echo " See Roadmap/Stage_1_Diagnosis/Reports/Stage1_Results_Digest.md"
echo " for the collected numbers — hand the whole Roadmap/Stage_1_Diagnosis/"
echo " tree back for the narrative reports and Stage1_Final_Report.md."
echo "================================================================"
