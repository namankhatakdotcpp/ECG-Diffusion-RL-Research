#!/usr/bin/env bash
#
# run_reliability_sweep.sh -- runs the reliability-weight validation study
# (Roadmap/Stage_4_Optimization/Reward_Design_v2.md Section 4a; pre-
# registration draft in Roadmap/Stage_4_Optimization/Decisions.md) --
# arms 2-5 of the 5-point reliability_alpha sweep (100/75/50/25/0%),
# each a 250-iteration (Gate-3-scale) run of step07_rl_finetuning.py.
#
# Arm 1 (reliability_alpha=1.00) is NOT run by this script -- it is
# already on record as `gate3_250_fixed` (same seed, same 250 iterations,
# same outputs/models/diffusion_best.pt base checkpoint, same reward
# weights, use_reliability_scaling=true == reliability_alpha=1.00
# bit-identically per Step 2's identity-check test). `gate3_250` (the
# OTHER pre-existing 250-iteration run) is NOT equivalent and must not be
# confused with it -- it ran before trtr_classifier.pt/
# trtr_classifier_eval.json were fixed, so its HYP r_diag is a constant
# 0.5 fallback (classifier unavailable), not real per-class reliability
# data. See Decisions.md's "Reliability-weight validation study" section
# for the full comparison.
#
# Does NOT modify any Python source file. Only calls the existing,
# already-verified entrypoint:
#   step07_rl_finetuning.py --config <file> --n-iterations 250 --run-tag <tag>
#
# Usage (run inside tmux or screen -- multi-hour job, same convention as
# run_everything.sh):
#   tmux new -s reliability_sweep
#   bash run_reliability_sweep.sh
#
# Env vars (all optional):
#   ALLOW_NO_TMUX=1   Proceed even if not running inside tmux/screen (NOT
#                      recommended -- an SSH disconnect kills a bare shell
#                      mid-sweep).
#   N_ITERATIONS=250  Override iteration count per arm (default: 250,
#                      Gate-3 scale -- the pre-registered value in
#                      Decisions.md). Changing this invalidates the
#                      pre-registration's "same iteration count across
#                      arms" comparability -- only override for a smoke
#                      test of this script itself, e.g. N_ITERATIONS=1.
#
# Stops immediately (does not run remaining arms) if any arm exits
# non-zero -- a silently incomplete sweep would be worse than stopping
# and reporting exactly which arm failed.
#
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

N_ITERATIONS="${N_ITERATIONS:-250}"

# Pre-flight: this script does not select or activate a conda/venv
# environment for you -- it just calls `python`, whatever that resolves
# to on the caller's PATH. Fail fast with a clear message if the active
# Python is missing a dependency step07_rl_finetuning.py actually needs,
# rather than a bare ModuleNotFoundError surfacing partway into arm 1.
if ! python - <<'PY' 2>/dev/null
import torch, omegaconf, neurokit2, pywt
PY
then
    echo "ERROR: the active Python is missing required dependencies" >&2
    echo "(torch / omegaconf / neurokit2 / pywt)." >&2
    echo "This script does not select an environment for you --" >&2
    echo "activate the project's environment first (e.g. 'conda activate ecg')," >&2
    echo "then re-run." >&2
    exit 1
fi

if [ -z "${TMUX:-}" ] && [ -z "${STY:-}" ] && [ "${ALLOW_NO_TMUX:-0}" != "1" ]; then
    echo "ERROR: not running inside tmux or screen." >&2
    echo "This is a multi-hour job (4 arms x ${N_ITERATIONS} iterations) -- an SSH" >&2
    echo "disconnect will kill a bare shell." >&2
    echo "Run:  tmux new -s reliability_sweep   then re-invoke this script inside it," >&2
    echo "or set ALLOW_NO_TMUX=1 to proceed anyway (not recommended)." >&2
    exit 1
fi

# alpha : config file : run-tag -- locked convention, matches the table in
# Decisions.md's pre-registration section and Reward_Design_v2.md Section 4a.
# Arm 1 (alpha=1.00) deliberately excluded -- see header comment above.
ARMS=(
    "0.75:config_reliability_sweep_alpha0.75.yaml:reliability_alpha075"
    "0.50:config_reliability_sweep_alpha0.50.yaml:reliability_alpha050"
    "0.25:config_reliability_sweep_alpha0.25.yaml:reliability_alpha025"
    "0.00:config_reliability_sweep_alpha0.00.yaml:reliability_alpha000"
)

echo "run_reliability_sweep.sh -- ${#ARMS[@]} arms, ${N_ITERATIONS} iterations each"
echo "Repo root: ${REPO_ROOT}"
echo

for arm in "${ARMS[@]}"; do
    IFS=':' read -r ALPHA CONFIG_FILE RUN_TAG <<< "$arm"

    if [ ! -f "$CONFIG_FILE" ]; then
        echo "ERROR: config file not found: ${CONFIG_FILE}" >&2
        exit 1
    fi

    echo "================================================================"
    echo " Arm: reliability_alpha=${ALPHA}  (run-tag: ${RUN_TAG})"
    echo "================================================================"

    CMD=(python step07_rl_finetuning.py --config "$CONFIG_FILE" --n-iterations "$N_ITERATIONS" --run-tag "$RUN_TAG")
    echo "+ ${CMD[*]}"
    "${CMD[@]}"
    echo "Arm ${RUN_TAG} finished (exit 0)."

    ARCHIVE="${RUN_TAG}_results.tar.gz"
    echo "Packaging results -> ${ARCHIVE}"
    tar czf "$ARCHIVE" \
        "outputs/models/${RUN_TAG}" \
        "outputs/results/${RUN_TAG}" \
        "logs/${RUN_TAG}"
    echo "Archived: ${ARCHIVE}"
    echo
done

echo "run_reliability_sweep.sh completed successfully -- all ${#ARMS[@]} arms ran and were archived."
