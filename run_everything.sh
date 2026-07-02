#!/usr/bin/env bash
#
# run_everything.sh -- single-command, production-grade orchestration of
# the full ECG diffusion pipeline: Steps 1-4, the entire Stage 1
# diagnosis/evaluation pipeline, and the Stage 2 gates (Verification Gate
# + Repository Audit).
#
# Does NOT modify any Python source file. Only calls existing, already-
# reviewed entrypoints:
#   step01_data_load_and_visualise.py
#   step02_preprocessing.py
#   step03_eda_and_class_mapping.py
#   Roadmap/Stage_1_Diagnosis/Code/Experiment_1_Baseline/run_experiment_1_for_real.py
#     (Step 4 -- NOT step04_transformer_diffusion.py directly. This wrapper
#      is what wraps training in ExperimentLogger and asserts
#      n_train_records_actual > 10000 before proceeding -- see
#      Roadmap/Stage_0_Pipeline_Audit/Reports/Pipeline_Code_Audit.md
#      Finding 14 for why that distinction is load-bearing, not stylistic.)
#   Roadmap/Stage_1_Diagnosis/run_stage1.sh
#     (Experiments 1.5, 2+2.5, 3, 3.5, 4, 4.5, and the results digest --
#      its own internal Experiment-1 step detects the checkpoint Step 4
#      above just produced and skips retraining automatically.)
#   Roadmap/_infra/audit_repository.py
#
# The Stage 2.0 Verification Gate has no dedicated script (its full form
# requires human/Claude judgment to independently recompute headline
# numbers -- see Roadmap/Stage_2_Architecture_Investigation/Reports/
# Verification_Gate_Report.md for what that looks like). This script runs
# the MECHANICAL subset only -- ledger completeness, no flagged anomalies,
# every experiment status == success -- inline, via a python3 one-liner,
# rather than a new committed .py file. Treat a PASS here as "safe to
# proceed to a human/Claude review," not as a substitute for one.
#
# Usage (run inside tmux or screen -- see the check below):
#   tmux new -s ecg_run
#   bash run_everything.sh
#
# Env vars (all optional):
#   SKIP_ARCHIVE=1        Don't archive outputs/ and logs/ first (default:
#                          archive any existing content, then recreate
#                          clean folders).
#   ARCHIVE_ROOT=<path>   Where archived outputs/logs get moved to.
#                          Default: a sibling directory OUTSIDE this repo
#                          (../ecg_run_archives/<run_id>/) -- deliberately
#                          not inside the repo, so an archived multi-GB
#                          outputs/ tree can never be accidentally `git add`ed
#                          (outputs/ is gitignored by name; outputs_archive_*
#                          would not be, so archiving outside the repo avoids
#                          that footgun entirely rather than relying on a
#                          .gitignore edit this script deliberately avoids).
#   MIN_DISK_FREE_PCT=10  Disk-headroom gate threshold (percent free).
#   ALLOW_NO_TMUX=1       Proceed even if not running inside tmux/screen
#                          (NOT recommended -- this is a multi-hour job and
#                          an SSH disconnect will kill a bare shell).
#   REQUIRE_CUDA=1        Set to 0 to skip the hard CUDA requirement (NOT
#                          recommended -- see Roadmap/Stage_1_Diagnosis/
#                          Decisions.md: CPU training benchmarked at
#                          23s/step, ~29 days for a 200-epoch run).
#
# Any extra CLI arguments to this script are forwarded to
# Roadmap/Stage_1_Diagnosis/run_stage1.sh (and from there to Experiment 2's
# run_dataset_scaling.py), e.g.:
#   bash run_everything.sh --epochs 100
#
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="logs/run_everything_${RUN_ID}"
mkdir -p "$RUN_LOG_DIR"
RUN_LOG="${RUN_LOG_DIR}/run_everything.log"
SUMMARY_FILE="${RUN_LOG_DIR}/RUN_SUMMARY.md"

# Tee all stdout/stderr to the run log as well as the console (needed both
# for live viewing inside tmux and for a durable record afterward).
exec > >(tee -a "$RUN_LOG") 2>&1

STAGE_NAME="startup"
STAGE_NUM=0
declare -a COMPLETED_STAGES=()

_write_summary() {
    local status_line="$1"
    {
        echo "# run_everything.sh -- Run Summary"
        echo
        echo "Run ID: ${RUN_ID}"
        echo "Status: ${status_line}"
        echo "Started: ${RUN_START_TIME:-unknown}"
        echo "Log: ${RUN_LOG}"
        echo
        echo "## Stages completed"
        if [ "${#COMPLETED_STAGES[@]}" -eq 0 ]; then
            echo "_None yet._"
        else
            for s in "${COMPLETED_STAGES[@]}"; do
                echo "- [x] ${s}"
            done
        fi
    } > "$SUMMARY_FILE"
}

_on_error() {
    local exit_code=$?
    echo
    echo "!!! run_everything.sh FAILED during stage: ${STAGE_NAME} (exit code ${exit_code}) !!!"
    echo "!!! See ${RUN_LOG} for full output.                                                !!!"
    _write_summary "FAILED during: ${STAGE_NAME} (exit code ${exit_code})"
    exit "$exit_code"
}
trap _on_error ERR

_section() {
    STAGE_NUM=$((STAGE_NUM + 1))
    STAGE_NAME="$1"
    echo
    echo "================================================================"
    echo " [${STAGE_NUM}] $(date '+%Y-%m-%d %H:%M:%S') -- ${STAGE_NAME}"
    echo "================================================================"
}

_mark_done() {
    COMPLETED_STAGES+=("$1")
    _write_summary "IN PROGRESS (last completed: $1)"
}

RUN_START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"
_write_summary "STARTING"

echo "run_everything.sh -- run ${RUN_ID}"
echo "Repo root: ${REPO_ROOT}"
echo "Full log:  ${RUN_LOG}"

# ── Safety checks: tmux/screen, CUDA, disk space ────────────────────────
_section "Safety checks: tmux/screen, CUDA, disk space"

if [ -z "${TMUX:-}" ] && [ -z "${STY:-}" ] && [ "${ALLOW_NO_TMUX:-0}" != "1" ]; then
    echo "ERROR: not running inside tmux or screen." >&2
    echo "This is a multi-hour job -- an SSH disconnect will kill a bare shell." >&2
    echo "Run:  tmux new -s ecg_run   then re-invoke this script inside it," >&2
    echo "or set ALLOW_NO_TMUX=1 to proceed anyway (not recommended)." >&2
    exit 1
fi
if [ -n "${TMUX:-}" ]; then
    echo "tmux/screen check: OK (tmux)"
elif [ -n "${STY:-}" ]; then
    echo "tmux/screen check: OK (screen)"
else
    echo "tmux/screen check: OVERRIDDEN (ALLOW_NO_TMUX=1) -- not recommended"
fi

if [ "${REQUIRE_CUDA:-1}" == "1" ]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "ERROR: nvidia-smi not found -- no CUDA GPU visible on this machine." >&2
        echo "Set REQUIRE_CUDA=0 to override (not recommended -- see Roadmap/Stage_1_Diagnosis/Decisions.md)." >&2
        exit 1
    fi
    nvidia-smi
    CUDA_OK="$(python3 -c "import torch; print('1' if torch.cuda.is_available() else '0')")"
    if [ "$CUDA_OK" != "1" ]; then
        echo "ERROR: torch.cuda.is_available() == False." >&2
        echo "CPU training is not viable for this pipeline -- benchmarked at 23s/step," >&2
        echo "~29 days for a single 200-epoch run (Roadmap/Stage_1_Diagnosis/Decisions.md)." >&2
        echo "Set REQUIRE_CUDA=0 to override (not recommended)." >&2
        exit 1
    fi
    echo "CUDA check: OK ($(python3 -c "import torch; print(torch.cuda.get_device_name(0))"))"
else
    echo "CUDA check: SKIPPED (REQUIRE_CUDA=0) -- proceeding without a GPU guarantee"
fi

MIN_DISK_FREE_PCT="${MIN_DISK_FREE_PCT:-10}"
AVAIL_PCT="$(df --output=pcent . | tail -1 | tr -d '% ')"
FREE_PCT=$((100 - AVAIL_PCT))
echo "Disk free: ${FREE_PCT}% (threshold: ${MIN_DISK_FREE_PCT}%)"
if [ "$FREE_PCT" -lt "$MIN_DISK_FREE_PCT" ]; then
    echo "ERROR: disk free (${FREE_PCT}%) is below the ${MIN_DISK_FREE_PCT}% threshold." >&2
    echo "Free up space or lower MIN_DISK_FREE_PCT (not recommended) before proceeding." >&2
    exit 1
fi
_mark_done "Safety checks (tmux/screen, CUDA, disk space)"

# ── Archive old outputs, recreate clean folders ─────────────────────────
_section "Archive old outputs/logs, recreate clean folders"

if [ "${SKIP_ARCHIVE:-0}" == "1" ]; then
    echo "SKIP_ARCHIVE=1 -- leaving outputs/ and logs/ as-is (not archived, not recreated)."
else
    ARCHIVE_ROOT="${ARCHIVE_ROOT:-$(dirname "$REPO_ROOT")/ecg_run_archives/${RUN_ID}}"
    ARCHIVED_ANY=0
    for d in outputs logs; do
        if [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ]; then
            mkdir -p "$ARCHIVE_ROOT"
            echo "Archiving ${d}/ -> ${ARCHIVE_ROOT}/${d}/ (rename, zero-copy, outside the repo)"
            mv "$d" "${ARCHIVE_ROOT}/${d}"
            ARCHIVED_ANY=1
        fi
    done
    if [ "$ARCHIVED_ANY" -eq 0 ]; then
        echo "Nothing to archive (outputs/ and logs/ were already empty or absent)."
    else
        echo "Archived to: ${ARCHIVE_ROOT}"
    fi
fi

mkdir -p outputs/processed outputs/models outputs/generated outputs/results \
         outputs/conditioning_analysis outputs/mentor_review \
         logs "$RUN_LOG_DIR"
echo "Recreated: outputs/{processed,models,generated,results,conditioning_analysis,mentor_review}, logs/"
echo
echo "NOTE: Roadmap/ (including Roadmap/Stage_1_Diagnosis/Logs/ and the"
echo "ExperimentLogger ledgers under Roadmap/*/Reports/results_ledger.jsonl)"
echo "is never touched by this step. Those are the permanent, cumulative"
echo "research record, not regenerable pipeline output -- archiving them"
echo "would destroy audit history this project depends on."
_mark_done "Archived old outputs/logs, recreated clean folders"

# ── Steps 1-3: data load, preprocessing, class mapping ──────────────────
_section "Step 1 -- data load and visualise"
python3 step01_data_load_and_visualise.py
_mark_done "Step 1: data load and visualise"

_section "Step 2 -- preprocessing"
python3 step02_preprocessing.py
_mark_done "Step 2: preprocessing"

_section "Step 3 -- EDA and class mapping"
python3 step03_eda_and_class_mapping.py
_mark_done "Step 3: EDA and class mapping"

# ── Step 4: baseline diffusion training ─────────────────────────────────
_section "Step 4 -- baseline diffusion training (ledger-integrated, assertion-protected)"
python3 Roadmap/Stage_1_Diagnosis/Code/Experiment_1_Baseline/run_experiment_1_for_real.py
_mark_done "Step 4: baseline diffusion training (ledger entry: exp1_baseline_reproduction)"

# ── Entire Stage 1 evaluation pipeline ──────────────────────────────────
# run_stage1.sh's own Experiment-1 step detects outputs/models/diffusion_best.pt
# (just produced above) and skips retraining automatically -- proceeds
# straight to 1.5 / 2(+2.5) / 3 / 3.5 / 4 / 4.5 and the results digest.
_section "Entire Stage 1 diagnosis/evaluation pipeline"
bash Roadmap/Stage_1_Diagnosis/run_stage1.sh "$@"
_mark_done "Stage 1 evaluation pipeline (1.5, 2+2.5, 3, 3.5, 4, 4.5, results digest)"

# ── Stage 2.0 -- Verification Gate (mechanical subset) ──────────────────
_section "Stage 2.0 -- Verification Gate (mechanical checks)"
GATE_REPORT="Roadmap/Stage_2_Architecture_Investigation/Reports/Verification_Gate_Mechanical_Check.md"
mkdir -p "$(dirname "$GATE_REPORT")"
python3 -c "
import json, sys
from pathlib import Path
from datetime import datetime, timezone

ledger_path = Path('Roadmap/Stage_1_Diagnosis/Reports/results_ledger.jsonl')
master_log_path = Path('Roadmap/Stage_1_Diagnosis/Reports/MASTER_LOG.md')
report_path = Path('${GATE_REPORT}')

lines = []
def log(msg):
    print(msg)
    lines.append(msg)

ok = True
log(f'# Verification Gate -- Mechanical Check')
log(f'')
log(f'Run: ${RUN_ID}')
log(f'Timestamp: {datetime.now(timezone.utc).isoformat()}')
log(f'')
log('Mechanical subset only: ledger completeness, no MASTER_LOG anomalies,')
log('every experiment status == success. Does NOT independently recompute')
log('headline numbers -- that still requires human/Claude review, see')
log('Roadmap/Stage_2_Architecture_Investigation/Reports/Verification_Gate_Report.md')
log('for what that fuller review looks like.')
log('')

if not ledger_path.exists() or not master_log_path.exists():
    missing = ledger_path if not ledger_path.exists() else master_log_path
    log(f'## FAIL: missing {missing}')
    ok = False
else:
    records = [json.loads(l) for l in ledger_path.read_text().splitlines() if l.strip()]
    log(f'Ledger entries: {len(records)}')
    if not records:
        log('## FAIL: ledger exists but is empty')
        ok = False
    else:
        non_success = [r for r in records if r['status'] != 'success']
        log(f'Non-success entries: {len(non_success)}')
        for r in non_success:
            log(f\"  - {r['experiment_id']}: status={r['status']}\")

        master_log_text = master_log_path.read_text()
        if 'No anomalies detected' not in master_log_text:
            log('## FAIL: MASTER_LOG.md Flags section reports anomalies -- review before proceeding.')
            ok = False

        if non_success:
            log('## FAIL: one or more ledger entries did not succeed.')
            ok = False

        if ok:
            log('## PASS: all ledger entries succeeded, no MASTER_LOG anomalies flagged.')

report_path.write_text('\n'.join(lines) + '\n')
sys.exit(0 if ok else 1)
"
_mark_done "Stage 2.0 Verification Gate (mechanical checks passed)"

# ── Stage 2.0.5 -- Repository Audit ─────────────────────────────────────
_section "Stage 2.0.5 -- Repository Audit"
python3 Roadmap/_infra/audit_repository.py outputs/ \
    Roadmap/Stage_2_Architecture_Investigation/Reports/Repository_Audit_Report.md
_mark_done "Stage 2.0.5 Repository Audit (0 error-severity findings -- see report for warnings)"

# ── Done ─────────────────────────────────────────────────────────────────
_section "Done"
_write_summary "SUCCESS -- all stages completed"
echo "Run summary: ${SUMMARY_FILE}"
echo "Full log:    ${RUN_LOG}"
echo
echo "run_everything.sh completed successfully."
