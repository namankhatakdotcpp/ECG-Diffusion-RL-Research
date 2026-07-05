"""
Stage 3 -- minimal sequential queue runner (NOT a scheduler/dashboard,
per the roadmap's own infrastructure-budget principle). For each
candidate ID given: train -> evaluate via the EXISTING mentor_eval
pipeline (same code path as the baseline-comparison protocol,
Stage3_Roadmap.md Sec. 6 -- classification_validation.py and
similarity_metrics.py both already support a --ckpt override, reused
here, not reimplemented) -> write metadata.json -> link results under
Results/<run_id>/.

This script does NOT itself decide gate outcomes -- see
evaluate_wave_gate() below, which is separate so it can be called
non-interactively once real metrics exist, without another chat
check-in (per the "automated, not chat-gated" directive).

Usage:
    python run_stage3_queue.py S3-002 S3-003          # train + evaluate, in order given
    python run_stage3_queue.py --gate S3-002 S3-003   # only evaluate the gate (assumes
                                                       # both already trained + evaluated)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

STAGE2_CODE_DIR = REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Code"
if str(STAGE2_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_CODE_DIR))

from common.io import load_config, get_logger  # noqa: E402
from common_train import train_variant, RESULTS_ROOT  # noqa: E402
from stage3_metadata import write_metadata, _metadata_path  # noqa: E402

VARIANT_BY_RUN_ID = {
    "S3-001": "baseline",
    "S3-002": "layerscale",
    "S3-003": "late_gain",
    "S3-004": "residual_scaling",
    "S3-005": "hybrid",
}

# Frozen baseline metrics, produced once against outputs/models/diffusion_best.pt,
# per Stage3_Roadmap.md Sec. 6's baseline-comparison protocol. Must exist before
# evaluate_wave_gate() can run -- this script does not fabricate a placeholder.
BASELINE_METRICS_PATH = REPO_ROOT / "outputs" / "mentor_review" / "classification_validation" / "classifier_real_eval.json"

# Primary metric used for the Wave 1 gate (>= 1 primary metric improving triggers
# proceed-to-Wave-2). Kept to ONE well-understood metric here rather than several,
# per the roadmap's own caution against manufacturing false precision.
PRIMARY_METRIC_KEY = "accuracy"


def run_candidate(run_id: str) -> None:
    """Train + evaluate one candidate. Train failures are FATAL (re-raised --
    stops the queue, per the existing STOP CONDITION semantics: an unknown
    training failure is worth halting on). Eval failures are NOT fatal --
    marked "eval_failed" in metadata.json and swallowed here, so a broken
    evaluator on one candidate can never block unrelated candidates' training
    later in the same queue invocation."""
    variant = VARIANT_BY_RUN_ID[run_id]
    cfg = load_config()
    log = get_logger(f"queue_{run_id}", cfg=cfg)

    write_metadata(run_id, variant, status="queued")
    write_metadata(run_id, variant, status="training")

    try:
        train_variant(cfg, log, variant=variant, run_id=run_id)
    except Exception:
        write_metadata(run_id, variant, status="train_failed")
        raise

    ckpt_path = RESULTS_ROOT / run_id / "checkpoints" / f"{run_id}_best.pt"
    if not ckpt_path.exists():
        write_metadata(run_id, variant, status="train_failed")
        raise FileNotFoundError(f"[{run_id}] expected checkpoint not found: {ckpt_path}")

    eval_dir = RESULTS_ROOT / run_id / "mentor_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the EXISTING mentor_eval scripts, pointed at this candidate's
    # checkpoint via their own --ckpt/--out-dir flags -- no new harness.
    try:
        for module in ("mentor_eval.classification_validation", "mentor_eval.similarity_metrics"):
            subprocess.run(
                [sys.executable, "-m", module, "--ckpt", str(ckpt_path), "--out-dir", str(eval_dir)],
                cwd=str(REPO_ROOT), check=True,
            )
    except subprocess.CalledProcessError as exc:
        write_metadata(run_id, variant, status="eval_failed", checkpoint_path=ckpt_path)
        log.error(f"[{run_id}] evaluation failed ({exc}) -- checkpoint is valid, training for "
                  f"the NEXT candidate will still proceed. Fix the evaluator and re-run "
                  f"evaluation only for {run_id}; do not retrain.")
        return

    write_metadata(run_id, variant, status="done", checkpoint_path=ckpt_path)
    log.info(f"[{run_id}] done -- checkpoint={ckpt_path}, eval output={eval_dir}")


def evaluate_wave_gate(run_ids: list[str]) -> bool:
    """Automated Wave 1 gate: proceed to Wave 2 IFF at least one of the
    given candidates improves on PRIMARY_METRIC_KEY vs. the frozen
    baseline. No chat check-in -- this is meant to run non-interactively
    once real metrics exist."""
    if not BASELINE_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Frozen baseline metrics not found at {BASELINE_METRICS_PATH} -- "
            f"the baseline-comparison protocol (Stage3_Roadmap.md Sec. 6) requires "
            f"this to exist before any gate can be evaluated. Not fabricating a "
            f"placeholder value."
        )
    with open(BASELINE_METRICS_PATH) as f:
        baseline = json.load(f)
    baseline_value = baseline[PRIMARY_METRIC_KEY]

    improved = []
    for run_id in run_ids:
        metrics_path = RESULTS_ROOT / run_id / "mentor_eval" / "classifier_real_eval.json"
        if not metrics_path.exists():
            meta_path = _metadata_path(run_id)
            status = None
            if meta_path.exists():
                with open(meta_path) as f:
                    status = json.load(f).get("status")
            if status == "eval_failed":
                raise FileNotFoundError(
                    f"[{run_id}] evaluation FAILED (metadata.json status=eval_failed) -- "
                    f"metrics at {metrics_path} were never produced. Fix the evaluator and "
                    f"re-run evaluation for {run_id} before this gate can be evaluated."
                )
            if status == "train_failed":
                raise FileNotFoundError(
                    f"[{run_id}] training FAILED (metadata.json status=train_failed) -- "
                    f"this candidate never produced a checkpoint to evaluate."
                )
            raise FileNotFoundError(f"[{run_id}] no evaluation metrics found at {metrics_path} "
                                     f"-- run_candidate() must complete before the gate can be evaluated.")
        with open(metrics_path) as f:
            candidate_metrics = json.load(f)
        candidate_value = candidate_metrics[PRIMARY_METRIC_KEY]
        if candidate_value > baseline_value:
            improved.append((run_id, candidate_value))

    if improved:
        print(f"GATE: PROCEED -- improved on {PRIMARY_METRIC_KEY} vs baseline "
              f"({baseline_value:.4f}): {improved}")
        return True
    print(f"GATE: STOP -- no candidate improved on {PRIMARY_METRIC_KEY} vs baseline "
          f"({baseline_value:.4f}). Do not train Wave 2 candidates on this evidence.")
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_ids", nargs="+", help="e.g. S3-002 S3-003")
    parser.add_argument("--gate", action="store_true",
                         help="only evaluate the gate for the given run_ids, do not train")
    args = parser.parse_args()

    if args.gate:
        proceed = evaluate_wave_gate(args.run_ids)
        sys.exit(0 if proceed else 1)

    for run_id in args.run_ids:
        run_candidate(run_id)


if __name__ == "__main__":
    main()
