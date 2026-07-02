"""
Stage 1 / Experiment 1 — Baseline Reproduction, FOR REAL, with a ledger.

This is NOT a new training or evaluation implementation. It is a thin
wrapper that calls the existing, already-verified entrypoints --
step04_transformer_diffusion's train() and _generate_final_samples(), and
mentor_eval.classification_validation's run() -- inside ExperimentLogger,
so the run actually lands in Roadmap/Stage_1_Diagnosis/Reports/
results_ledger.jsonl and MASTER_LOG.md instead of only producing scattered
output files that a later chat has to describe secondhand.

Every prior "Stage 1 finding" discussed before this script existed has no
corresponding artifact under version control in this repository -- see
Roadmap/Stage_2_Architecture_Investigation/Reports/Verification_Gate_Report.md.
This script is how that changes: it is the first Experiment 1 run whose
output the Verification Gate can actually check against.

Must be run on real GPU hardware -- this machine (8GB unified memory
Apple Silicon, no CUDA) confirmed OOM on step04's model at batch=32
during Stage 1 benchmarking; see Roadmap/Stage_1_Diagnosis/Decisions.md.

Usage (on the GPU server):
    python Roadmap/Stage_1_Diagnosis/Code/Experiment_1_Baseline/run_experiment_1_for_real.py

    # Run this FIRST, before the real run above, on a shared/near-capacity
    # GPU server: truncates to 3 epochs, caps batch size, saves a
    # checkpoint every epoch (so one actually gets produced within 3
    # epochs -- config's default save_every=25 would otherwise save
    # nothing at all in a 3-epoch run), and skips the expensive
    # classifier-retraining evaluation step (which trains its own
    # MentorClassifier on ~14k real records for 30 epochs -- not
    # something a crash-sanity-check needs to pay for). Confirms the
    # pipeline executes end-to-end, a checkpoint saves correctly, and the
    # ExperimentLogger ledger entry looks right, in minutes instead of
    # however long the full 200-epoch run takes on unknown hardware.
    python Roadmap/Stage_1_Diagnosis/Code/Experiment_1_Baseline/run_experiment_1_for_real.py --sanity-check

Writes (in addition to everything step04/classification_validation
already write):
    Roadmap/Stage_1_Diagnosis/Logs/exp1_baseline_reproduction_<ts>.log
        (or exp1_baseline_sanity_check_<ts>.log with --sanity-check)
    Roadmap/Stage_1_Diagnosis/Reports/results_ledger.jsonl  (appended)
    Roadmap/Stage_1_Diagnosis/Reports/MASTER_LOG.md         (regenerated)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "Roadmap" / "_infra"))

from experiment_logger import ExperimentLogger
from utils import load_config, get_logger, set_seed
from utils.backup import snapshot_before_write

# Existing, already-verified entrypoints -- called unmodified below.
from step04_transformer_diffusion import (
    train as step04_train,
    _generate_final_samples,
    _load_class_labels,
)
from mentor_eval.classification_validation import run as classification_validation_run

STAGE1_ROOT = REPO_ROOT / "Roadmap" / "Stage_1_Diagnosis"


def _actual_train_size(cfg, log) -> int:
    """
    Reads the same on-disk artifacts step04.train() reads (processed
    arrays + PTB-XL metadata) and reuses step04's own _load_class_labels()
    to report the POST-CLASS-MAPPING training set size -- i.e. the number
    step04 actually trains on, not the raw X_train.npy row count. This
    calls the existing function, it does not reimplement its logic.
    """
    import numpy as np
    import pandas as pd

    processed_dir = Path(cfg.paths.outputs.processed)
    class_names = json.load(open(processed_dir / "class_names.json"))
    class_mapping = json.load(open(processed_dir / "class_mapping.json"))
    rec_ids_train = np.load(processed_dir / "record_ids_train.npy")
    db_path = Path(cfg.paths.data.ptbxl) / "ptbxl_database.csv"
    ptbxl_db = pd.read_csv(str(db_path), index_col="ecg_id")
    valid_idx, _ = _load_class_labels(rec_ids_train, ptbxl_db, class_mapping, class_names, log)
    return int(len(valid_idx))


SANITY_CHECK_EPOCHS = 3
SANITY_CHECK_MAX_BATCH_SIZE = 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sanity-check", action="store_true",
        help=(
            f"Truncate to {SANITY_CHECK_EPOCHS} epochs, cap batch_size at "
            f"{SANITY_CHECK_MAX_BATCH_SIZE}, save a checkpoint every epoch, "
            f"and skip classifier-retraining evaluation. Confirms the "
            f"pipeline runs end-to-end and the ledger/checkpoint look right "
            f"before committing to the full run. Logged under a distinct "
            f"experiment_id so it can never be mistaken for real evidence."
        ),
    )
    args = parser.parse_args()

    cfg = load_config()
    experiment_id = "exp1_baseline_sanity_check" if args.sanity_check else "exp1_baseline_reproduction"
    log = get_logger(experiment_id, cfg=cfg)
    seed = int(cfg.seeds[0])
    set_seed(seed)

    if args.sanity_check:
        log.warning(
            f"--sanity-check: truncating n_epochs {int(cfg.diffusion.n_epochs)} -> {SANITY_CHECK_EPOCHS}, "
            f"batch_size {int(cfg.diffusion.batch_size)} -> "
            f"{min(int(cfg.diffusion.batch_size), SANITY_CHECK_MAX_BATCH_SIZE)}, "
            f"save_every {int(cfg.diffusion.save_every)} -> 1. "
            f"THESE NUMBERS ARE NOT EVIDENCE -- crash/plumbing check only."
        )
        cfg.diffusion.n_epochs = SANITY_CHECK_EPOCHS
        cfg.diffusion.batch_size = min(int(cfg.diffusion.batch_size), SANITY_CHECK_MAX_BATCH_SIZE)
        cfg.diffusion.save_every = 1

    models_dir = Path(cfg.paths.outputs.models)
    snapshot_before_write(models_dir)

    params = {
        "sanity_check": bool(args.sanity_check),
        "n_epochs": int(cfg.diffusion.n_epochs),
        "batch_size": int(cfg.diffusion.batch_size),
        "save_every": int(cfg.diffusion.save_every),
        "lr": float(cfg.diffusion.lr),
        "model_dim": int(cfg.diffusion.model_dim),
        "n_transformer_layers": int(cfg.diffusion.n_transformer_layers),
        "p_uncond": float(getattr(cfg.diffusion, "p_uncond", 0.10)),
        "seed": seed,
    }

    with ExperimentLogger(
        experiment_id=experiment_id,
        stage="Stage_1_Diagnosis",
        root_dir=STAGE1_ROOT,
        params=params,
        seed=seed,
        repo_dir=REPO_ROOT,
    ) as exp:
        if args.sanity_check:
            exp.log_note(
                "SANITY CHECK RUN -- truncated epochs/batch size, checkpoint "
                "correctness and pipeline plumbing only. Do not cite any "
                "metric from this run as a Stage 1 finding."
            )

        n_train_actual = _actual_train_size(cfg, log)
        exp.log_metric("n_train_records_actual", n_train_actual)
        exp.log_note(
            f"n_train_records_actual computed via step04's own "
            f"_load_class_labels() against outputs/processed/*.npy -- "
            f"reused, not reimplemented."
        )

        # ── Train the baseline diffusion model (existing entrypoint) ───────
        best_val_loss = step04_train(cfg, log)
        exp.log_metric("best_val_loss", best_val_loss)

        ckpt_path = models_dir / "diffusion_best.pt"
        if ckpt_path.exists():
            exp.log_artifact(ckpt_path, "trained diffusion checkpoint")
            exp.log_metric("checkpoint_saved", True)
        else:
            exp.log_metric("checkpoint_saved", False)
            exp.log_note(
                f"diffusion_best.pt does NOT exist at {ckpt_path} after training -- "
                f"with save_every={int(cfg.diffusion.save_every)} and "
                f"n_epochs={int(cfg.diffusion.n_epochs)}, no checkpoint boundary "
                f"was ever reached. This is the exact failure mode --sanity-check "
                f"exists to catch before a full run wastes GPU time on it."
            )
        exp.log_artifact(models_dir / "diffusion_architecture.json", "model architecture + best_val_loss")
        exp.log_artifact(Path(cfg.paths.logs) / "diffusion_training_log.csv", "per-epoch training log")

        # ── Generate baseline samples (existing entrypoint) ────────────────
        _generate_final_samples(cfg, log)
        exp.log_note("Baseline samples generated via step04's _generate_final_samples().")

        if args.sanity_check:
            exp.log_note(
                "Skipping classification_validation_run() for the sanity check -- "
                "it retrains its own MentorClassifier on ~14k real records for "
                "30 epochs, which a crash/plumbing check does not need to pay for. "
                "Run without --sanity-check for real accuracy/macro-F1 numbers."
            )
            print(f"Sanity check complete. checkpoint_saved={ckpt_path.exists()}  best_val_loss={best_val_loss:.5f}")
            return

        # ── Real-data + generated-data classifier validation (existing entrypoint) ──
        cv_out_dir = Path(cfg.paths.outputs.results).parent / "mentor_review" / "classification_validation"
        classification_validation_run(ckpt_path, cv_out_dir, cfg, seed, log)

        real_eval_path = cv_out_dir / "classifier_real_eval.json"
        gen_eval_path = cv_out_dir / "classifier_generated_eval.json"

        if real_eval_path.exists():
            real_eval = json.load(open(real_eval_path))
            exp.log_metric("real_data_accuracy", real_eval["accuracy"])
            exp.log_metric("real_data_macro_f1", real_eval["macro_f1"])
            exp.log_metric("real_data_macro_auc", real_eval.get("macro_auc"))
            exp.log_artifact(real_eval_path, "real-data classifier eval")
        else:
            exp.log_note(f"real_eval_path missing at {real_eval_path} -- classification_validation.run() did not produce it.")

        if gen_eval_path.exists():
            gen_eval = json.load(open(gen_eval_path))
            exp.log_metric("generated_data_accuracy", gen_eval["accuracy"])
            exp.log_metric("generated_data_macro_f1", gen_eval["macro_f1"])
            exp.log_metric("generated_data_excluded_classes", gen_eval.get("excluded_classes"))
            exp.log_artifact(gen_eval_path, "generated-data classifier eval")
        else:
            exp.log_note(
                f"gen_eval_path missing at {gen_eval_path} -- this means classification_validation.run() "
                f"could not evaluate generated samples even though a checkpoint now exists. Investigate "
                f"before treating Experiment 1 as complete."
            )

        print(f"Experiment 1 (real run) complete. best_val_loss={best_val_loss:.5f}")


if __name__ == "__main__":
    main()
