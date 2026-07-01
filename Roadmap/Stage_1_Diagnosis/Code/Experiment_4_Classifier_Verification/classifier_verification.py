"""
Stage 1 / Experiment 4 — MentorClassifier Verification: is AFIB an
out-of-distribution reject bucket?

Every other Stage 1 experiment treats MentorClassifier's prediction as
ground truth for "did conditioning work." That's only valid if the
classifier's class assignments are trustworthy. AFIB is the class most at
risk of being a de-facto reject bucket: it's a rhythm code (not a
morphology code like the others), it has the fewest real examples anywhere
in this pipeline (103 records total, see mentor_eval/class_mapping.py), and
it has no generation path — every AFIB prediction in this pipeline is
necessarily about REAL AFIB ECGs, never generated ones.

This experiment does NOT touch the diffusion model at all — it only
corrupts REAL ECGs with increasing noise and watches what MentorClassifier
does. No generation, no checkpoint required.

Method
------
1. Load real test-fold ECGs for all 4 mentor classes.
2. For each noise level sigma in NOISE_LEVELS (Gaussian noise added
   directly in z-score space, clipped to the same [-4, 4] range used
   throughout preprocessing so noisy inputs stay in-distribution for the
   *input scale*, even though they become increasingly semantically
   corrupted):
     a. Noise robustness: per-class accuracy at this noise level.
     b. Prediction drift: fraction of samples whose predicted class differs
        from their sigma=0 prediction.
     c. AFIB attraction: of samples that flip prediction as noise
        increases, what fraction flip specifically TO AFIB, versus flipping
        to each other class? If AFIB acts as a reject bucket, this fraction
        should be disproportionately high relative to AFIB's share of
        classes (1/4) and should grow with noise.
     d. Confidence calibration: mean max-softmax confidence of predictions,
        broken out by predicted class. A classifier that is well-calibrated
        should become LESS confident as input is progressively destroyed.
        If AFIB predictions stay confident even at high noise while other
        classes' confidence drops, that is a symptom of AFIB functioning as
        an "I don't recognize this" bucket that the model is nonetheless
        confident about (poor calibration specifically for that class).

Writes to Roadmap/Stage_1_Diagnosis/Outputs/Experiment_4_Classifier_Verification/:
  noise_robustness.csv        — accuracy per class per noise level
  prediction_drift.csv        — flip-to-class fractions per noise level
  confidence_calibration.csv  — mean confidence per predicted class per noise level
Writes to Roadmap/Stage_1_Diagnosis/Figures/Experiment_4_Classifier_Verification/:
  accuracy_vs_noise.png, afib_attraction_vs_noise.png, confidence_vs_noise.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from utils import load_config, get_logger, set_seed
from mentor_eval.class_mapping import MENTOR_CLASSES, load_ptbxl_database, filter_to_mentor_classes
from mentor_eval.classification_validation import (
    MentorClassifier, train_classifier, _load_signals_for_fold,
    TRAIN_FOLDS, VAL_FOLDS, TEST_FOLDS,
)

OUT_DIR = REPO_ROOT / "Roadmap" / "Stage_1_Diagnosis" / "Outputs" / "Experiment_4_Classifier_Verification"
FIG_DIR = REPO_ROOT / "Roadmap" / "Stage_1_Diagnosis" / "Figures" / "Experiment_4_Classifier_Verification"
CACHED_CLF = REPO_ROOT / "outputs" / "conditioning_analysis" / "mentor_classifier.pt"
CLIP_RANGE = (-4.0, 4.0)  # matches preprocessing.clip_range in config.yaml
NOISE_LEVELS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
SEED = 42


def main() -> None:
    cfg = load_config()
    log = get_logger("classifier_verification", cfg=cfg)
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ptbxl_dir = Path(cfg.paths.data.ptbxl)
    filtered = filter_to_mentor_classes(load_ptbxl_database(ptbxl_dir))
    n_classes = len(MENTOR_CLASSES)
    name_to_idx = {n: i for i, n in enumerate(MENTOR_CLASSES)}
    afib_idx = name_to_idx["AFIB"]

    clf = MentorClassifier(n_classes=n_classes).to(device)
    if CACHED_CLF.exists():
        log.info(f"Loading cached MentorClassifier from {CACHED_CLF}")
        clf.load_state_dict(torch.load(str(CACHED_CLF), map_location=device))
    else:
        log.info("No cached classifier — training fresh on real PTB-XL …")
        X_train, y_train = _load_signals_for_fold(ptbxl_dir, filtered, TRAIN_FOLDS, log)
        X_val, y_val = _load_signals_for_fold(ptbxl_dir, filtered, VAL_FOLDS, log)
        clf = train_classifier(X_train, y_train, X_val, y_val, n_classes, device, log)
        CACHED_CLF.parent.mkdir(parents=True, exist_ok=True)
        torch.save(clf.state_dict(), str(CACHED_CLF))
    clf.eval()

    log.info("Loading real test-fold ECGs for all 4 mentor classes …")
    X_test, y_test = _load_signals_for_fold(ptbxl_dir, filtered, TEST_FOLDS, log)
    log.info(f"Test set: {len(X_test)} records, "
             f"per-class counts={ {c: int((y_test == i).sum()) for c, i in name_to_idx.items()} }")

    Xt_clean = torch.from_numpy(X_test.transpose(0, 2, 1)).float().to(device)

    rng = np.random.default_rng(SEED)
    clean_pred = None
    accuracy_rows, drift_rows, confidence_rows = [], [], []

    for sigma in NOISE_LEVELS:
        if sigma == 0.0:
            Xt_noisy = Xt_clean
        else:
            noise = torch.from_numpy(
                rng.normal(0, sigma, size=X_test.shape).astype(np.float32)
            ).permute(0, 2, 1).to(device)
            Xt_noisy = torch.clamp(Xt_clean + noise, *CLIP_RANGE)

        with torch.no_grad():
            logits = clf(Xt_noisy)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        pred = probs.argmax(axis=1)
        conf = probs.max(axis=1)

        if sigma == 0.0:
            clean_pred = pred.copy()

        # (a) noise robustness — per-class accuracy
        for cls, ci in name_to_idx.items():
            mask = y_test == ci
            if mask.sum() == 0:
                continue
            acc = float((pred[mask] == ci).mean())
            accuracy_rows.append({"sigma": sigma, "class": cls, "accuracy": acc, "n": int(mask.sum())})

        # (b) prediction drift + (c) AFIB attraction
        flipped = pred != clean_pred
        n_flipped = int(flipped.sum())
        flip_to_counts = {c: int(((pred == ci) & flipped).sum()) for c, ci in name_to_idx.items()}
        drift_rows.append({
            "sigma": sigma,
            "n_flipped": n_flipped,
            "frac_flipped": float(n_flipped / len(pred)),
            **{f"flip_to_{c}_frac": (flip_to_counts[c] / n_flipped if n_flipped > 0 else 0.0)
               for c in MENTOR_CLASSES},
            "afib_attraction_ratio": (
                (flip_to_counts["AFIB"] / n_flipped) / (1.0 / n_classes) if n_flipped > 0 else float("nan")
            ),  # >1 means AFIB gains disproportionately more flipped predictions than a 1/n_classes base rate
        })

        # (d) confidence calibration — mean confidence per predicted class
        for cls, ci in name_to_idx.items():
            mask = pred == ci
            if mask.sum() == 0:
                continue
            confidence_rows.append({
                "sigma": sigma, "predicted_class": cls,
                "mean_confidence": float(conf[mask].mean()),
                "n_predicted": int(mask.sum()),
            })

        log.info(f"sigma={sigma}: frac_flipped={drift_rows[-1]['frac_flipped']:.4f}  "
                 f"afib_attraction_ratio={drift_rows[-1]['afib_attraction_ratio']:.3f}")

    acc_df = pd.DataFrame(accuracy_rows)
    drift_df = pd.DataFrame(drift_rows)
    conf_df = pd.DataFrame(confidence_rows)
    acc_df.to_csv(OUT_DIR / "noise_robustness.csv", index=False)
    drift_df.to_csv(OUT_DIR / "prediction_drift.csv", index=False)
    conf_df.to_csv(OUT_DIR / "confidence_calibration.csv", index=False)

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    for cls in MENTOR_CLASSES:
        sub = acc_df[acc_df["class"] == cls]
        ax.plot(sub["sigma"], sub["accuracy"], marker="o", label=cls)
    ax.set_xlabel("Gaussian noise sigma (z-score units)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Noise robustness — per-class accuracy vs. corruption")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "accuracy_vs_noise.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(drift_df["sigma"], drift_df["afib_attraction_ratio"], marker="o", color="crimson")
    ax.axhline(1.0, linestyle="--", color="gray", label="Chance level (1/n_classes base rate)")
    ax.set_xlabel("Gaussian noise sigma (z-score units)")
    ax.set_ylabel("AFIB attraction ratio")
    ax.set_title("Does AFIB disproportionately absorb noise-flipped predictions?\n(ratio > 1 = AFIB acts as a reject bucket)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "afib_attraction_vs_noise.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for cls in MENTOR_CLASSES:
        sub = conf_df[conf_df["predicted_class"] == cls]
        ax.plot(sub["sigma"], sub["mean_confidence"], marker="o", label=cls)
    ax.set_xlabel("Gaussian noise sigma (z-score units)")
    ax.set_ylabel("Mean softmax confidence of predicted class")
    ax.set_title("Confidence calibration under corruption\n(flat/high AFIB line despite noise = miscalibrated reject bucket)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confidence_vs_noise.png", dpi=200)
    plt.close(fig)

    log.info(f"Done. See {OUT_DIR} and {FIG_DIR}")
    print(f"✓ Classifier verification complete. See {OUT_DIR} and {FIG_DIR}")


if __name__ == "__main__":
    main()
