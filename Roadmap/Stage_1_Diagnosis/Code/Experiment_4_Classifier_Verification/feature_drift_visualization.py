"""
Stage 1 / Experiment 4.5 — Feature Drift Visualization
(Real -> Noise -> Generated, in MentorClassifier embedding space).

Experiment 4 showed AFIB absorbs a disproportionate share of
misclassifications at moderate noise (numeric evidence). This experiment
makes that finding visual and ties it to Experiment 3's generated-sample
embeddings: plot real, progressively-noised-real, and (if a checkpoint is
available) generated ECGs all in the SAME projected 2D space, so it's
possible to see whether noised-real samples drift toward the AFIB region
specifically (matching Experiment 4's numeric finding) and whether
generated samples end up near real clusters or off in their own region
(matching Experiment 3's directional-conditioning question).

Method
------
1. Extract MentorClassifier penultimate features (128-dim) for:
     a. Clean real samples, N_PER_CLASS per mentor class.
     b. The SAME real samples with Gaussian noise added at each sigma in
        NOISE_LEVELS (paired — same underlying ECG, so its drift path can
        be traced sample-by-sample, not just cluster-by-cluster).
     c. Generated samples per generatable class (skipped gracefully with a
        [PARTIAL] note if no checkpoint is available yet — this half of
        the picture depends on Experiment 1).
2. Fit ONE PCA on the union of all points (real + noised + generated) so
   every source lives in a shared, comparable 2D space (fitting separate
   projections per source would make "distance to AFIB" visually
   meaningless).
3. Plot: color = class (real class label, or nearest requested class for
   generated), marker = source stage, with faint connecting lines tracing
   a handful of individual samples' real -> noise1 -> noise2 -> ... path,
   so the AFIB-attraction effect from Experiment 4 is visible as literal
   drift toward the AFIB cluster.

Writes to Roadmap/Stage_1_Diagnosis/Outputs/Experiment_4_Classifier_Verification/:
  feature_drift_features.csv   — all points' 2D coords, source, class
Writes to Roadmap/Stage_1_Diagnosis/Figures/Experiment_4_Classifier_Verification/:
  feature_drift_pca.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from utils import load_config, get_logger, set_seed
from mentor_eval.class_mapping import (
    MENTOR_CLASSES, MENTOR_TO_TRAINED_CLASS, load_ptbxl_database, filter_to_mentor_classes,
)
from mentor_eval.classification_validation import MentorClassifier, TRAIN_FOLDS, VAL_FOLDS, TEST_FOLDS
from mentor_eval.embedding_visualization import _load_n_per_class, _train_fresh_classifier, _extract_features
from mentor_eval.checkpoint_utils import load_checkpoint, generate_for_class

OUT_DIR = REPO_ROOT / "Roadmap" / "Stage_1_Diagnosis" / "Outputs" / "Experiment_4_Classifier_Verification"
FIG_DIR = REPO_ROOT / "Roadmap" / "Stage_1_Diagnosis" / "Figures" / "Experiment_4_Classifier_Verification"
CACHED_CLF = REPO_ROOT / "outputs" / "conditioning_analysis" / "mentor_classifier.pt"
CLIP_RANGE = (-4.0, 4.0)
NOISE_LEVELS = [0.25, 0.5, 1.0]   # subset of Experiment 4's levels — the moderate-noise regime that matters most
N_PER_CLASS = 30
N_TRAJECTORY_SAMPLES = 3  # per class, how many individual drift paths to draw
N_GEN = 60
SEED = 42


def main() -> None:
    cfg = load_config()
    log = get_logger("feature_drift_visualization", cfg=cfg)
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ptbxl_dir = Path(cfg.paths.data.ptbxl)
    filtered = filter_to_mentor_classes(load_ptbxl_database(ptbxl_dir))
    n_classes = len(MENTOR_CLASSES)

    clf = MentorClassifier(n_classes=n_classes).to(device)
    if CACHED_CLF.exists():
        log.info(f"Loading cached MentorClassifier from {CACHED_CLF}")
        clf.load_state_dict(torch.load(str(CACHED_CLF), map_location=device))
    else:
        log.info("No cached classifier — training fresh …")
        clf = _train_fresh_classifier(ptbxl_dir, filtered, device, log)
        CACHED_CLF.parent.mkdir(parents=True, exist_ok=True)
        torch.save(clf.state_dict(), str(CACHED_CLF))
    clf.eval()

    # ── Real samples (paired across noise levels) ───────────────────────────────
    log.info(f"Loading {N_PER_CLASS} real samples per mentor class …")
    all_folds = TEST_FOLDS + VAL_FOLDS + TRAIN_FOLDS
    X_real, y_real = _load_n_per_class(ptbxl_dir, filtered, all_folds, N_PER_CLASS, log)

    rng = np.random.default_rng(SEED)
    records = []  # each: dict(feat=(128,), class_idx, source, sample_id)

    feat_clean = _extract_features(clf, X_real, device)
    for i in range(len(X_real)):
        records.append({"feat": feat_clean[i], "class_idx": int(y_real[i]), "source": "real_clean", "sample_id": i})

    for sigma in NOISE_LEVELS:
        noise = rng.normal(0, sigma, size=X_real.shape).astype(np.float32)
        X_noisy = np.clip(X_real + noise, *CLIP_RANGE)
        feat_noisy = _extract_features(clf, X_noisy, device)
        for i in range(len(X_noisy)):
            records.append({"feat": feat_noisy[i], "class_idx": int(y_real[i]),
                            "source": f"noise_sigma_{sigma}", "sample_id": i})

    # ── Generated samples (if a diffusion checkpoint exists) ────────────────────
    ckpt_path = Path(cfg.paths.outputs.models) / "diffusion_best.pt"
    loaded = load_checkpoint(ckpt_path, cfg)
    stats_path = Path(cfg.paths.outputs.processed) / "preprocessing_stats.json"
    prep_stats = json.load(open(stats_path)) if stats_path.exists() else None

    has_generated = False
    if loaded is None:
        print(f"[PARTIAL] No checkpoint at {ckpt_path} — plotting real+noise only. "
              "Re-run after Experiment 1 for the full real->noise->generated picture.")
    else:
        name_to_idx = {n: i for i, n in enumerate(MENTOR_CLASSES)}
        for cls in MENTOR_CLASSES:
            trained_cls = MENTOR_TO_TRAINED_CLASS.get(cls)
            if trained_cls is None:
                continue
            samples, err = generate_for_class(loaded, trained_cls, n_samples=N_GEN, cfg=cfg, seed=SEED, stats=prep_stats)
            if err:
                continue
            feats = _extract_features(clf, samples, device)
            for i in range(len(feats)):
                records.append({"feat": feats[i], "class_idx": name_to_idx[cls], "source": "generated", "sample_id": -1})
            has_generated = True

    # ── Shared PCA fit ───────────────────────────────────────────────────────────
    all_feats = np.stack([r["feat"] for r in records])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(all_feats)
    for r, c in zip(records, coords):
        r["x"], r["y"] = float(c[0]), float(c[1])

    import csv
    with open(OUT_DIR / "feature_drift_features.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "class", "sample_id", "x", "y"])
        for r in records:
            w.writerow(["real_clean" if r["source"] == "real_clean" else r["source"],
                        MENTOR_CLASSES[r["class_idx"]], r["sample_id"], r["x"], r["y"]])

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    palette = plt.cm.tab10.colors
    source_order = ["real_clean"] + [f"noise_sigma_{s}" for s in NOISE_LEVELS] + (["generated"] if has_generated else [])
    source_alpha = {s: a for s, a in zip(source_order, np.linspace(0.35, 0.9, len(source_order)))}
    source_marker = {"real_clean": "o", "generated": "^"}
    for s in [f"noise_sigma_{sg}" for sg in NOISE_LEVELS]:
        source_marker[s] = "s"

    for ci, cls in enumerate(MENTOR_CLASSES):
        for src in source_order:
            pts = [r for r in records if r["class_idx"] == ci and r["source"] == src]
            if not pts:
                continue
            xs = [p["x"] for p in pts]
            ys = [p["y"] for p in pts]
            ax.scatter(xs, ys, color=palette[ci % len(palette)], marker=source_marker.get(src, "x"),
                      alpha=source_alpha.get(src, 0.6), s=35,
                      edgecolors="k" if src == "generated" else "none", linewidths=0.4,
                      label=f"{cls} ({src})")

    # Trajectory lines: real_clean -> noise_sigma_0.25 -> ... for a few sample ids per class
    for ci, cls in enumerate(MENTOR_CLASSES):
        sample_ids = sorted({r["sample_id"] for r in records
                             if r["class_idx"] == ci and r["source"] == "real_clean"})[:N_TRAJECTORY_SAMPLES]
        for sid in sample_ids:
            path_sources = ["real_clean"] + [f"noise_sigma_{s}" for s in NOISE_LEVELS]
            pts = []
            for src in path_sources:
                match = [r for r in records if r["class_idx"] == ci and r["source"] == src and r["sample_id"] == sid]
                if match:
                    pts.append(match[0])
            if len(pts) >= 2:
                ax.plot([p["x"] for p in pts], [p["y"] for p in pts],
                       color=palette[ci % len(palette)], alpha=0.5, linewidth=1.0, zorder=1)

    ax.set_title("Feature drift: real -> noise -> generated\n"
                 "(circles=clean real, squares=noised real, triangles=generated;\n"
                 "lines trace individual samples' drift as noise increases)", fontsize=10)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=6, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_drift_pca.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    log.info(f"Done (has_generated={has_generated}). See {OUT_DIR} and {FIG_DIR}")
    print(f"✓ Feature drift visualization complete (has_generated={has_generated}). See {OUT_DIR} and {FIG_DIR}")


if __name__ == "__main__":
    main()
