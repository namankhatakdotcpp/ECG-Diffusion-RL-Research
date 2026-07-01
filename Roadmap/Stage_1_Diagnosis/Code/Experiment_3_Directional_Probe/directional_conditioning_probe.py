"""
Stage 1 / Experiment 3 — Directional Conditioning Analysis.

mentor_eval/conditioning_sensitivity_probe.py already establishes whether
changing the class label changes the model's predicted noise AT ALL
(||eps_A - eps_B|| > 0). It cannot say whether that change points toward the
correct class's data manifold or in some arbitrary direction — a model could
score nonzero sensitivity while still generating semantically wrong output
for every class. This script closes that gap using MentorClassifier's
128-dim penultimate embeddings (same feature space as
mentor_eval/embedding_visualization.py) as a proxy for "clinically
meaningful ECG semantics."

Method
------
1. Extract MentorClassifier penultimate features for real ECGs of every
   generatable mentor class (Normal/STEMI/NSTEMI — AFIB has no trained
   diffusion class, see mentor_eval/class_mapping.py). Compute each class's
   real centroid mu_real(c) = mean feature vector.

2. For every ordered pair of classes (A, B), generate n_samples ECGs with
   class label A and class label B using the SAME random seed per pair
   index. Because generate_ecg() reseeds torch's global RNG immediately
   before drawing the initial noise x_T, using the same seed for both calls
   means the two generations share the identical starting noise trajectory
   and differ ONLY in the conditioning label. Extract features for both and
   compute the per-sample displacement:

       delta_gen_i = feat(gen_B_i) - feat(gen_A_i)

   This isolates the effect of the class label from the effect of sampling
   noise — a paired design, not an independent-samples comparison.

3. Compare the mean generated displacement to the real displacement between
   class centroids:

       delta_real(A, B) = mu_real(B) - mu_real(A)

       directional_score(A, B) = cosine_similarity(mean_i(delta_gen_i), delta_real(A, B))

   directional_score in [-1, 1]:
     +1  → changing the label moves generated samples exactly toward the
           correct real-data direction (conditioning works semantically)
      0  → the label changes the output, but in a direction unrelated to
           the real class difference (magnitude-only conditioning: the
           sensitivity probe would show nonzero effect, but it's the wrong
           effect)
     -1  → the label moves samples in the OPPOSITE of the correct direction

4. Centroid-movement sanity check: for each class A, is
   feat(gen_A) actually closer to mu_real(A) than to mu_real(other classes)?
   Reported as a distance-ratio, independent of the classifier's own
   decision boundary (a purely geometric check in embedding space).

5. Feature trajectory: for ONE fixed noise seed, generate all classes and
   project their embeddings (plus the real centroids) into 2D via PCA,
   drawing arrows from each class's generated point in the order the
   classes were requested — a visual complement to the numeric cosine
   scores.

Requires:
  outputs/models/diffusion_best.pt (or --ckpt override — e.g. one of
  Experiment 2's per-size checkpoints, to see if directional accuracy
  improves with more data)

Writes to Roadmap/Stage_1_Diagnosis/Outputs/Experiment_3_Directional_Probe/:
  directional_scores.csv       — cosine score + distance ratios per (A,B) pair
  directional_scores_heatmap.png
  feature_trajectory_pca.png
  directional_probe_raw.json   — full numeric detail for the written report
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import permutations
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
from mentor_eval.classification_validation import (
    MentorClassifier, TRAIN_FOLDS, VAL_FOLDS, TEST_FOLDS,
)
from mentor_eval.embedding_visualization import (
    _load_n_per_class, _train_fresh_classifier, _extract_features,
)
from mentor_eval.checkpoint_utils import load_checkpoint, generate_for_class

OUT_DIR = REPO_ROOT / "Roadmap" / "Stage_1_Diagnosis" / "Outputs" / "Experiment_3_Directional_Probe"
FIG_DIR = REPO_ROOT / "Roadmap" / "Stage_1_Diagnosis" / "Figures" / "Experiment_3_Directional_Probe"
CACHED_CLF = REPO_ROOT / "outputs" / "conditioning_analysis" / "mentor_classifier.pt"

N_REAL_PER_CLASS = 100
N_PAIRED_SAMPLES = 50


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Diffusion checkpoint (default: outputs/models/diffusion_best.pt)")
    parser.add_argument("--tag", type=str, default="baseline",
                        help="Label for this run in output filenames, e.g. 'baseline' or 'size_5000'")
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("directional_conditioning_probe", cfg=cfg)
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.ckpt) if args.ckpt else Path(cfg.paths.outputs.models) / "diffusion_best.pt"
    loaded = load_checkpoint(ckpt_path, cfg)
    if loaded is None:
        print(f"[BLOCKED] Checkpoint not found at {ckpt_path}. Run Experiment 1 (or 2) first.")
        return

    # ── Classifier: reuse cache if present (same one embedding_visualization.py uses) ──
    ptbxl_dir = Path(cfg.paths.data.ptbxl)
    filtered = filter_to_mentor_classes(load_ptbxl_database(ptbxl_dir))
    clf = MentorClassifier(n_classes=len(MENTOR_CLASSES)).to(device)
    if CACHED_CLF.exists():
        log.info(f"Loading cached MentorClassifier from {CACHED_CLF}")
        clf.load_state_dict(torch.load(str(CACHED_CLF), map_location=device))
    else:
        log.info("No cached classifier — training fresh (will be cached for reuse) …")
        clf = _train_fresh_classifier(ptbxl_dir, filtered, device, log)
        CACHED_CLF.parent.mkdir(parents=True, exist_ok=True)
        torch.save(clf.state_dict(), str(CACHED_CLF))
    clf.eval()

    # ── Real centroids ─────────────────────────────────────────────────────────
    log.info(f"Loading up to {N_REAL_PER_CLASS} real samples per mentor class …")
    all_folds = TEST_FOLDS + VAL_FOLDS + TRAIN_FOLDS
    X_real, y_real = _load_n_per_class(ptbxl_dir, filtered, all_folds, N_REAL_PER_CLASS, log)
    feat_real = _extract_features(clf, X_real, device)

    generatable = [c for c in MENTOR_CLASSES if MENTOR_TO_TRAINED_CLASS.get(c) is not None]
    log.info(f"Generatable mentor classes (have a trained diffusion class): {generatable}")
    excluded = [c for c in MENTOR_CLASSES if c not in generatable]
    if excluded:
        log.info(f"Excluded (no trained diffusion class — real-data-only in this analysis): {excluded}")

    centroid_real = {}
    name_to_idx = {n: i for i, n in enumerate(MENTOR_CLASSES)}
    for cls in MENTOR_CLASSES:
        mask = y_real == name_to_idx[cls]
        if mask.sum() == 0:
            log.warning(f"No real samples loaded for {cls} — skipping its centroid.")
            continue
        centroid_real[cls] = feat_real[mask].mean(axis=0)

    stats_path = Path(cfg.paths.outputs.processed) / "preprocessing_stats.json"
    prep_stats = json.load(open(stats_path)) if stats_path.exists() else None

    # ── Paired generation per ordered class pair ────────────────────────────────
    results = []
    raw = {"generatable_classes": generatable, "excluded_classes": excluded, "pairs": {}}

    for cls_a, cls_b in permutations(generatable, 2):
        trained_a = MENTOR_TO_TRAINED_CLASS[cls_a]
        trained_b = MENTOR_TO_TRAINED_CLASS[cls_b]
        log.info(f"Pair ({cls_a} -> {cls_b}): generating {N_PAIRED_SAMPLES} paired samples …")

        deltas = []
        feats_a_all, feats_b_all = [], []
        for i in range(N_PAIRED_SAMPLES):
            seed = 1000 + i  # identical seed for both calls -> identical initial noise x_T
            samp_a, err_a = generate_for_class(loaded, trained_a, n_samples=1, cfg=cfg, seed=seed, stats=prep_stats)
            samp_b, err_b = generate_for_class(loaded, trained_b, n_samples=1, cfg=cfg, seed=seed, stats=prep_stats)
            if err_a or err_b:
                log.warning(f"  skip pair sample {i}: {err_a or err_b}")
                continue
            fa = _extract_features(clf, samp_a, device)[0]
            fb = _extract_features(clf, samp_b, device)[0]
            feats_a_all.append(fa)
            feats_b_all.append(fb)
            deltas.append(fb - fa)

        if not deltas:
            log.warning(f"  no valid paired samples for ({cls_a} -> {cls_b}) — skipping.")
            continue

        deltas = np.stack(deltas)
        mean_delta_gen = deltas.mean(axis=0)
        delta_real = centroid_real[cls_b] - centroid_real[cls_a]
        score = cosine_sim(mean_delta_gen, delta_real)

        # per-sample cosine, for a distribution (not just the mean-displacement score)
        per_sample_scores = [cosine_sim(d, delta_real) for d in deltas]

        # geometric sanity check: is gen(cls_a) closer to real centroid A than to real centroid B?
        feats_a_all = np.stack(feats_a_all)
        dist_to_own = np.linalg.norm(feats_a_all - centroid_real[cls_a], axis=1).mean()
        dist_to_other = np.linalg.norm(feats_a_all - centroid_real[cls_b], axis=1).mean()
        distance_ratio = float(dist_to_own / (dist_to_other + 1e-12))  # < 1 means correctly closer to own class

        row = {
            "class_a": cls_a, "class_b": cls_b,
            "directional_score": score,
            "per_sample_score_mean": float(np.mean(per_sample_scores)),
            "per_sample_score_std": float(np.std(per_sample_scores)),
            "n_paired_samples": len(deltas),
            "distance_ratio_a_own_vs_other": distance_ratio,
        }
        results.append(row)
        raw["pairs"][f"{cls_a}->{cls_b}"] = {
            **row,
            "per_sample_scores": per_sample_scores,
        }
        log.info(f"  directional_score={score:.4f}  distance_ratio={distance_ratio:.4f}")

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / f"directional_scores_{args.tag}.csv", index=False)
    with open(OUT_DIR / f"directional_probe_raw_{args.tag}.json", "w") as f:
        json.dump(raw, f, indent=2)

    # ── Heatmap of directional scores ───────────────────────────────────────────
    if results:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        mat = np.full((len(generatable), len(generatable)), np.nan)
        idx = {c: i for i, c in enumerate(generatable)}
        for row in results:
            mat[idx[row["class_a"]], idx[row["class_b"]]] = row["directional_score"]
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(generatable))); ax.set_xticklabels(generatable, rotation=45, ha="right")
        ax.set_yticks(range(len(generatable))); ax.set_yticklabels(generatable)
        ax.set_xlabel("Target class B"); ax.set_ylabel("Source class A")
        ax.set_title(f"Directional conditioning score cos(delta_gen, delta_real)\n({args.tag})")
        for i in range(len(generatable)):
            for j in range(len(generatable)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            color="white" if abs(mat[i, j]) > 0.5 else "black")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"directional_scores_heatmap_{args.tag}.png", dpi=200)
        plt.close(fig)

    # ── Feature trajectory (single fixed seed, all generatable classes) ────────
    log.info("Building feature trajectory (fixed seed, all classes) …")
    fixed_seed = 777
    traj_feats, traj_labels = [], []
    for cls in generatable:
        trained_cls = MENTOR_TO_TRAINED_CLASS[cls]
        samp, err = generate_for_class(loaded, trained_cls, n_samples=1, cfg=cfg, seed=fixed_seed, stats=prep_stats)
        if err:
            continue
        traj_feats.append(_extract_features(clf, samp, device)[0])
        traj_labels.append(cls)

    if traj_feats:
        combined = np.stack(traj_feats + [centroid_real[c] for c in generatable if c in centroid_real])
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(combined)
        n_gen = len(traj_feats)
        gen_coords = coords[:n_gen]
        real_coords = coords[n_gen:]

        fig, ax = plt.subplots(figsize=(6, 6))
        palette = plt.cm.tab10.colors
        for i, cls in enumerate(traj_labels):
            ax.scatter(*gen_coords[i], marker="^", s=120, color=palette[i], label=f"{cls} (generated)")
        for i, cls in enumerate([c for c in generatable if c in centroid_real]):
            ax.scatter(*real_coords[i], marker="o", s=120, color=palette[i],
                       edgecolors="k", label=f"{cls} (real centroid)")
        for i in range(n_gen - 1):
            ax.annotate("", xy=gen_coords[i + 1], xytext=gen_coords[i],
                        arrowprops=dict(arrowstyle="->", color="gray", alpha=0.6))
        ax.set_title(f"Feature trajectory — fixed noise seed, class label swept\n({args.tag})")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"feature_trajectory_pca_{args.tag}.png", dpi=200)
        plt.close(fig)

    log.info(f"Done. See {OUT_DIR} and {FIG_DIR}")
    print(f"✓ Directional conditioning probe complete (tag={args.tag}). See {OUT_DIR} and {FIG_DIR}")


if __name__ == "__main__":
    main()
