"""
mentor_eval/similarity_metrics.py — Mahalanobis, Bhattacharyya, cosine
similarity between real and generated ECGs, per class.

Feature representation: per-lead summary statistics (mean, std, skew,
kurtosis) -> a 48-dim vector (12 leads x 4 stats) per ECG. Chosen over a
CNN embedding (e.g. FEDEncoder) because it needs no additional training
and is far more stable for a 48-dim covariance estimate than a
high-dimensional or trained-embedding space. Swapping in FEDEncoder
embeddings instead is a one-line change to `extract_features` if preferred.

REQUIRES generated samples, which requires outputs/models/diffusion_best.pt.
Not available on this local machine — write-only deliverable until run on
the GPU server.

Stability check: Mahalanobis/Bhattacharyya need a well-conditioned
covariance estimate. With 48 features, we require at least
5 x 48 = 240 real samples per class before trusting the covariance-based
metrics; classes below that are flagged in the output, not silently
computed anyway.

Writes:
  outputs/mentor_review/similarity_metrics/similarity_metrics.csv
  outputs/mentor_review/similarity_metrics/README.md  (how to read each number)

Usage:
    python -m mentor_eval.similarity_metrics [--ckpt PATH] [--out-dir PATH] [--n-generated 200]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy import stats as sp_stats
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_config, get_logger, set_seed
from mentor_eval.class_mapping import (
    MENTOR_CLASSES, MENTOR_TO_TRAINED_CLASS, load_ptbxl_database, filter_to_mentor_classes,
)
from mentor_eval.checkpoint_utils import load_checkpoint, generate_for_class

MIN_SAMPLES_FOR_COVARIANCE_MULTIPLIER = 5  # require >= 5x feature_dim real samples


def extract_features(signals: np.ndarray) -> np.ndarray:
    """(N, 1000, 12) -> (N, 48): per-lead [mean, std, skew, kurtosis]."""
    mean = signals.mean(axis=1)                       # (N, 12)
    std  = signals.std(axis=1)                         # (N, 12)
    skew = sp_stats.skew(signals, axis=1)               # (N, 12)
    kurt = sp_stats.kurtosis(signals, axis=1)           # (N, 12)
    return np.concatenate([mean, std, skew, kurt], axis=1)  # (N, 48)


def mahalanobis_distance(gen_feats: np.ndarray, real_feats: np.ndarray) -> float:
    """Mean Mahalanobis distance of generated samples from the real-class
    distribution (mean + regularised covariance of real_feats)."""
    mu = real_feats.mean(axis=0)
    cov = np.cov(real_feats, rowvar=False)
    cov_reg = cov + np.eye(cov.shape[0]) * 1e-6  # ridge regularisation for stability
    inv_cov = np.linalg.pinv(cov_reg)
    diffs = gen_feats - mu
    d2 = np.einsum("ij,jk,ik->i", diffs, inv_cov, diffs)
    d2 = np.clip(d2, 0, None)
    return float(np.sqrt(d2).mean())


def bhattacharyya_distance(real_feats: np.ndarray, gen_feats: np.ndarray) -> float:
    """Gaussian-assumption Bhattacharyya distance between real and generated
    feature distributions."""
    mu1, mu2 = real_feats.mean(axis=0), gen_feats.mean(axis=0)
    cov1 = np.cov(real_feats, rowvar=False) + np.eye(real_feats.shape[1]) * 1e-6
    cov2 = np.cov(gen_feats, rowvar=False) + np.eye(gen_feats.shape[1]) * 1e-6
    cov_avg = (cov1 + cov2) / 2.0

    diff = mu1 - mu2
    term1 = 0.125 * diff @ np.linalg.pinv(cov_avg) @ diff

    sign1, logdet1 = np.linalg.slogdet(cov1)
    sign2, logdet2 = np.linalg.slogdet(cov2)
    sign_avg, logdet_avg = np.linalg.slogdet(cov_avg)
    if min(sign1, sign2, sign_avg) <= 0:
        return float("nan")  # covariance not positive-definite — don't fabricate a number
    term2 = 0.5 * (logdet_avg - 0.5 * (logdet1 + logdet2))
    return float(term1 + term2)


def matched_cosine_similarity(real_signals: np.ndarray, gen_signals: np.ndarray) -> float:
    """Mean cosine similarity between each generated ECG (flattened 12-lead)
    and its nearest-neighbour real ECG (flattened 12-lead) — "matched" in
    the same nearest-neighbour sense already used for DTW elsewhere in this
    repo (step05_baseline_eval.py)."""
    real_flat = real_signals.reshape(len(real_signals), -1)
    gen_flat = gen_signals.reshape(len(gen_signals), -1)

    real_norm = real_flat / (np.linalg.norm(real_flat, axis=1, keepdims=True) + 1e-8)
    gen_norm = gen_flat / (np.linalg.norm(gen_flat, axis=1, keepdims=True) + 1e-8)

    sims = []
    for g in gen_norm:
        dists = cdist(g[None, :], real_norm, metric="euclidean")[0]
        nn_idx = int(np.argmin(dists))
        sims.append(float(np.dot(g, real_norm[nn_idx])))
    return float(np.mean(sims))


def run(ckpt_path: Path, out_dir: Path, cfg, n_generated: int, seed: int, log) -> pd.DataFrame:
    loaded = load_checkpoint(ckpt_path, cfg)
    if loaded is None:
        print(
            f"\n[BLOCKED] No checkpoint found at {ckpt_path}.\n"
            f"  Similarity metrics compare real vs. GENERATED distributions — there is\n"
            f"  nothing to compare without the trained model. Train on the GPU server,\n"
            f"  then re-run this script there.\n"
            f"  No metrics were computed — nothing fabricated.\n"
        )
        sys.exit(1)

    ptbxl_dir = Path(cfg.paths.data.ptbxl)
    db = load_ptbxl_database(ptbxl_dir)
    filtered = filter_to_mentor_classes(db)

    stats_path = Path(cfg.paths.outputs.processed) / "preprocessing_stats.json"
    prep_stats = None
    if stats_path.exists():
        import json
        prep_stats = json.load(open(stats_path))

    rng = np.random.default_rng(seed)
    rows = []

    for cls in MENTOR_CLASSES:
        candidates = filtered[filtered["mentor_class"] == cls]
        real_signals = []
        order = rng.permutation(len(candidates))
        for idx in order[: min(len(candidates), max(n_generated * 2, 300))]:
            rec = candidates.iloc[int(idx)]
            try:
                sig = wfdb.rdrecord(str(ptbxl_dir / str(rec["filename_lr"]))).p_signal
            except Exception:
                continue
            if sig.shape == (1000, 12) and np.isfinite(sig).all():
                real_signals.append(sig)
        real_signals = np.array(real_signals)

        trained_cls = MENTOR_TO_TRAINED_CLASS.get(cls)
        if trained_cls is None or len(real_signals) == 0:
            rows.append({
                "class": cls, "n_real": len(real_signals), "n_generated": 0,
                "mahalanobis": None, "bhattacharyya": None, "cosine_similarity": None,
                "flag": f"No generated samples available ('{cls}' has no trained-model class)" if trained_cls is None
                        else "No readable real samples found",
            })
            continue

        gen_signals, err = generate_for_class(
            loaded, trained_cls, n_samples=n_generated, cfg=cfg, seed=seed, stats=prep_stats,
        )
        if err:
            rows.append({
                "class": cls, "n_real": len(real_signals), "n_generated": 0,
                "mahalanobis": None, "bhattacharyya": None, "cosine_similarity": None,
                "flag": err,
            })
            continue

        real_feats = extract_features(real_signals)
        gen_feats = extract_features(gen_signals)

        min_required = MIN_SAMPLES_FOR_COVARIANCE_MULTIPLIER * real_feats.shape[1]
        if len(real_feats) < min_required:
            rows.append({
                "class": cls, "n_real": len(real_signals), "n_generated": len(gen_signals),
                "mahalanobis": None, "bhattacharyya": None,
                "cosine_similarity": round(matched_cosine_similarity(real_signals, gen_signals), 4),
                "flag": (
                    f"Only {len(real_feats)} real samples (<{min_required} required for stable "
                    f"48-dim covariance) — Mahalanobis/Bhattacharyya skipped, not fabricated."
                ),
            })
            continue

        rows.append({
            "class": cls, "n_real": len(real_signals), "n_generated": len(gen_signals),
            "mahalanobis": round(mahalanobis_distance(gen_feats, real_feats), 4),
            "bhattacharyya": round(bhattacharyya_distance(real_feats, gen_feats), 4),
            "cosine_similarity": round(matched_cosine_similarity(real_signals, gen_signals), 4),
            "flag": "",
        })
        log.info(f"{cls}: computed similarity metrics (n_real={len(real_signals)}, n_gen={len(gen_signals)})")

    out_df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_dir / "similarity_metrics.csv", index=False)

    readme = """# How to read these similarity metrics

**Mahalanobis distance** — how far the average generated ECG (in 48-dim
feature space: per-lead mean/std/skew/kurtosis) sits from the real
distribution, scaled by the real distribution's spread.
**Lower is better.** 0 would mean the generated samples are
indistinguishable from real ones; values much above ~3-5 suggest the
generator drifts outside the realistic range for that class.

**Bhattacharyya distance** — how much the real and generated feature
distributions overlap (Gaussian assumption). **Lower is better** (0 =
identical distributions). Unlike Mahalanobis, this is symmetric and also
penalises mismatched spread/shape, not just a mean-shift.

**Cosine similarity** — for each generated ECG, the cosine similarity to
its nearest real ECG (flattened 12-lead waveform). **Higher is better**,
range [-1, 1]. Close to 1 means the generated waveform's overall shape
closely matches a real exemplar; this is a shape/morphology check, not a
distributional one.

A `flag` column explains why a metric is blank for a class (too few real
samples for a stable estimate, or no generated samples available for that
class at all — see SUMMARY.md for the AFIB caveat).
"""
    (out_dir / "README.md").write_text(readme)

    print(out_df.to_string(index=False))
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Mahalanobis/Bhattacharyya/cosine similarity, real vs generated.")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--n-generated", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("similarity_metrics", cfg=cfg)
    set_seed(args.seed)

    ckpt_path = Path(args.ckpt) if args.ckpt else Path(cfg.paths.outputs.models) / "diffusion_best.pt"
    out_dir   = Path(args.out_dir) if args.out_dir else Path(cfg.paths.outputs.results).parent / "mentor_review" / "similarity_metrics"

    run(ckpt_path, out_dir, cfg, args.n_generated, args.seed, log)
    print(f"✓ Similarity metrics written to {out_dir}")


if __name__ == "__main__":
    main()
