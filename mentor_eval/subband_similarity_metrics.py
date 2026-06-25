"""
mentor_eval/subband_similarity_metrics.py — Mahalanobis / Bhattacharyya /
cosine similarity, computed PER SUBBAND instead of on the aggregate 48-dim
raw-statistical feature space used by mentor_eval/similarity_metrics.py
(item 8).

Reuses mahalanobis_distance, bhattacharyya_distance, and
matched_cosine_similarity from mentor_eval.similarity_metrics directly (no
reimplementation) — only the feature extraction changes, from per-lead
[mean, std, skew, kurtosis] to per-lead subband energy (one subband at a
time, 12-dim per subband, per mentor_eval.subband_features).

This answers a question the aggregate metric in item 8 couldn't: does the
model match real ECGs well in some subbands (e.g. QRS-dominant D2) but
poorly in others (e.g. slow-wave A3)? The summary explicitly names the
worst-matching (subband, class) pair.

REQUIRES outputs/models/diffusion_best.pt — prints [BLOCKED] and writes
nothing if absent.

Writes:
  outputs/sharma_inspired_analysis/subband_similarity_metrics.csv
  outputs/sharma_inspired_analysis/subband_similarity_README.md

Usage:
    python -m mentor_eval.subband_similarity_metrics [--ckpt PATH] [--out-dir PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_config, get_logger, set_seed
from mentor_eval.class_mapping import (
    MENTOR_TO_TRAINED_CLASS, load_ptbxl_database, filter_to_mentor_classes,
)
from mentor_eval.checkpoint_utils import load_checkpoint, generate_for_class
from mentor_eval.similarity_metrics import (
    mahalanobis_distance, bhattacharyya_distance, matched_cosine_similarity,
    MIN_SAMPLES_FOR_COVARIANCE_MULTIPLIER,
)
from mentor_eval.subband_features import (
    SUBBAND_NAMES, SUBBAND_CLINICAL_LABEL, extract_subband_energy_batch, subband_output_dir,
)

BOX_CLASSES = ["Normal", "STEMI", "NSTEMI"]  # AFIB excluded - no trained model class


def _load_real_signals_for_class(candidates, ptbxl_dir: Path, n: int, rng: np.random.Generator) -> np.ndarray:
    order = rng.permutation(len(candidates))
    sigs = []
    for idx in order:
        if len(sigs) >= n:
            break
        rec = candidates.iloc[int(idx)]
        try:
            sig = wfdb.rdrecord(str(ptbxl_dir / str(rec["filename_lr"]))).p_signal
        except Exception:
            continue
        if sig.shape == (1000, 12) and np.isfinite(sig).all():
            sigs.append(sig)
    return np.array(sigs)


def _subband_only_features(full_feats: np.ndarray, band_idx: int, n_leads: int = 12) -> np.ndarray:
    """full_feats: (N, len(SUBBAND_NAMES)*12) -> (N, 12) slice for one subband."""
    return full_feats[:, band_idx * n_leads:(band_idx + 1) * n_leads]


def run(ckpt_path: Path, out_dir: Path, cfg, n_per_class: int, seed: int, log) -> pd.DataFrame:
    loaded = load_checkpoint(ckpt_path, cfg)
    if loaded is None:
        print(
            f"\n[BLOCKED] No checkpoint found at {ckpt_path}.\n"
            f"  Subband similarity metrics compare real vs. GENERATED subband energy —\n"
            f"  there is nothing to compare without the trained model. Train on the GPU\n"
            f"  server, then re-run this script there.\n"
            f"  No metrics were computed — nothing fabricated.\n"
        )
        sys.exit(1)

    ptbxl_dir = Path(cfg.paths.data.ptbxl)
    db = load_ptbxl_database(ptbxl_dir)
    filtered = filter_to_mentor_classes(db)
    rng = np.random.default_rng(seed)

    stats_path = Path(cfg.paths.outputs.processed) / "preprocessing_stats.json"
    prep_stats = None
    if stats_path.exists():
        import json
        prep_stats = json.load(open(stats_path))

    rows = []
    for cls in BOX_CLASSES:
        candidates = filtered[filtered["mentor_class"] == cls]
        real_signals = _load_real_signals_for_class(candidates, ptbxl_dir, n_per_class, rng)
        trained_cls = MENTOR_TO_TRAINED_CLASS.get(cls)

        if len(real_signals) == 0 or trained_cls is None:
            for band in SUBBAND_NAMES:
                rows.append({
                    "class": cls, "subband": band, "n_real": len(real_signals), "n_generated": 0,
                    "mahalanobis": None, "bhattacharyya": None, "cosine_similarity": None,
                    "flag": "No readable real samples" if len(real_signals) == 0 else f"'{cls}' has no trained-model class",
                })
            continue

        gen_signals, err = generate_for_class(loaded, trained_cls, n_samples=n_per_class, cfg=cfg, seed=seed, stats=prep_stats)
        if err:
            for band in SUBBAND_NAMES:
                rows.append({
                    "class": cls, "subband": band, "n_real": len(real_signals), "n_generated": 0,
                    "mahalanobis": None, "bhattacharyya": None, "cosine_similarity": None, "flag": err,
                })
            continue

        real_full = extract_subband_energy_batch(real_signals)   # (N, 4*12)
        gen_full = extract_subband_energy_batch(gen_signals)

        min_required = MIN_SAMPLES_FOR_COVARIANCE_MULTIPLIER * 12  # 12-dim per-subband feature
        for band_idx, band in enumerate(SUBBAND_NAMES):
            real_feats = _subband_only_features(real_full, band_idx)
            gen_feats = _subband_only_features(gen_full, band_idx)

            if len(real_feats) < min_required:
                rows.append({
                    "class": cls, "subband": band, "n_real": len(real_feats), "n_generated": len(gen_feats),
                    "mahalanobis": None, "bhattacharyya": None,
                    "cosine_similarity": round(matched_cosine_similarity(real_signals, gen_signals), 4),
                    "flag": f"Only {len(real_feats)} real samples (<{min_required} required) — Mahalanobis/Bhattacharyya skipped.",
                })
                continue

            rows.append({
                "class": cls, "subband": band, "n_real": len(real_feats), "n_generated": len(gen_feats),
                "mahalanobis": round(mahalanobis_distance(gen_feats, real_feats), 4),
                "bhattacharyya": round(bhattacharyya_distance(real_feats, gen_feats), 4),
                "cosine_similarity": round(matched_cosine_similarity(real_signals, gen_signals), 4),
                "flag": "",
            })
        log.info(f"{cls}: computed subband similarity metrics for all {len(SUBBAND_NAMES)} subbands")

    out_df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_dir / "subband_similarity_metrics.csv", index=False)

    # Explicit "largest divergence" callout — the new, more diagnostic question.
    valid = out_df.dropna(subset=["mahalanobis"])
    callout = "Not enough valid (class, subband) pairs to identify a largest-divergence callout."
    if not valid.empty:
        worst = valid.loc[valid["mahalanobis"].idxmax()]
        callout = (
            f"Largest real-vs-generated divergence: subband {worst['subband']} "
            f"({SUBBAND_CLINICAL_LABEL[worst['subband']]}) for class {worst['class']} "
            f"(Mahalanobis={worst['mahalanobis']:.4f}). "
            f"The model matches real ECGs least well there relative to other (class, subband) pairs."
        )
    log.info(callout)
    print(callout)

    readme = f"""# How to read subband-level similarity metrics

Same metrics as `similarity_metrics/README.md` (Mahalanobis: lower=better;
Bhattacharyya: lower=better; cosine similarity: higher=better, range [-1,1]),
but computed independently PER SUBBAND instead of one aggregate number per
class. This answers: does the generator match real ECGs well in some
frequency content (e.g. QRS-dominant D2) but poorly in others (e.g. slow-wave
A3)?

Subbands (see `mentor_eval/subband_features.py` for the full derivation):
{chr(10).join(f"- **{b}**: {SUBBAND_CLINICAL_LABEL[b]}" for b in SUBBAND_NAMES)}

{callout}
"""
    (out_dir / "subband_similarity_README.md").write_text(readme)

    print(out_df.to_string(index=False))
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Subband-level Mahalanobis/Bhattacharyya/cosine similarity.")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--n-per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("subband_similarity_metrics", cfg=cfg)
    set_seed(args.seed)

    ckpt_path = Path(args.ckpt) if args.ckpt else Path(cfg.paths.outputs.models) / "diffusion_best.pt"
    out_dir = Path(args.out_dir) if args.out_dir else subband_output_dir(cfg)

    run(ckpt_path, out_dir, cfg, args.n_per_class, args.seed, log)
    print(f"✓ Subband similarity metrics written to {out_dir}")


if __name__ == "__main__":
    main()
