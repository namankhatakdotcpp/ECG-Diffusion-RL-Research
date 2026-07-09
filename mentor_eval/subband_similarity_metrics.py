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
from utils.backup import snapshot_before_write
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


def _load_real_signals_for_class(
    candidates, ptbxl_dir: Path, n: int, rng: np.random.Generator,
) -> tuple[np.ndarray, list]:
    """Returns (signals, ecg_ids) — ecg_ids retained so a disjointness check
    is possible when the same pool is later split (see --self-check)."""
    order = rng.permutation(len(candidates))
    sigs, ids = [], []
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
            ids.append(rec.name)   # ecg_id (candidates is indexed by ecg_id)
    return np.array(sigs), ids


def _subband_only_features(full_feats: np.ndarray, band_idx: int, n_leads: int = 12) -> np.ndarray:
    """full_feats: (N, len(SUBBAND_NAMES)*12) -> (N, 12) slice for one subband."""
    return full_feats[:, band_idx * n_leads:(band_idx + 1) * n_leads]


def _self_check_one_class(
    cls: str, real_signals: np.ndarray, real_ids: list, rng: np.random.Generator, log,
) -> list[dict]:
    """
    Split the SAME real pool already drawn for the generated-vs-real
    comparison into two disjoint halves and compute the identical
    Mahalanobis/Bhattacharyya/cosine metrics between them — the real-vs-real
    noise floor, using the same functions and same code path as the
    generated-vs-real comparison (mahalanobis_distance/bhattacharyya_
    distance/matched_cosine_similarity, imported not reimplemented).

    Purely additive: does not touch or alter the generated-vs-real rows
    computed in run(); reuses the same real_signals array already loaded.
    """
    rows: list[dict] = []
    n = len(real_signals)
    order = rng.permutation(n)
    half = n // 2
    idx_a, idx_b = order[:half], order[half:2 * half]

    ids_a = {real_ids[i] for i in idx_a}
    ids_b = {real_ids[i] for i in idx_b}
    overlap = ids_a & ids_b
    if overlap:
        raise RuntimeError(
            f"_self_check_one_class({cls!r}): {len(overlap)} record ID(s) appear in "
            "BOTH halves — the split is not disjoint, self-check would be comparing "
            "a set against (partially) itself. Refusing to report a fabricated "
            "noise-floor number. This indicates a bug in the split logic above."
        )

    real_a, real_b = real_signals[idx_a], real_signals[idx_b]
    feats_a_full = extract_subband_energy_batch(real_a)
    feats_b_full = extract_subband_energy_batch(real_b)

    min_required = MIN_SAMPLES_FOR_COVARIANCE_MULTIPLIER * 12
    for band_idx, band in enumerate(SUBBAND_NAMES):
        feats_a = _subband_only_features(feats_a_full, band_idx)
        feats_b = _subband_only_features(feats_b_full, band_idx)

        if len(feats_a) < min_required or len(feats_b) < min_required:
            rows.append({
                "class": cls, "subband": band, "n_half_a": len(feats_a), "n_half_b": len(feats_b),
                "mahalanobis": None, "bhattacharyya": None, "cosine_similarity": None,
                "flag": f"Only {min(len(feats_a), len(feats_b))} samples in one half "
                        f"(<{min_required} required) — Mahalanobis/Bhattacharyya skipped.",
            })
            continue

        rows.append({
            "class": cls, "subband": band, "n_half_a": len(feats_a), "n_half_b": len(feats_b),
            "mahalanobis": round(mahalanobis_distance(feats_b, feats_a), 4),
            "bhattacharyya": round(bhattacharyya_distance(feats_a, feats_b), 4),
            "cosine_similarity": round(matched_cosine_similarity(real_a, real_b), 4),
            "flag": "",
        })
    log.info(f"{cls}: self-check (real-vs-real, disjoint halves n={half}+{n - half}) computed for all {len(SUBBAND_NAMES)} subbands")
    return rows


def run(ckpt_path: Path, out_dir: Path, cfg, n_per_class: int, seed: int, log, self_check: bool = False) -> pd.DataFrame:
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
    self_check_rows = []
    for cls in BOX_CLASSES:
        candidates = filtered[filtered["mentor_class"] == cls]
        real_signals, real_ids = _load_real_signals_for_class(candidates, ptbxl_dir, n_per_class, rng)
        trained_cls = MENTOR_TO_TRAINED_CLASS.get(cls)

        if self_check and len(real_signals) > 0:
            # Same real pool this class already drew for the generated-vs-real
            # comparison below, split into two disjoint halves — the honest
            # real-vs-real noise floor at this sample size, not a fresh/larger
            # draw. See _self_check_one_class docstring.
            self_check_rows.extend(_self_check_one_class(cls, real_signals, real_ids, rng, log))

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

    if self_check:
        self_df = pd.DataFrame(self_check_rows)
        self_df.to_csv(out_dir / "subband_self_comparison.csv", index=False)

        self_valid = self_df.dropna(subset=["mahalanobis"])
        self_callout = "Not enough valid (class, subband) pairs for a real-vs-real callout."
        if not self_valid.empty:
            worst_self = self_valid.loc[self_valid["mahalanobis"].idxmax()]
            self_callout = (
                f"Largest REAL-vs-REAL divergence (noise floor): subband {worst_self['subband']} "
                f"({SUBBAND_CLINICAL_LABEL[worst_self['subband']]}) for class {worst_self['class']} "
                f"(Mahalanobis={worst_self['mahalanobis']:.4f})."
            )
        log.info(self_callout)
        print("\n" + "=" * 70)
        print("REAL-vs-REAL SELF-CHECK (noise floor — same real pool, disjoint halves)")
        print("=" * 70)
        print(self_callout)
        print(
            "\nThis is NOT a generated-vs-real comparison — it's the same real "
            "class distribution split in half and compared against itself, using "
            "the identical Mahalanobis/Bhattacharyya/cosine functions above. "
            "Compare these numbers directly against the matching (class, subband) "
            "rows in subband_similarity_metrics.csv: if generated-vs-real is not "
            "clearly larger than this real-vs-real noise floor, that divergence "
            "is not distinguishable from sampling noise at this sample size."
        )
        print(self_df.to_string(index=False))
        (out_dir / "subband_self_comparison_README.md").write_text(
            "# Real-vs-real self-check (noise floor)\n\n"
            "Same real class pool already drawn for the generated-vs-real "
            "comparison, split into two disjoint halves (record IDs verified "
            "non-overlapping — see subband_similarity_metrics.py's "
            "_self_check_one_class), scored with the SAME "
            "Mahalanobis/Bhattacharyya/cosine functions used for "
            "generated-vs-real in subband_similarity_metrics.csv.\n\n"
            "Purpose: a generated-vs-real divergence number means nothing on its "
            "own — it needs a noise floor to compare against. If "
            "generated-vs-real Mahalanobis for a (class, subband) pair is not "
            "meaningfully larger than the real-vs-real number here for the same "
            "pair, that divergence may just be sampling noise, not a real gap "
            "between the model and real ECGs.\n\n"
            f"{self_callout}\n"
        )

    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Subband-level Mahalanobis/Bhattacharyya/cosine similarity.")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--n-per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--self-check", action="store_true",
        help="Also compute a real-vs-real noise floor: split the same real "
             "pool into two disjoint halves and score them against each "
             "other with the same Mahalanobis/Bhattacharyya/cosine metrics. "
             "Writes subband_self_comparison.csv alongside the normal output.",
    )
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("subband_similarity_metrics", cfg=cfg)
    set_seed(args.seed)

    ckpt_path = Path(args.ckpt) if args.ckpt else Path(cfg.paths.outputs.models) / "diffusion_best.pt"
    out_dir = Path(args.out_dir) if args.out_dir else subband_output_dir(cfg)
    snapshot_before_write(out_dir)
    run(ckpt_path, out_dir, cfg, args.n_per_class, args.seed, log, self_check=args.self_check)
    print(f"✓ Subband similarity metrics written to {out_dir}")


if __name__ == "__main__":
    main()
