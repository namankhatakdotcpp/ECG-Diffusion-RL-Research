"""
mentor_eval/disease_similarity_table.py — mentor-requested disease-wise
comparison table: Cosine Similarity, Mahalanobis Distance, Hausdorff
Distance, and a 4th metric (provisional Bhattacharyya, NOT confirmed by
Dr. Balaji as of 2026-07-07 -- see 4th-metric caveat below), one row per
disease class.

Reuses mentor_eval.similarity_metrics's existing functions (mahalanobis_
distance, bhattacharyya_distance, matched_cosine_similarity, extract_
features) and mentor_eval.hausdorff_distance's matched_hausdorff --
no metric is reimplemented here, only assembled into the mentor's
requested table shape. Same real-data-loading / generated-sample
convention as similarity_metrics.py (reused, not duplicated).

4th-metric caveat (pre-registered 2026-07-07, stated here rather than
silently resolved): the 07-07-2026 sync-up document's table template
lists a 4th metric column as "to be finalized as per project
requirements, e.g., Peak-to-Peak or Drop Distance" -- that is a
suggestion in the source document, not a confirmation from Dr. Balaji.
Bhattacharyya distance is computed as a PROVISIONAL placeholder (already
in the pipeline, zero marginal cost) -- the table and report both label
this column "4th Metric (PENDING CONFIRMATION -- provisional:
Bhattacharyya)" rather than asserting it as final.

Healthy Sinus caveat: the mentor's table template includes a "Healthy
Sinus" row distinct from "Normal". This project's class taxonomy
(mentor_eval.class_mapping.MENTOR_CLASSES) has no such class -- PTB-XL's
scp_statements.csv DOES have a separate "SR" (sinus rhythm) code under a
different statement category (rhythm) from "NORM" (diagnostic), so
"Healthy Sinus" plausibly means "diagnostically Normal AND confirmed
sinus rhythm" -- a stricter subset of Normal, not the same thing. NOT
silently merged into the Normal row here -- reported as its own row,
explicitly marked pending clarification.

REQUIRES a trained checkpoint (outputs/models/diffusion_best.pt or a
Stage 3 candidate's checkpoint via --ckpt) -- same as similarity_metrics.py,
not available on this local machine; write-only deliverable until run on
the GPU server.

Writes:
  <out-dir>/disease_similarity_table.csv
  <out-dir>/disease_similarity_table.md

Usage:
    python -m mentor_eval.disease_similarity_table [--ckpt PATH] [--out-dir PATH] [--n-generated 200]
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
    MENTOR_CLASSES, MENTOR_TO_TRAINED_CLASS, load_ptbxl_database, filter_to_mentor_classes,
)
from mentor_eval.checkpoint_utils import load_checkpoint, generate_for_class
from mentor_eval.similarity_metrics import (
    extract_features, mahalanobis_distance, bhattacharyya_distance,
    matched_cosine_similarity, MIN_SAMPLES_FOR_COVARIANCE_MULTIPLIER,
)
from mentor_eval.hausdorff_distance import matched_hausdorff

FOURTH_METRIC_LABEL = "4th Metric (PENDING CONFIRMATION -- provisional: Bhattacharyya)"

HEALTHY_SINUS_ROW = {
    "Disease": "Healthy Sinus",
    "Cosine Similarity": None,
    "Mahalanobis Distance": None,
    "Hausdorff Distance": None,
    FOURTH_METRIC_LABEL: None,
    "Remarks": (
        "PENDING -- clarify with Dr. Balaji whether distinct from Normal. "
        "This project's class taxonomy (mentor_eval.class_mapping.MENTOR_CLASSES) "
        "has no separate Healthy Sinus class; PTB-XL has an 'SR' (sinus rhythm) "
        "code under a different statement category than 'NORM', so this may be "
        "a stricter subset of Normal, not a synonym -- not computed, not merged "
        "into the Normal row."
    ),
}


def run(ckpt_path: Path, out_dir: Path, cfg, n_generated: int, seed: int, log) -> pd.DataFrame:
    loaded = load_checkpoint(ckpt_path, cfg)
    if loaded is None:
        print(
            f"\n[BLOCKED] No checkpoint found at {ckpt_path}.\n"
            f"  Disease-wise similarity table compares real vs. GENERATED ECGs --\n"
            f"  there is nothing to compare without the trained model. Train on the\n"
            f"  GPU server, then re-run this script there.\n"
            f"  No metrics were computed -- nothing fabricated.\n"
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
        if cls == "AFIB":
            rows.append({
                "Disease": cls, "Cosine Similarity": None, "Mahalanobis Distance": None,
                "Hausdorff Distance": None, FOURTH_METRIC_LABEL: None,
                "Remarks": "NOT AVAILABLE -- no generated samples for this class "
                           "(no trained-model class; established project-wide convention, "
                           "not substituted with OTHER).",
            })
            continue

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
                "Disease": cls, "Cosine Similarity": None, "Mahalanobis Distance": None,
                "Hausdorff Distance": None, FOURTH_METRIC_LABEL: None,
                "Remarks": f"No generated samples available ('{cls}' has no trained-model class)"
                           if trained_cls is None else "No readable real samples found",
            })
            continue

        gen_signals, err = generate_for_class(
            loaded, trained_cls, n_samples=n_generated, cfg=cfg, seed=seed, stats=prep_stats,
        )
        if err:
            rows.append({
                "Disease": cls, "Cosine Similarity": None, "Mahalanobis Distance": None,
                "Hausdorff Distance": None, FOURTH_METRIC_LABEL: None, "Remarks": err,
            })
            continue

        hausdorff_val = round(matched_hausdorff(real_signals, gen_signals), 4)
        cosine_val = round(matched_cosine_similarity(real_signals, gen_signals), 4)

        real_feats = extract_features(real_signals)
        gen_feats = extract_features(gen_signals)
        min_required = MIN_SAMPLES_FOR_COVARIANCE_MULTIPLIER * real_feats.shape[1]
        if len(real_feats) < min_required:
            rows.append({
                "Disease": cls, "Cosine Similarity": cosine_val, "Mahalanobis Distance": None,
                "Hausdorff Distance": hausdorff_val, FOURTH_METRIC_LABEL: None,
                "Remarks": (
                    f"Only {len(real_feats)} real samples (<{min_required} required for stable "
                    f"48-dim covariance) -- Mahalanobis/Bhattacharyya skipped, not fabricated."
                ),
            })
            continue

        rows.append({
            "Disease": cls,
            "Cosine Similarity": cosine_val,
            "Mahalanobis Distance": round(mahalanobis_distance(gen_feats, real_feats), 4),
            "Hausdorff Distance": hausdorff_val,
            FOURTH_METRIC_LABEL: round(bhattacharyya_distance(real_feats, gen_feats), 4),
            "Remarks": "",
        })
        log.info(f"{cls}: computed disease-wise metrics (n_real={len(real_signals)}, n_gen={len(gen_signals)})")

    rows.append(HEALTHY_SINUS_ROW)

    out_df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_dir / "disease_similarity_table.csv", index=False)

    header = list(out_df.columns)
    table_lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for _, row in out_df.iterrows():
        cells = ["--" if pd.isna(v) or v is None else str(v) for v in row]
        table_lines.append("| " + " | ".join(cells) + " |")

    md_lines = [
        "# Disease-wise Similarity Comparison\n",
        "**4th-metric caveat**: the column below is provisional (Bhattacharyya "
        "distance, already computed elsewhere in this pipeline) -- NOT confirmed "
        "by Dr. Balaji as the intended 4th metric. The 2026-07-07 sync-up document "
        "lists this as \"to be finalized,\" not decided.\n",
        "**Hausdorff Distance caveat**: Hausdorff distance is included as a "
        "worst-case amplitude deviation metric. It complements cosine "
        "similarity, Mahalanobis distance, and Bhattacharyya distance by "
        "quantifying the maximum amplitude mismatch between matched real and "
        "generated ECGs. It is not intended to evaluate temporal alignment or "
        "waveform morphology.\n",
        "**Nearest-neighbour pairing**: generated ECGs are compared against "
        "their nearest-neighbour real ECG within the same disease class (via "
        "Euclidean distance in the 12000-dim raw waveform space) rather than "
        "random or index-aligned pairing. This reduces pairing bias and "
        "maintains consistency with the existing cosine similarity evaluation "
        "protocol in similarity_metrics.py, which uses the same matching "
        "convention.\n",
        "**Lead-averaging caveat**: the Hausdorff value reported per disease "
        "is a mean across 12 leads. Given this project's Stage 3 finding that "
        "disease-discriminative failure concentrates in specific frequency "
        "subbands and is most visually apparent in Lead V1, a per-lead "
        "breakdown may reveal amplitude-range anomalies this aggregate "
        "obscures -- see compute_hausdorff_per_lead() in hausdorff_distance.py.\n",
        "\n".join(table_lines),
    ]
    (out_dir / "disease_similarity_table.md").write_text("\n".join(md_lines) + "\n")

    print(out_df.to_string(index=False))
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Disease-wise similarity comparison table.")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--n-generated", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("disease_similarity_table", cfg=cfg)
    set_seed(args.seed)

    ckpt_path = Path(args.ckpt) if args.ckpt else Path(cfg.paths.outputs.models) / "diffusion_best.pt"
    out_dir = (Path(args.out_dir) if args.out_dir
               else Path(cfg.paths.outputs.results).parent / "mentor_review" / "disease_similarity_table")
    snapshot_before_write(out_dir)
    run(ckpt_path, out_dir, cfg, args.n_generated, args.seed, log)
    print(f"Disease-wise similarity table written to {out_dir}")


if __name__ == "__main__":
    main()
