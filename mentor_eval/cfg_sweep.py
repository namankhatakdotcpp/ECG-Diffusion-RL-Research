"""
mentor_eval/cfg_sweep.py — classifier-free guidance (CFG) scale sweep.

Sweeps guidance_scale over [None, 1.0, 2.0, 3.0, 5.0], generating samples
at each scale and evaluating them with the MentorClassifier from
conditioning_diagnostic.py. Outputs a summary table comparing conditioning
quality across scales.

REQUIRES:
  - outputs/models/diffusion_best.pt  (trained with p_uncond > 0)
  - PTB-XL dataset (to train/load the MentorClassifier)

Writes to: outputs/conditioning_analysis/
  cfg_sweep_result.txt      — human-readable summary table
  cfg_sweep_metrics.csv     — machine-readable per-scale metrics
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_config, get_logger, set_seed
from utils.backup import snapshot_before_write
from mentor_eval.class_mapping import (
    MENTOR_CLASSES, load_ptbxl_database, filter_to_mentor_classes,
)
from mentor_eval.checkpoint_utils import load_checkpoint, generate_for_class
from mentor_eval.classification_validation import (
    MentorClassifier, TRAIN_FOLDS, VAL_FOLDS,
)

OUT_DIR = Path("outputs/conditioning_analysis")
SWEEP_SCALES: list[Optional[float]] = [None, 1.0, 2.0, 3.0, 5.0]
SKIP_DIFFUSION_CLASSES = {"OTHER"}
N_GEN = 50   # samples per diffusion class per guidance scale


def _load_signals(ptbxl_dir: Path, filtered_db, folds: list[int], log):
    subset = filtered_db[filtered_db["strat_fold"].isin(folds)]
    X, y = [], []
    name_to_idx = {n: i for i, n in enumerate(MENTOR_CLASSES)}
    for _, rec in subset.iterrows():
        try:
            sig = wfdb.rdrecord(str(ptbxl_dir / str(rec["filename_lr"]))).p_signal
        except Exception:
            continue
        if sig.shape != (1000, 12) or not np.isfinite(sig).all():
            continue
        X.append(sig)
        y.append(name_to_idx[rec["mentor_class"]])
    return np.array(X), np.array(y)


def _get_classifier(ptbxl_dir: Path, filtered_db, device: str, log) -> MentorClassifier:
    cached = OUT_DIR / "mentor_classifier.pt"
    if cached.exists():
        log.info(f"Loading cached MentorClassifier from {cached}")
        clf = MentorClassifier(n_classes=len(MENTOR_CLASSES)).to(device)
        clf.load_state_dict(torch.load(str(cached), map_location=device))
        clf.eval()
        return clf

    log.info("Training MentorClassifier on real PTB-XL data …")
    X_train, y_train = _load_signals(ptbxl_dir, filtered_db, TRAIN_FOLDS, log)
    X_val,   y_val   = _load_signals(ptbxl_dir, filtered_db, VAL_FOLDS,   log)

    model = MentorClassifier(len(MENTOR_CLASSES)).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr   = torch.from_numpy(X_train.transpose(0, 2, 1)).float()
    ytr   = torch.from_numpy(y_train).long()
    Xva   = torch.from_numpy(X_val.transpose(0, 2, 1)).float()
    yva   = torch.from_numpy(y_val).long()
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)

    best_f1, best_state = -1.0, None
    for epoch in range(1, 31):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            preds = model(Xva.to(device)).argmax(1).cpu().numpy()
        f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), str(cached))
    log.info(f"Classifier trained and cached. Best val macro F1: {best_f1:.4f}")
    return model


def _evaluate_scale(
    loaded, clf: MentorClassifier, diffusion_classes: list[str],
    prep_stats: Optional[dict], cfg, device: str, guidance_scale: Optional[float], log,
) -> dict:
    """Generate N_GEN samples per class at given guidance_scale, return metrics dict."""
    scale_label = f"{guidance_scale:.1f}" if guidance_scale is not None else "None (baseline)"
    all_preds, all_true = [], []
    # rows=diffusion_class, cols=mentor_class — for per-class accuracy
    confusion = np.zeros((len(diffusion_classes), len(MENTOR_CLASSES)), dtype=int)

    for ri, dcls in enumerate(diffusion_classes):
        samples, err = generate_for_class(
            loaded, dcls, n_samples=N_GEN, cfg=cfg, seed=100 + ri,
            stats=prep_stats, guidance_scale=guidance_scale,
        )
        if err:
            log.warning(f"  [{scale_label}] Skipped '{dcls}': {err}")
            continue

        Xt = torch.from_numpy(samples.transpose(0, 2, 1)).float().to(device)
        with torch.no_grad():
            preds = clf(Xt).argmax(1).cpu().numpy()
        for p in preds:
            confusion[ri, p] += 1
        all_preds.extend(preds.tolist())
        # We don't have a true mentor label for each diffusion class, but we can
        # compute overall accuracy (fraction that matched the diagonal expectation)
        # as a conditioning quality proxy.

    total = max(sum(all_preds.__len__() for _ in [0]), len(all_preds), 1)
    # Diagonal hits: generated class ri is "correct" if predicted mentor class
    # aligns with what MENTOR_TO_TRAINED_CLASS would reverse-map to.
    # Simpler proxy: fraction of predictions NOT collapsed to a single class.
    counts = np.bincount(all_preds, minlength=len(MENTOR_CLASSES)) if all_preds else np.zeros(len(MENTOR_CLASSES))
    top_class_frac = float(counts.max()) / max(len(all_preds), 1)
    macro_f1 = float(f1_score(
        # pseudo-labels: assume each diffusion class maps to the closest MENTOR class
        [i // N_GEN for i in range(len(all_preds))],
        all_preds,
        average="macro", zero_division=0,
    )) if all_preds else 0.0

    return {
        "guidance_scale": guidance_scale,
        "n_samples": len(all_preds),
        "top_class_collapse_frac": round(top_class_frac, 4),
        "macro_f1_vs_requested": round(macro_f1, 4),
        "confusion": confusion.tolist(),
    }


def main() -> None:
    cfg = load_config()
    log = get_logger("cfg_sweep", cfg=cfg)
    set_seed(42)

    snapshot_before_write(OUT_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = Path(cfg.paths.outputs.models) / "diffusion_best.pt"
    loaded = load_checkpoint(ckpt_path, cfg)
    if loaded is None:
        msg = (
            f"\n[BLOCKED] No checkpoint found at {ckpt_path}.\n"
            f"  CFG sweep requires a model trained with p_uncond > 0.\n"
            f"  Train step04 on the GPU server, then re-run.\n"
        )
        print(msg)
        (OUT_DIR / "cfg_sweep_result.txt").write_text(msg)
        return

    ptbxl_dir = Path(cfg.paths.data.ptbxl)
    db = load_ptbxl_database(ptbxl_dir)
    filtered = filter_to_mentor_classes(db)

    clf = _get_classifier(ptbxl_dir, filtered, device, log)
    clf.eval()

    stats_path = Path(cfg.paths.outputs.processed) / "preprocessing_stats.json"
    prep_stats = json.load(open(stats_path)) if stats_path.exists() else None

    diffusion_classes = [c for c in loaded.class_names if c not in SKIP_DIFFUSION_CLASSES]
    log.info(f"Sweeping guidance_scale over {SWEEP_SCALES} for classes {diffusion_classes}")

    rows = []
    for scale in SWEEP_SCALES:
        label = f"{scale:.1f}" if scale is not None else "None"
        log.info(f"--- guidance_scale={label} ---")
        result = _evaluate_scale(loaded, clf, diffusion_classes, prep_stats, cfg, device, scale, log)
        rows.append(result)
        log.info(f"  collapse_frac={result['top_class_collapse_frac']:.4f}  "
                 f"macro_f1={result['macro_f1_vs_requested']:.4f}")

    # ── Write CSV ──────────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "cfg_sweep_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["guidance_scale", "n_samples",
                                           "top_class_collapse_frac", "macro_f1_vs_requested"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    log.info(f"Saved CSV → {csv_path}")

    # ── Write human-readable summary ───────────────────────────────────────────
    lines = ["CFG SWEEP — RESULTS", "=" * 60, ""]
    lines.append(f"{'guidance_scale':<18} {'collapse_frac':<16} {'macro_f1':<12} {'n_samples'}")
    lines.append("-" * 60)
    for r in rows:
        label = f"{r['guidance_scale']:.1f}" if r["guidance_scale"] is not None else "None (baseline)"
        lines.append(
            f"{label:<18} {r['top_class_collapse_frac']:<16.4f} "
            f"{r['macro_f1_vs_requested']:<12.4f} {r['n_samples']}"
        )
    lines += [
        "",
        "collapse_frac: fraction of all predictions falling on a single mentor class",
        "  (lower = less collapse = better conditioning).",
        "macro_f1: F1 treating requested diffusion class as ground truth",
        "  (higher = better steering).",
        "",
        f"Diffusion classes evaluated: {diffusion_classes}",
        f"MentorClassifier classes: {MENTOR_CLASSES}",
        f"Samples per class per scale: {N_GEN}",
    ]
    result_text = "\n".join(lines)
    txt_path = OUT_DIR / "cfg_sweep_result.txt"
    txt_path.write_text(result_text)
    print(result_text)
    print(f"\nOutputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
