"""
mentor_eval/conditioning_diagnostic.py — measure whether diffusion class
conditioning is steering generation.

For each trained diffusion class (NORM, MI, STTC, CD, HYP; skip OTHER),
generate N=50 samples and run them through the MentorClassifier trained on
real PTB-XL data (same classifier as classification_validation.py).

Output: confusion-style table where rows = requested diffusion class and
columns = MentorClassifier-predicted mentor class (Normal/STEMI/NSTEMI/AFIB).
If conditioning works, rows should be diagonal-ish. If collapsed, every row
predicts the same column.

Requires:
  - PTB-XL dataset (to train the MentorClassifier inline)
  - outputs/models/diffusion_best.pt  (GPU server — BLOCKED without it)

Writes to: outputs/conditioning_analysis/
  conditioning_confusion.csv
  conditioning_heatmap.png
  mentor_classifier.pt          (cached for embedding_visualization.py)
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
import torch.nn.functional as F
import wfdb
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_config, get_logger, set_seed
from mentor_eval.class_mapping import (
    MENTOR_CLASSES, load_ptbxl_database, filter_to_mentor_classes,
)
from mentor_eval.checkpoint_utils import load_checkpoint, generate_for_class
from mentor_eval.classification_validation import (
    MentorClassifier, _ConvBlock1D,
    TRAIN_FOLDS, VAL_FOLDS,
)

# Diffusion model's trained classes, in the order they typically appear in
# class_names.json (OTHER excluded — the model has no meaningful conditioning signal for it)
SKIP_DIFFUSION_CLASSES = {"OTHER"}

OUT_DIR = Path("outputs/conditioning_analysis")


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
    log.info(f"Loaded {len(X)} records for folds {folds}")
    return np.array(X), np.array(y)


def _train_classifier(X_train, y_train, X_val, y_val, device, log, n_classes=4, n_epochs=30):
    model = MentorClassifier(n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    Xtr = torch.from_numpy(X_train.transpose(0, 2, 1)).float()
    ytr = torch.from_numpy(y_train).long()
    Xva = torch.from_numpy(X_val.transpose(0, 2, 1)).float()
    yva = torch.from_numpy(y_val).long()

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)
    best_f1, best_state = -1.0, None

    for epoch in range(1, n_epochs + 1):
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
        if epoch % 10 == 0:
            log.info(f"  classifier epoch {epoch}/{n_epochs}: val_macro_f1={f1:.4f}")

    model.load_state_dict(best_state)
    log.info(f"Classifier trained. Best val macro F1: {best_f1:.4f}")
    return model


def run(cfg, log) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)

    # ── Train MentorClassifier on real PTB-XL ──────────────────────────────────
    ptbxl_dir = Path(cfg.paths.data.ptbxl)
    cached_clf = OUT_DIR / "mentor_classifier.pt"

    db = load_ptbxl_database(ptbxl_dir)
    filtered = filter_to_mentor_classes(db)

    if cached_clf.exists():
        log.info(f"Loading cached MentorClassifier from {cached_clf}")
        clf = MentorClassifier(n_classes=len(MENTOR_CLASSES)).to(device)
        clf.load_state_dict(torch.load(str(cached_clf), map_location=device))
        clf.eval()
    else:
        log.info("Training MentorClassifier on real PTB-XL data …")
        X_train, y_train = _load_signals(ptbxl_dir, filtered, TRAIN_FOLDS, log)
        X_val,   y_val   = _load_signals(ptbxl_dir, filtered, VAL_FOLDS,   log)
        if len(X_train) == 0:
            print("[BLOCKED] No PTB-XL records readable — cannot train classifier.")
            return
        clf = _train_classifier(X_train, y_train, X_val, y_val, device, log)
        torch.save(clf.state_dict(), str(cached_clf))
        log.info(f"Classifier cached → {cached_clf}")

    # ── Load diffusion checkpoint ──────────────────────────────────────────────
    ckpt_path = Path(cfg.paths.outputs.models) / "diffusion_best.pt"
    loaded = load_checkpoint(ckpt_path, cfg)
    if loaded is None:
        print(
            f"\n[BLOCKED] Diffusion checkpoint not found at {ckpt_path}.\n"
            f"  MentorClassifier has been trained and cached at {cached_clf}.\n"
            f"  Re-run on GPU server after training step04.\n"
        )
        return

    diffusion_classes = [c for c in loaded.class_names if c not in SKIP_DIFFUSION_CLASSES]
    log.info(f"Diffusion model classes to probe: {diffusion_classes}")

    stats_path = Path(cfg.paths.outputs.processed) / "preprocessing_stats.json"
    prep_stats = json.load(open(stats_path)) if stats_path.exists() else None

    # ── Generate + classify ────────────────────────────────────────────────────
    N_GEN = 50
    # rows=diffusion_class, cols=mentor_class
    confusion = np.zeros((len(diffusion_classes), len(MENTOR_CLASSES)), dtype=int)

    clf.eval()
    for ri, dcls in enumerate(diffusion_classes):
        log.info(f"Generating {N_GEN} samples for diffusion class '{dcls}' …")
        samples, err = generate_for_class(
            loaded, dcls, n_samples=N_GEN, cfg=cfg, seed=100 + ri, stats=prep_stats,
        )
        if err:
            log.warning(f"  Skipped '{dcls}': {err}")
            continue

        # samples: (N_GEN, 1000, 12) → (N_GEN, 12, 1000)
        Xt = torch.from_numpy(samples.transpose(0, 2, 1)).float().to(device)
        with torch.no_grad():
            preds = clf(Xt).argmax(1).cpu().numpy()
        for p in preds:
            confusion[ri, p] += 1

        top_mentor = MENTOR_CLASSES[int(np.argmax(np.bincount(preds, minlength=len(MENTOR_CLASSES))))]
        log.info(f"  '{dcls}' → most predicted mentor class: {top_mentor}")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    import csv
    csv_path = OUT_DIR / "conditioning_confusion.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["diffusion_class"] + MENTOR_CLASSES)
        for ri, dcls in enumerate(diffusion_classes):
            w.writerow([dcls] + confusion[ri].tolist())
    log.info(f"Saved CSV → {csv_path}")

    # ── Save heatmap PNG ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, max(3, len(diffusion_classes) * 0.9)))
    im = ax.imshow(confusion, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(MENTOR_CLASSES)))
    ax.set_xticklabels(MENTOR_CLASSES, fontsize=10)
    ax.set_yticks(range(len(diffusion_classes)))
    ax.set_yticklabels(diffusion_classes, fontsize=10)
    ax.set_xlabel("MentorClassifier predicted class", fontsize=11)
    ax.set_ylabel("Requested diffusion class", fontsize=11)
    ax.set_title(
        f"Conditioning diagnostic — {N_GEN} samples per diffusion class\n"
        "Diagonal = conditioning working; single column = conditioning collapsed",
        fontsize=10,
    )
    for ri in range(len(diffusion_classes)):
        for ci in range(len(MENTOR_CLASSES)):
            ax.text(ci, ri, str(confusion[ri, ci]), ha="center", va="center",
                    fontsize=11, color="black" if confusion[ri, ci] < confusion.max() * 0.6 else "white")
    fig.colorbar(im, ax=ax, shrink=0.8, label="count")
    fig.tight_layout()
    png_path = OUT_DIR / "conditioning_heatmap.png"
    fig.savefig(str(png_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved heatmap → {png_path}")

    # ── Console summary ────────────────────────────────────────────────────────
    print("\nConditioning diagnostic results")
    print("=" * 60)
    col_w = 10
    header = "Diffusion →".ljust(14) + "".join(c.ljust(col_w) for c in MENTOR_CLASSES)
    print(header)
    for ri, dcls in enumerate(diffusion_classes):
        row = dcls.ljust(14) + "".join(str(confusion[ri, ci]).ljust(col_w) for ci in range(len(MENTOR_CLASSES)))
        print(row)
    print(f"\nOutputs written to {OUT_DIR}/")


def main() -> None:
    cfg = load_config()
    log = get_logger("conditioning_diagnostic", cfg=cfg)
    run(cfg, log)


if __name__ == "__main__":
    main()
