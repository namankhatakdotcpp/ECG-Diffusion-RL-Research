"""
mentor_eval/classification_validation.py — downstream classification
validation for the 4 mentor-review classes.

Two stages:
  1. Train a 12-lead CNN classifier (same architecture as Simple1DCNN in
     step05_baseline_eval.py) on REAL PTB-XL data for Normal/STEMI/NSTEMI/
     AFIB, using the official strat_fold train/val/test split. This stage
     needs no diffusion checkpoint and runs today.
  2. Evaluate that classifier on model-GENERATED ECGs: accuracy, confusion
     matrix, ROC+AUC (per-class and macro). This stage REQUIRES
     outputs/models/diffusion_best.pt, which doesn't exist on this local
     machine. AFIB additionally has no generation path at all (see
     mentor_eval/class_mapping.py — model never learned a distinct AFIB
     class), so AFIB is excluded from stage 2 regardless of checkpoint
     availability, and that's flagged in the output rather than silently
     dropped.

Writes:
  outputs/mentor_review/classification_validation/classifier_real_eval.json
  outputs/mentor_review/classification_validation/confusion_matrix_real.png
  outputs/mentor_review/classification_validation/ (stage 2 outputs, once a
    checkpoint is available: confusion_matrix_generated.png, roc_curves.png,
    classification_metrics.csv)

Usage:
    python -m mentor_eval.classification_validation [--ckpt PATH] [--out-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wfdb
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, auc, f1_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_config, get_logger, set_seed
from mentor_eval.class_mapping import (
    MENTOR_CLASSES, MENTOR_TO_TRAINED_CLASS, load_ptbxl_database, filter_to_mentor_classes,
)
from mentor_eval.checkpoint_utils import load_checkpoint, generate_for_class

TRAIN_FOLDS = list(range(1, 9))
VAL_FOLDS = [9]
TEST_FOLDS = [10]


class _ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, pool=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool),
        )

    def forward(self, x):
        return self.net(x)


class MentorClassifier(nn.Module):
    """Same architecture as step05's Simple1DCNN, retrained for the 4
    mentor-review classes (different label scheme)."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            _ConvBlock1D(12, 32, 7, pool=4),
            _ConvBlock1D(32, 64, 5, pool=4),
            _ConvBlock1D(64, 128, 5, pool=2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        return self.head(self.encoder(x).squeeze(-1))


def _load_signals_for_fold(ptbxl_dir: Path, filtered_db, folds: list[int], log):
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
    log.info(f"Loaded {len(X)} readable records for folds {folds}")
    return np.array(X), np.array(y)


def train_classifier(X_train, y_train, X_val, y_val, n_classes: int, device: str, log, n_epochs: int = 30):
    model = MentorClassifier(n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    Xtr = torch.from_numpy(X_train.transpose(0, 2, 1)).float()
    ytr = torch.from_numpy(y_train).long()
    Xva = torch.from_numpy(X_val.transpose(0, 2, 1)).float()
    yva = torch.from_numpy(y_val).long()

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)

    best_val_f1, best_state = -1.0, None
    for epoch in range(1, n_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xva.to(device)).argmax(dim=1).cpu().numpy()
        val_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1, best_state = val_f1, {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            log.info(f"  classifier epoch {epoch}/{n_epochs}: val_macro_f1={val_f1:.4f}")

    model.load_state_dict(best_state)
    log.info(f"Best val macro F1: {best_val_f1:.4f}")
    return model


def evaluate_classifier(model, X, y, n_classes, device, class_names, out_path_cm: Path, title: str):
    Xt = torch.from_numpy(X.transpose(0, 2, 1)).float().to(device)
    model.eval()
    with torch.no_grad():
        logits = model(Xt)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    pred = probs.argmax(axis=1)

    acc = accuracy_score(y, pred)
    cm = confusion_matrix(y, pred, labels=list(range(n_classes)))

    y_bin = label_binarize(y, classes=list(range(n_classes)))
    roc_data = {}
    aucs = []
    for i, cname in enumerate(class_names):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        roc_data[cname] = (fpr.tolist(), tpr.tolist(), roc_auc)
        aucs.append(roc_auc)
    macro_auc = float(np.mean(aucs)) if aucs else None

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n_classes)); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(n_classes)); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    out_path_cm.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path_cm), dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "accuracy": float(acc),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "macro_auc": macro_auc,
        "per_class_auc": {k: v[2] for k, v in roc_data.items()},
        "confusion_matrix": cm.tolist(),
    }, roc_data


def plot_roc_curves(roc_data: dict, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for cname, (fpr, tpr, roc_auc) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{cname} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(ckpt_path: Path, out_dir: Path, cfg, seed: int, log) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ptbxl_dir = Path(cfg.paths.data.ptbxl)
    out_dir.mkdir(parents=True, exist_ok=True)

    db = load_ptbxl_database(ptbxl_dir)
    filtered = filter_to_mentor_classes(db)
    n_classes = len(MENTOR_CLASSES)

    # ── Stage 1: train + evaluate on REAL data (no checkpoint needed) ──────────
    log.info("Stage 1: training classifier on real PTB-XL data (4 mentor classes) …")
    X_train, y_train = _load_signals_for_fold(ptbxl_dir, filtered, TRAIN_FOLDS, log)
    X_val, y_val     = _load_signals_for_fold(ptbxl_dir, filtered, VAL_FOLDS, log)
    X_test, y_test   = _load_signals_for_fold(ptbxl_dir, filtered, TEST_FOLDS, log)

    if len(X_train) == 0:
        raise RuntimeError("No readable training records found — cannot train classifier.")

    model = train_classifier(X_train, y_train, X_val, y_val, n_classes, device, log)

    real_metrics, real_roc = evaluate_classifier(
        model, X_test, y_test, n_classes, device, MENTOR_CLASSES,
        out_dir / "confusion_matrix_real.png", "Classifier on REAL test data",
    )
    plot_roc_curves(real_roc, out_dir / "roc_curves_real.png", "ROC — classifier on real test data")
    with open(out_dir / "classifier_real_eval.json", "w") as f:
        json.dump(real_metrics, f, indent=2)
    log.info(f"Stage 1 done. Real-data test accuracy={real_metrics['accuracy']:.4f}, macro_f1={real_metrics['macro_f1']:.4f}")
    print(f"Stage 1 (real data) — accuracy={real_metrics['accuracy']:.4f}  macro_f1={real_metrics['macro_f1']:.4f}  macro_auc={real_metrics['macro_auc']}")

    # ── Stage 2: evaluate on GENERATED data (needs checkpoint) ─────────────────
    loaded = load_checkpoint(ckpt_path, cfg)
    if loaded is None:
        print(
            f"\n[PARTIAL] Stage 1 (real-data classifier) complete — see {out_dir}.\n"
            f"[BLOCKED] Stage 2 (evaluate on generated ECGs) needs {ckpt_path}, which doesn't\n"
            f"  exist on this machine. Re-run this script on the GPU server once trained.\n"
            f"  No generated-data metrics were fabricated.\n"
        )
        return

    stats_path = Path(cfg.paths.outputs.processed) / "preprocessing_stats.json"
    prep_stats = None
    if stats_path.exists():
        prep_stats = json.load(open(stats_path))

    log.info("Stage 2: generating samples and evaluating classifier …")
    gen_X, gen_y, skipped = [], [], []
    name_to_idx = {n: i for i, n in enumerate(MENTOR_CLASSES)}
    for cls in MENTOR_CLASSES:
        trained_cls = MENTOR_TO_TRAINED_CLASS.get(cls)
        if trained_cls is None:
            skipped.append(cls)
            continue
        samples, err = generate_for_class(loaded, trained_cls, n_samples=100, cfg=cfg, seed=seed, stats=prep_stats)
        if err:
            skipped.append(cls)
            continue
        gen_X.append(samples)
        gen_y.append(np.full(len(samples), name_to_idx[cls]))

    if skipped:
        log.warning(f"Classes excluded from generated-data evaluation (no model class to condition on): {skipped}")

    if not gen_X:
        print("[BLOCKED] No classes could be generated — stage 2 produced no metrics.")
        return

    gen_X = np.concatenate(gen_X, axis=0)
    gen_y = np.concatenate(gen_y, axis=0)

    gen_metrics, gen_roc = evaluate_classifier(
        model, gen_X, gen_y, n_classes, device, MENTOR_CLASSES,
        out_dir / "confusion_matrix_generated.png", "Classifier on GENERATED data",
    )
    plot_roc_curves(gen_roc, out_dir / "roc_curves_generated.png", "ROC — classifier on generated data")
    gen_metrics["excluded_classes"] = skipped
    with open(out_dir / "classifier_generated_eval.json", "w") as f:
        json.dump(gen_metrics, f, indent=2)
    log.info(f"Stage 2 done. Generated-data accuracy={gen_metrics['accuracy']:.4f}")
    print(f"Stage 2 (generated data) — accuracy={gen_metrics['accuracy']:.4f}  macro_f1={gen_metrics['macro_f1']:.4f}  excluded={skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Downstream classification validation, real vs generated.")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("classification_validation", cfg=cfg)
    set_seed(args.seed)

    ckpt_path = Path(args.ckpt) if args.ckpt else Path(cfg.paths.outputs.models) / "diffusion_best.pt"
    out_dir   = Path(args.out_dir) if args.out_dir else Path(cfg.paths.outputs.results).parent / "mentor_review" / "classification_validation"

    run(ckpt_path, out_dir, cfg, args.seed, log)
    print(f"✓ Outputs (whatever stage(s) completed) written to {out_dir}")


if __name__ == "__main__":
    main()
