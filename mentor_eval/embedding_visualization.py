"""
mentor_eval/embedding_visualization.py — UMAP and t-SNE of MentorClassifier
penultimate-layer embeddings for real and generated ECGs.

Registers a forward hook on MentorClassifier.encoder to extract 128-dim
features before the final linear head. Reduces to 2D using both UMAP and
t-SNE. Plots colored by class, real=circles, generated=triangles.

Requires:
  - PTB-XL dataset (for 100 real samples per mentor class)
  - outputs/models/diffusion_best.pt  (GPU server — generated side blocked without it)
  - outputs/conditioning_analysis/mentor_classifier.pt  (cached by
    conditioning_diagnostic.py; re-trained fresh if not found)

Writes to: outputs/conditioning_analysis/
  embedding_umap.png
  embedding_tsne.png
  embedding_features.csv   (2D coords, class, source=real/generated)
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
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_config, get_logger, set_seed
from mentor_eval.class_mapping import (
    MENTOR_CLASSES, MENTOR_TO_TRAINED_CLASS,
    load_ptbxl_database, filter_to_mentor_classes,
)
from mentor_eval.checkpoint_utils import load_checkpoint, generate_for_class
from mentor_eval.classification_validation import (
    MentorClassifier, TRAIN_FOLDS, VAL_FOLDS, TEST_FOLDS,
)

OUT_DIR = Path("outputs/conditioning_analysis")
N_REAL = 100
N_GEN  = 100


def _load_n_per_class(ptbxl_dir: Path, filtered_db, folds: list[int], n_per_class: int, log):
    """Load up to n_per_class real samples per MENTOR_CLASS from the given folds."""
    subset = filtered_db[filtered_db["strat_fold"].isin(folds)]
    name_to_idx = {n: i for i, n in enumerate(MENTOR_CLASSES)}
    per_class: dict[str, list] = {n: [] for n in MENTOR_CLASSES}

    for _, rec in subset.iterrows():
        cls = rec["mentor_class"]
        if len(per_class[cls]) >= n_per_class:
            continue
        try:
            sig = wfdb.rdrecord(str(ptbxl_dir / str(rec["filename_lr"]))).p_signal
        except Exception:
            continue
        if sig.shape != (1000, 12) or not np.isfinite(sig).all():
            continue
        per_class[cls].append(sig)

    X, y = [], []
    for cls, sigs in per_class.items():
        for sig in sigs:
            X.append(sig)
            y.append(name_to_idx[cls])
        log.info(f"  real {cls}: {len(sigs)} samples")
    return np.array(X), np.array(y)


def _train_fresh_classifier(ptbxl_dir, filtered_db, device, log, n_classes=4, n_epochs=30):
    from mentor_eval.classification_validation import _load_signals_for_fold
    X_train, y_train = _load_signals_for_fold(ptbxl_dir, filtered_db, TRAIN_FOLDS, log)
    X_val,   y_val   = _load_signals_for_fold(ptbxl_dir, filtered_db, VAL_FOLDS,   log)

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
    return model


def _extract_features(clf: MentorClassifier, X: np.ndarray, device: str) -> np.ndarray:
    """Extract 128-dim penultimate features by hooking clf.encoder."""
    captured = {}

    def _hook(module, inp, out):
        # out: (B, 128, 1) from AdaptiveAvgPool1d
        captured["feat"] = out.squeeze(-1).cpu()

    handle = clf.encoder.register_forward_hook(_hook)
    clf.eval()
    all_feats = []
    Xt = torch.from_numpy(X.transpose(0, 2, 1)).float()
    loader = DataLoader(TensorDataset(Xt), batch_size=64, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            clf(xb.to(device))
            all_feats.append(captured["feat"].numpy())
    handle.remove()
    return np.concatenate(all_feats, axis=0)  # (N, 128)


def _plot_embedding(coords_2d, labels, sources, class_names, method_name, out_path):
    """Plot 2D embedding: color=class, marker=source (circle=real, triangle=gen)."""
    palette = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(8, 7))

    for ci, cls in enumerate(class_names):
        for src, marker, label_suffix, alpha in [
            ("real",      "o", " (real)",      0.6),
            ("generated", "^", " (generated)", 0.85),
        ]:
            mask = (labels == ci) & (sources == src)
            if mask.sum() == 0:
                continue
            ax.scatter(
                coords_2d[mask, 0], coords_2d[mask, 1],
                c=[palette[ci % len(palette)]],
                marker=marker, s=30, alpha=alpha,
                label=f"{cls}{label_suffix}",
                edgecolors="none" if src == "real" else "k",
                linewidths=0.4,
            )

    ax.set_title(f"{method_name} — MentorClassifier penultimate features\n"
                 "Circles = real, Triangles = generated", fontsize=11)
    ax.set_xlabel(f"{method_name}-1"); ax.set_ylabel(f"{method_name}-2")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(cfg, log) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)

    ptbxl_dir = Path(cfg.paths.data.ptbxl)
    db = load_ptbxl_database(ptbxl_dir)
    filtered = filter_to_mentor_classes(db)

    # ── Load or train MentorClassifier ─────────────────────────────────────────
    cached_clf = OUT_DIR / "mentor_classifier.pt"
    clf = MentorClassifier(n_classes=len(MENTOR_CLASSES)).to(device)
    if cached_clf.exists():
        log.info(f"Loading cached classifier from {cached_clf}")
        clf.load_state_dict(torch.load(str(cached_clf), map_location=device))
    else:
        log.info("No cached classifier found — training fresh …")
        clf = _train_fresh_classifier(ptbxl_dir, filtered, device, log)
        torch.save(clf.state_dict(), str(cached_clf))
        log.info(f"Classifier cached → {cached_clf}")
    clf.eval()

    # ── Load 100 real samples per mentor class ─────────────────────────────────
    log.info("Loading real samples (up to 100 per class) from test fold …")
    all_folds = TEST_FOLDS + VAL_FOLDS + TRAIN_FOLDS  # prefer test, fall back
    X_real, y_real = _load_n_per_class(ptbxl_dir, filtered, all_folds, N_REAL, log)
    if len(X_real) == 0:
        print("[BLOCKED] No PTB-XL records readable — cannot extract real features.")
        return

    # ── Extract real features ──────────────────────────────────────────────────
    log.info("Extracting penultimate features from real samples …")
    feat_real = _extract_features(clf, X_real, device)

    # ── Load diffusion checkpoint and generate ─────────────────────────────────
    ckpt_path = Path(cfg.paths.outputs.models) / "diffusion_best.pt"
    loaded = load_checkpoint(ckpt_path, cfg)

    feat_gen, y_gen = [], []
    stats_path = Path(cfg.paths.outputs.processed) / "preprocessing_stats.json"
    prep_stats = json.load(open(stats_path)) if stats_path.exists() else None

    if loaded is None:
        print(
            f"\n[PARTIAL] Diffusion checkpoint not found at {ckpt_path}.\n"
            f"  Real-data embeddings are available ({len(X_real)} samples).\n"
            f"  Generated-data embeddings are BLOCKED — re-run on GPU server.\n"
        )
    else:
        name_to_idx = {n: i for i, n in enumerate(MENTOR_CLASSES)}
        for ci, cls in enumerate(MENTOR_CLASSES):
            trained_cls = MENTOR_TO_TRAINED_CLASS.get(cls)
            if trained_cls is None:
                log.info(f"  Skipping generated {cls}: no diffusion class mapping (expected for AFIB).")
                continue
            log.info(f"  Generating {N_GEN} samples for {cls} (diffusion class '{trained_cls}') …")
            samples, err = generate_for_class(
                loaded, trained_cls, n_samples=N_GEN, cfg=cfg, seed=200 + ci, stats=prep_stats,
            )
            if err:
                log.warning(f"  Skipped {cls}: {err}")
                continue
            feats = _extract_features(clf, samples, device)
            feat_gen.append(feats)
            y_gen.extend([ci] * len(feats))

    # ── Assemble combined feature matrix ───────────────────────────────────────
    all_feats   = [feat_real]
    all_labels  = list(y_real)
    all_sources = ["real"] * len(y_real)

    if feat_gen:
        all_feats.append(np.concatenate(feat_gen, axis=0))
        all_labels.extend(y_gen)
        all_sources.extend(["generated"] * len(y_gen))

    all_feats   = np.concatenate(all_feats, axis=0)
    all_labels  = np.array(all_labels)
    all_sources = np.array(all_sources)
    log.info(f"Total feature matrix: {all_feats.shape} (real={np.sum(all_sources=='real')}, gen={np.sum(all_sources=='generated')})")

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    log.info("Running t-SNE …")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_feats) - 1))
    coords_tsne = tsne.fit_transform(all_feats)
    _plot_embedding(coords_tsne, all_labels, all_sources, MENTOR_CLASSES,
                    "t-SNE", OUT_DIR / "embedding_tsne.png")
    log.info(f"Saved → {OUT_DIR}/embedding_tsne.png")

    # ── UMAP ──────────────────────────────────────────────────────────────────
    try:
        import umap
        log.info("Running UMAP …")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        coords_umap = reducer.fit_transform(all_feats)
        _plot_embedding(coords_umap, all_labels, all_sources, MENTOR_CLASSES,
                        "UMAP", OUT_DIR / "embedding_umap.png")
        log.info(f"Saved → {OUT_DIR}/embedding_umap.png")
    except ImportError:
        log.warning("umap-learn not installed — skipping UMAP. Run: pip install umap-learn")
        print("[BLOCKED] UMAP skipped: pip install umap-learn, then re-run.")
        coords_umap = None

    # ── Save CSV ───────────────────────────────────────────────────────────────
    import csv
    csv_path = OUT_DIR / "embedding_features.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["source", "class"]
        if coords_umap is not None:
            header += ["umap_x", "umap_y"]
        header += ["tsne_x", "tsne_y"]
        w.writerow(header)
        for i in range(len(all_labels)):
            row = [all_sources[i], MENTOR_CLASSES[all_labels[i]]]
            if coords_umap is not None:
                row += [f"{coords_umap[i,0]:.6f}", f"{coords_umap[i,1]:.6f}"]
            row += [f"{coords_tsne[i,0]:.6f}", f"{coords_tsne[i,1]:.6f}"]
            w.writerow(row)
    log.info(f"Saved raw coords → {csv_path}")
    print(f"\nEmbedding visualization complete. Outputs in {OUT_DIR}/")


def main() -> None:
    cfg = load_config()
    log = get_logger("embedding_visualization", cfg=cfg)
    run(cfg, log)


if __name__ == "__main__":
    main()
