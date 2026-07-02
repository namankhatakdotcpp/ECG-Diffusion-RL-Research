"""
Stage 1 / Experiment 2 — Dataset Scaling.

Progressively increase the amount of (already-downloaded) PTB-XL training
data used to train the diffusion model, and measure whether class
conditioning improves with data alone. No new dataset is downloaded — every
subset here is a stratified sample of the existing outputs/processed/*.npy
arrays.

For each target size in DATASET_SIZES, this script:
  1. Builds a stratified subset of X_train (same class proportions as the
     full set, fixed seed for reproducibility).
  2. Trains a fresh ECGTransformerDiffusion model on that subset only
     (same architecture/hyperparameters as step04_transformer_diffusion.py
     — nothing about the model changes, only the amount of training data).
  3. Times training and records peak GPU memory.
  4. Generates samples per class and evaluates them with a SINGLE
     MentorClassifier trained once on the FULL real dataset (kept fixed
     across all six runs, so the classifier itself is not a confound —
     only the diffusion model's training-data size varies).
  5. Records accuracy, macro F1, and collapse fraction (the fraction of all
     generated samples predicted as the single most common class — this is
     what "conditioning collapse" looks like quantitatively).
  6. (Experiment 2.5) Also records a TRAINING CURVE for each dataset size —
     loss, sensitivity_metric, collapse_frac, and macro_f1 every
     --curve-every epochs (default 25, matching step04's save_every) —
     instead of only the final numbers. This distinguishes "this dataset
     size never conditions well" from "this dataset size would condition
     well with more epochs" the same way Experiment 1.5 does for the
     single baseline run.

Run on the GPU server:
    python Roadmap/Stage_1_Diagnosis/Code/Experiment_2_Dataset_Scaling/run_dataset_scaling.py \\
        [--epochs 200] [--sizes 380,1000,2500,5000,10000,full] [--curve-every 25]

Writes:
    Roadmap/Stage_1_Diagnosis/Outputs/Experiment_2_Dataset_Scaling/
        checkpoints/size_{N}/diffusion_best.pt
        dataset_scaling_metrics.csv
        dataset_scaling_metrics.json
        training_curves_size_{N}.csv          (Experiment 2.5)
    Roadmap/Stage_1_Diagnosis/Figures/Experiment_2_Dataset_Scaling/
        accuracy_vs_size.png, macro_f1_vs_size.png, collapse_vs_size.png,
        train_time_vs_size.png
        training_curves_all_sizes.png          (Experiment 2.5)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Code/ for common_probes

from utils import load_config, get_logger, set_seed
from step04_transformer_diffusion import (
    ECGTransformerDiffusion, GaussianDiffusion, EMA,
    ECGDataset, _make_weighted_sampler, _load_class_labels,
    _resolve_device, generate_ecg,
)
from mentor_eval.classification_validation import (
    MentorClassifier, train_classifier, evaluate_classifier,
    _load_signals_for_fold, TRAIN_FOLDS, VAL_FOLDS, TEST_FOLDS,
)
from mentor_eval.class_mapping import (
    MENTOR_CLASSES, MENTOR_TO_TRAINED_CLASS, load_ptbxl_database, filter_to_mentor_classes,
)
from common_probes import sensitivity_metric, collapse_and_macro_f1

sys.path.insert(0, str(REPO_ROOT / "Roadmap" / "_infra"))
from experiment_logger import ExperimentLogger

OUT_DIR = REPO_ROOT / "Roadmap" / "Stage_1_Diagnosis" / "Outputs" / "Experiment_2_Dataset_Scaling"
FIG_DIR = REPO_ROOT / "Roadmap" / "Stage_1_Diagnosis" / "Figures" / "Experiment_2_Dataset_Scaling"
DATASET_SIZES_DEFAULT = [380, 1000, 2500, 5000, 10000, "full"]
SEED = 42


def stratified_subset(labels: np.ndarray, target_size: int, rng: np.random.Generator) -> np.ndarray:
    """Indices for a class-proportional subset of `labels` totalling
    (approximately) target_size, at least 1 sample per class present."""
    n_total = len(labels)
    if target_size >= n_total:
        return np.arange(n_total)

    idx_by_class = {c: np.where(labels == c)[0] for c in np.unique(labels)}
    frac = target_size / n_total
    chosen = []
    for c, idxs in idx_by_class.items():
        n_c = max(1, round(len(idxs) * frac))
        n_c = min(n_c, len(idxs))
        chosen.append(rng.choice(idxs, size=n_c, replace=False))
    out = np.concatenate(chosen)
    rng.shuffle(out)
    return out


def train_diffusion_on_subset(
    X_sub: np.ndarray, y_sub: np.ndarray, n_classes: int, cfg, n_epochs: int,
    device: str, log,
    curve_every: int = 0, curve_clf=None, curve_class_names=None, curve_prep_stats=None,
    curve_n_gen: int = 20, curve_seed: int = 42,
) -> tuple[ECGTransformerDiffusion, GaussianDiffusion, EMA, dict, list[dict]]:
    """Minimal training loop mirroring step04's train(), parameterized by an
    arbitrary training subset. Reuses the same model/diffusion/EMA classes —
    does not redefine architecture or CFG logic.

    Experiment 2.5: if curve_every > 0 and curve_clf is provided, records a
    training-curve row (train_loss, sensitivity_metric, collapse_frac,
    macro_f1) every `curve_every` epochs, instead of only the final numbers
    Experiment 2 originally reported. Uses a smaller generation count
    (curve_n_gen, default 20/class) than the end-of-run evaluation, since
    this runs many times per dataset size rather than once.
    """
    d = cfg.diffusion
    train_ds = ECGDataset(X_sub, y_sub)
    sampler = _make_weighted_sampler(y_sub)
    loader = DataLoader(
        train_ds, batch_size=int(d.batch_size), sampler=sampler,
        num_workers=0, pin_memory=(device == "cuda"), drop_last=True,
    )

    model = ECGTransformerDiffusion(cfg, n_classes=n_classes).to(device)
    diffusion = GaussianDiffusion(T=int(d.T), beta_schedule=str(d.beta_schedule), device=device)
    ema = EMA(model, decay=float(d.ema_decay))
    p_uncond = float(getattr(d, "p_uncond", 0.10))

    decay_params = [p for n, p in model.named_parameters() if n != "class_emb.weight"]
    nodecay_params = [p for n, p in model.named_parameters() if n == "class_emb.weight"]
    optimiser = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": float(d.weight_decay)},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=float(d.lr),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=n_epochs)
    use_amp = (device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    curve_rows: list[dict] = []
    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_cls in loader:
            batch_x, batch_cls = batch_x.to(device), batch_cls.to(device)
            B = batch_x.shape[0]
            if p_uncond > 0.0:
                null_mask = torch.bernoulli(torch.full((B,), p_uncond, device=device)).bool()
                batch_cls = batch_cls.clone()
                batch_cls[null_mask] = model.null_class_index

            t_diff = torch.randint(0, int(d.T), (B,), device=device)
            noise = torch.randn_like(batch_x)
            x_t, _ = diffusion.q_sample(batch_x, t_diff, noise)

            with torch.cuda.amp.autocast(enabled=use_amp):
                eps_pred = model(x_t, t_diff, batch_cls)
                loss = F.mse_loss(eps_pred, noise)

            optimiser.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(d.grad_clip))
            scaler.step(optimiser)
            scaler.update()
            ema.update(model)
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        if epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs:
            log.info(f"  epoch {epoch}/{n_epochs}: train_loss={avg_loss:.5f}")

        # Experiment 2.5: periodic training-curve snapshot (loss/sensitivity/collapse/F1),
        # not just the final numbers.
        if curve_every > 0 and curve_clf is not None and (epoch % curve_every == 0 or epoch == n_epochs):
            model.eval()
            sens = sensitivity_metric(model, device, n_classes, cfg)
            collapse_frac, macro_f1 = collapse_and_macro_f1(
                model, diffusion, curve_class_names, curve_clf, device, cfg, curve_prep_stats,
                MENTOR_CLASSES, MENTOR_TO_TRAINED_CLASS, curve_n_gen, curve_seed,
            )
            curve_rows.append({
                "epoch": epoch, "train_loss": avg_loss,
                "sensitivity_metric": sens, "collapse_frac": collapse_frac, "macro_f1": macro_f1,
            })
            log.info(f"  [curve] epoch {epoch}: sensitivity={sens:.4f} collapse={collapse_frac:.4f} macro_f1={macro_f1:.4f}")
            model.train()

    train_time_sec = time.time() - t0
    peak_gpu_gb = (
        torch.cuda.max_memory_allocated(device) / 1e9 if device == "cuda" else float("nan")
    )
    stats = {"train_time_sec": train_time_sec, "peak_gpu_mem_gb": peak_gpu_gb,
              "final_train_loss": avg_loss}
    return model, diffusion, ema, stats, curve_rows


def evaluate_with_fixed_classifier(
    model, diffusion, class_names: list[str], clf: MentorClassifier, device: str,
    cfg, prep_stats, n_gen_per_class: int, seed: int,
) -> dict:
    """Generate n_gen_per_class samples for every trained diffusion class
    (except OTHER — not a clinically meaningful target), classify them with
    the fixed MentorClassifier, and compute accuracy/macro-F1/collapse
    against the diffusion class as ground truth (mapped through
    MENTOR_TO_TRAINED_CLASS's inverse — see note below).
    """
    trained_to_mentor = {v: k for k, v in MENTOR_TO_TRAINED_CLASS.items() if v is not None}
    gen_X, gen_y_mentor = [], []
    mentor_name_to_idx = {n: i for i, n in enumerate(MENTOR_CLASSES)}

    for cls_name in class_names:
        mentor_name = trained_to_mentor.get(cls_name)
        if mentor_name is None:
            continue  # OTHER (and anything else with no mentor-class counterpart)
        samples = generate_ecg(
            model, diffusion, class_label=class_names.index(cls_name),
            n_samples=n_gen_per_class, device=device, cfg=cfg, seed=seed,
            stats=prep_stats,
        )
        gen_X.append(samples)
        gen_y_mentor.append(np.full(len(samples), mentor_name_to_idx[mentor_name]))

    gen_X = np.concatenate(gen_X, axis=0)
    gen_y_mentor = np.concatenate(gen_y_mentor, axis=0)

    Xt = torch.from_numpy(gen_X.transpose(0, 2, 1)).float().to(device)
    clf.eval()
    with torch.no_grad():
        pred = clf(Xt).argmax(dim=1).cpu().numpy()

    acc = float((pred == gen_y_mentor).mean())
    from sklearn.metrics import f1_score
    macro_f1 = float(f1_score(gen_y_mentor, pred, average="macro", zero_division=0))
    counts = Counter(pred.tolist())
    collapse_frac = float(max(counts.values())) / len(pred)

    return {"accuracy": acc, "macro_f1": macro_f1, "collapse_frac": collapse_frac,
             "n_generated": len(pred)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None,
                        help="Epochs per dataset size (default: config.yaml diffusion.n_epochs)")
    parser.add_argument("--sizes", type=str, default=None,
                        help="Comma-separated sizes, e.g. 380,1000,2500,5000,10000,full")
    parser.add_argument("--n-gen-per-class", type=int, default=100)
    parser.add_argument("--curve-every", type=int, default=25,
                        help="Record a training-curve snapshot every N epochs (Experiment 2.5). 0 disables.")
    parser.add_argument("--curve-n-gen", type=int, default=20,
                        help="Samples/class generated for each training-curve snapshot (kept small — runs many times per size).")
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("dataset_scaling", cfg=cfg)
    set_seed(SEED)
    device = _resolve_device(cfg)
    log.info(f"Device: {device}")

    n_epochs = args.epochs or int(cfg.diffusion.n_epochs)
    sizes = (
        [s if s == "full" else int(s) for s in args.sizes.split(",")]
        if args.sizes else DATASET_SIZES_DEFAULT
    )

    processed_dir = Path(cfg.paths.outputs.processed)
    class_names = json.load(open(processed_dir / "class_names.json"))
    class_mapping = json.load(open(processed_dir / "class_mapping.json"))
    n_classes = len(class_names)

    X_train = np.load(processed_dir / "X_train.npy")
    rec_ids_train = np.load(processed_dir / "record_ids_train.npy")
    db_path = Path(cfg.paths.data.ptbxl) / "ptbxl_database.csv"
    ptbxl_db = pd.read_csv(str(db_path), index_col="ecg_id")
    vi_train, y_train_full = _load_class_labels(rec_ids_train, ptbxl_db, class_mapping, class_names, log)
    X_train_full = X_train[vi_train]
    log.info(f"Full mapped training pool: {len(X_train_full)} records, "
             f"distribution={dict(Counter(class_names[i] for i in y_train_full.tolist()))}")

    stats_path = processed_dir / "preprocessing_stats.json"
    prep_stats = json.load(open(stats_path)) if stats_path.exists() else None

    # ── Fixed MentorClassifier, trained once on FULL real data ────────────────
    log.info("Training fixed MentorClassifier on full real PTB-XL (used for all dataset sizes) …")
    ptbxl_dir = Path(cfg.paths.data.ptbxl)
    mentor_db = filter_to_mentor_classes(load_ptbxl_database(ptbxl_dir))
    Xc_train, yc_train = _load_signals_for_fold(ptbxl_dir, mentor_db, TRAIN_FOLDS, log)
    Xc_val, yc_val = _load_signals_for_fold(ptbxl_dir, mentor_db, VAL_FOLDS, log)
    clf = train_classifier(Xc_train, yc_train, Xc_val, yc_val, len(MENTOR_CLASSES), device, log)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    rows = []
    for size in sizes:
        with ExperimentLogger(
            experiment_id=f"exp2_dataset_scaling_{size}",
            stage="Stage_1_Diagnosis",
            root_dir=REPO_ROOT / "Roadmap" / "Stage_1_Diagnosis",
            params={"dataset_size_requested": size, "n_epochs": n_epochs,
                    "curve_every": args.curve_every, "n_gen_per_class": args.n_gen_per_class},
            seed=SEED,
            repo_dir=REPO_ROOT,
        ) as exp:
            target = len(X_train_full) if size == "full" else size
            log.info(f"=== Dataset size: {size} (target={target}) ===")
            sub_idx = stratified_subset(y_train_full, target, rng)
            X_sub, y_sub = X_train_full[sub_idx], y_train_full[sub_idx]
            log.info(f"  Subset: {len(X_sub)} records, "
                     f"distribution={dict(Counter(class_names[i] for i in y_sub.tolist()))}")
            exp.log_metric("n_train_records_actual", len(X_sub))

            model, diffusion, ema, train_stats, curve_rows = train_diffusion_on_subset(
                X_sub, y_sub, n_classes, cfg, n_epochs, device, log,
                curve_every=args.curve_every, curve_clf=clf, curve_class_names=class_names,
                curve_prep_stats=prep_stats, curve_n_gen=args.curve_n_gen, curve_seed=SEED,
            )
            exp.log_metrics(train_stats)

            if curve_rows:
                curve_csv = OUT_DIR / f"training_curves_size_{size}.csv"
                pd.DataFrame(curve_rows).to_csv(curve_csv, index=False)
                exp.log_artifact(curve_csv, "per-epoch training curve for this dataset size")

            ckpt_dir = OUT_DIR / "checkpoints" / f"size_{size}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / "diffusion_best.pt"
            torch.save(
                {"model": model.state_dict(), "ema_shadow": ema.shadow,
                 "class_names": class_names, "n_classes": n_classes},
                ckpt_path,
            )
            exp.log_artifact(ckpt_path, f"diffusion checkpoint trained on {len(X_sub)} records")

            eval_stats = evaluate_with_fixed_classifier(
                model, diffusion, class_names, clf, device, cfg, prep_stats,
                args.n_gen_per_class, SEED,
            )
            exp.log_metrics(eval_stats)
            log.info(f"  {train_stats} | {eval_stats}")

            rows.append({"dataset_size": size, "n_train_records": len(X_sub),
                         **train_stats, **eval_stats})
            pd.DataFrame(rows).to_csv(OUT_DIR / "dataset_scaling_metrics.csv", index=False)
            del model, diffusion, ema
            if device == "cuda":
                torch.cuda.empty_cache()

    with open(OUT_DIR / "dataset_scaling_metrics.json", "w") as f:
        json.dump(rows, f, indent=2)

    # ── Plots ──────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(rows)
    x_labels = [str(s) for s in df["dataset_size"]]
    for metric, fname, ylabel in [
        ("accuracy", "accuracy_vs_size.png", "Accuracy (generated, mentor classes)"),
        ("macro_f1", "macro_f1_vs_size.png", "Macro F1 (generated, mentor classes)"),
        ("collapse_frac", "collapse_vs_size.png", "Collapse fraction"),
        ("train_time_sec", "train_time_vs_size.png", "Training time (s)"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x_labels, df[metric], marker="o")
        ax.set_xlabel("Training set size")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs. PTB-XL training set size")
        fig.tight_layout()
        fig.savefig(FIG_DIR / fname, dpi=200)
        plt.close(fig)

    # ── Experiment 2.5: combined training-curve plot across all sizes ──────────
    curve_files = sorted(OUT_DIR.glob("training_curves_size_*.csv"))
    if curve_files:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for cf in curve_files:
            size_label = cf.stem.replace("training_curves_size_", "")
            cdf = pd.read_csv(cf)
            axes[0].plot(cdf["epoch"], cdf["sensitivity_metric"], marker="o", label=size_label)
            axes[1].plot(cdf["epoch"], cdf["collapse_frac"], marker="o", label=size_label)
            axes[2].plot(cdf["epoch"], cdf["macro_f1"], marker="o", label=size_label)
        for ax, title in zip(axes, ["Sensitivity metric vs epoch", "Collapse fraction vs epoch", "Macro F1 vs epoch"]):
            ax.set_xlabel("Epoch")
            ax.set_title(title)
            ax.legend(fontsize=7, title="Dataset size")
        fig.suptitle("Experiment 2.5 — Training curves across dataset sizes")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "training_curves_all_sizes.png", dpi=200)
        plt.close(fig)

    log.info(f"Done. Metrics: {OUT_DIR / 'dataset_scaling_metrics.csv'}")
    print(f"✓ Dataset scaling experiment complete. See {OUT_DIR} and {FIG_DIR}")


if __name__ == "__main__":
    main()
