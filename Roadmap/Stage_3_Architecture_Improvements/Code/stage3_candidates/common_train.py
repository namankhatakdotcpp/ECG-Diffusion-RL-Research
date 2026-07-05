"""
Stage 3 / Phase 1-2 -- shared training entry point for all 5 candidates.

Reuses step04_transformer_diffusion.py's data-loading helpers,
GaussianDiffusion, EMA, and _validation_plot UNMODIFIED (imported, not
copied) -- only the model-construction line and checkpoint output
paths differ from step04's own `train()`, so each candidate gets its
own model (via model_variants.build_variant_model) and its own
Results/<run_id>/ output directory instead of overwriting the real
outputs/models/diffusion_best.pt used by mentor_eval's baseline
protocol.

Per Stage3_Roadmap.md Sec. 4: this is Track B implementation +
local smoke-testing only. GPU training itself is gated on Decision
Gate A having resolved candidate priority (see Stage2_Decision_Report.md
addendum, 2026-07-05) -- calling train_variant() with n_epochs>0 is a
Phase 2 action, not performed by this pass.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from step04_transformer_diffusion import (  # noqa: E402
    GaussianDiffusion,
    EMA,
    ECGDataset,
    _load_class_labels,
    _make_weighted_sampler,
    _validation_plot,
    _resolve_device,
)

from model_variants import build_variant_model  # noqa: E402

RESULTS_ROOT = (
    REPO_ROOT / "Roadmap" / "Stage_3_Architecture_Improvements" / "Results"
)

KEEP_LAST_N_CHECKPOINTS = 2  # same retention policy as step04, per candidate run


def train_variant(cfg, log, variant: str, run_id: str, n_epochs_override: Optional[int] = None) -> float:
    """Trains one Stage 3 candidate. Identical procedure to step04's own
    `train()` (data loading, optimizer groups, CFG dropout, AMP,
    checkpoint retention) -- only the model class (via
    build_variant_model) and output directory (Results/<run_id>/,
    never outputs/models/) differ. Returns best val loss."""
    device = _resolve_device(cfg)
    log.info(f"[{run_id}] Device: {device} | variant={variant}")

    processed_dir = Path(cfg.paths.outputs.processed)
    run_dir       = RESULTS_ROOT / run_id
    models_dir    = run_dir / "checkpoints"
    results_dir   = run_dir / "plots"
    logs_dir      = run_dir / "logs"
    for d_path in (models_dir, results_dir, logs_dir):
        d_path.mkdir(parents=True, exist_ok=True)

    cls_names_path = processed_dir / "class_names.json"
    cls_map_path   = processed_dir / "class_mapping.json"
    if cls_names_path.exists() and cls_map_path.exists():
        with open(cls_names_path) as f:
            class_names = json.load(f)
        with open(cls_map_path) as f:
            class_mapping = json.load(f)
    else:
        class_names   = list(cfg.ptbxl.classes)
        class_mapping = {c: c for c in class_names}
    n_classes = len(class_names)

    assert int(cfg.ptbxl.n_classes) == n_classes, (
        f"config.yaml declares ptbxl.n_classes={int(cfg.ptbxl.n_classes)} but "
        f"the class list actually in use has {n_classes} classes: {class_names}."
    )

    for p in (
        processed_dir / "X_train.npy", processed_dir / "X_val.npy",
        processed_dir / "record_ids_train.npy", processed_dir / "record_ids_val.npy",
    ):
        if not p.exists():
            log.error(f"Missing: {p}. Run step02_preprocessing.py first.")
            raise FileNotFoundError(p)

    X_train       = np.load(str(processed_dir / "X_train.npy"))
    X_val         = np.load(str(processed_dir / "X_val.npy"))
    rec_ids_train = np.load(str(processed_dir / "record_ids_train.npy"))
    rec_ids_val   = np.load(str(processed_dir / "record_ids_val.npy"))

    db_path = Path(cfg.paths.data.ptbxl) / "ptbxl_database.csv"
    if not db_path.exists():
        log.error(f"ptbxl_database.csv not found at {db_path}. Run step01 first.")
        raise FileNotFoundError(db_path)
    ptbxl_db = pd.read_csv(str(db_path), index_col="ecg_id")

    vi_train, train_labels = _load_class_labels(rec_ids_train, ptbxl_db, class_mapping, class_names, log)
    vi_val,   val_labels   = _load_class_labels(rec_ids_val,   ptbxl_db, class_mapping, class_names, log)
    X_train, X_val = X_train[vi_train], X_val[vi_val]
    log.info(f"[{run_id}] Train distribution: "
             f"{dict(Counter(class_names[i] for i in train_labels.tolist()))}")

    d = cfg.diffusion
    train_ds = ECGDataset(X_train, train_labels)
    val_ds   = ECGDataset(X_val,   val_labels)
    sampler  = _make_weighted_sampler(train_labels)
    train_loader = DataLoader(
        train_ds, batch_size=int(d.batch_size), sampler=sampler,
        num_workers=0, pin_memory=(device == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(d.batch_size), shuffle=False,
        num_workers=0, pin_memory=(device == "cuda"),
    )

    model = build_variant_model(cfg, n_classes=n_classes, variant=variant).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"[{run_id}] Model parameters: {n_params / 1e6:.2f}M")

    diffusion = GaussianDiffusion(T=int(d.T), beta_schedule=str(d.beta_schedule), device=device)
    ema       = EMA(model, decay=float(d.ema_decay))

    p_uncond = float(getattr(d, "p_uncond", 0.10))

    _lr = float(d.lr)
    _wd = float(d.weight_decay)
    # class_emb.weight: excluded per Item 2's finding that decaying a
    # conditioning-carrying parameter suppresses its signal. gamma1/gamma2
    # (LayerScale, late_gain, residual_scaling) and boost (hybrid's extra
    # late-block gain, model_variants.py:128) are the same class of
    # learnable per-branch gain and get the same exclusion -- decay was
    # otherwise pulling every one of them toward 0 every step with no
    # protection, unlike class_emb.weight.
    _nodecay_names = lambda n: n == "class_emb.weight" or "gamma" in n or "boost" in n
    _decay_params   = [p for n, p in model.named_parameters() if not _nodecay_names(n)]
    _nodecay_params = [p for n, p in model.named_parameters() if _nodecay_names(n)]

    # Tripwire: the name-based exclusion above only knows about gain
    # parameters that exist today (gamma1/gamma2/boost). A future variant
    # (e.g. S3-006) could introduce a differently-named learnable gain
    # scalar that silently reintroduces the same unprotected-decay bug.
    # Flag any small 1D parameter (gain-shaped) that isn't already
    # excluded and isn't a norm/bias param (which legitimately decay
    # under this codebase's convention) -- refuse to start training
    # rather than train silently under an unreviewed assumption.
    _uncaught_gain_like = [
        n for n, p in model.named_parameters()
        if p.ndim <= 1 and p.numel() <= 512
        and not _nodecay_names(n)
        and "norm" not in n and "bias" not in n
    ]
    if _uncaught_gain_like:
        raise RuntimeError(
            f"[{run_id}] found {len(_uncaught_gain_like)} small 1D parameter(s) "
            f"not covered by the gamma/boost/class_emb.weight no-decay exclusion "
            f"and not a norm/bias param: {_uncaught_gain_like}. Verify whether "
            f"these are gain-like parameters that need decay excluded before "
            f"training (see 2026-07-05 weight-decay finding)."
        )
    optimiser = torch.optim.AdamW(
        [
            {"params": _decay_params,   "weight_decay": _wd},
            {"params": _nodecay_params, "weight_decay": 0.0},
        ],
        lr=_lr,
    )
    n_epochs   = int(n_epochs_override if n_epochs_override is not None else d.n_epochs)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(n_epochs, 1))
    use_amp    = (device == "cuda")
    scaler     = torch.cuda.amp.GradScaler(enabled=use_amp)

    log_path = logs_dir / "training_log.csv"
    log_fh   = open(log_path, "w", newline="")
    writer   = csv.writer(log_fh)
    writer.writerow(["epoch", "step", "train_loss", "val_loss", "lr"])
    log_fh.flush()

    best_val_loss = float("inf")
    global_step   = 0
    save_every    = int(d.save_every)
    saved_periodic_ckpts: list[Path] = []

    log.info(f"[{run_id}] Training: {n_epochs} epochs x {len(train_loader)} steps")

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_x, batch_cls in train_loader:
            batch_x, batch_cls = batch_x.to(device), batch_cls.to(device)
            B = batch_x.shape[0]

            if p_uncond > 0.0:
                null_mask = torch.bernoulli(torch.full((B,), p_uncond, device=device)).bool()
                batch_cls = batch_cls.clone()
                batch_cls[null_mask] = model.null_class_index

            t_diff = torch.randint(0, int(d.T), (B,), device=device)
            noise  = torch.randn_like(batch_x)
            x_t, _ = diffusion.q_sample(batch_x, t_diff, noise)

            with torch.cuda.amp.autocast(enabled=use_amp):
                eps_pred = model(x_t, t_diff, batch_cls)
                loss     = F.mse_loss(eps_pred, noise)

            optimiser.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(d.grad_clip))
            scaler.step(optimiser)
            scaler.update()
            ema.update(model)

            epoch_loss  += loss.item()
            global_step += 1

        scheduler.step()
        avg_train = epoch_loss / len(train_loader)

        if epoch % save_every == 0:
            model.eval()
            val_total, val_steps = 0.0, 0
            with ema.ema_scope(model), torch.no_grad():
                for batch_x, batch_cls in val_loader:
                    batch_x, batch_cls = batch_x.to(device), batch_cls.to(device)
                    B = batch_x.shape[0]
                    t_diff = torch.randint(0, int(d.T), (B,), device=device)
                    noise  = torch.randn_like(batch_x)
                    x_t, _ = diffusion.q_sample(batch_x, t_diff, noise)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        eps_pred = model(x_t, t_diff, batch_cls)
                        v_loss   = F.mse_loss(eps_pred, noise)
                    val_total += v_loss.item()
                    val_steps += 1
                    if val_steps >= 100:
                        break
            val_loss = val_total / max(val_steps, 1)

            nn_mse = _validation_plot(
                model, diffusion, ema, X_val, val_labels,
                class_names, cfg, epoch, results_dir, device,
            )
            log.info(f"[{run_id}] Epoch {epoch:04d}/{n_epochs} | train={avg_train:.5f} | "
                     f"val={val_loss:.5f} | nn_mse={nn_mse:.4f} | elapsed={time.time()-t0:.1f}s")

            ckpt = {
                "epoch": epoch, "model": model.state_dict(), "ema_shadow": ema.shadow,
                "optimiser": optimiser.state_dict(), "val_loss": val_loss,
                "class_names": class_names, "n_classes": n_classes, "variant": variant,
            }
            ckpt_path = models_dir / f"{run_id}_ep{epoch:04d}.pt"
            torch.save(ckpt, str(ckpt_path))
            saved_periodic_ckpts.append(ckpt_path)
            while len(saved_periodic_ckpts) > KEEP_LAST_N_CHECKPOINTS:
                stale = saved_periodic_ckpts.pop(0)
                stale.unlink(missing_ok=True)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(ckpt, str(models_dir / f"{run_id}_best.pt"))

            writer.writerow([epoch, global_step, f"{avg_train:.6f}", f"{val_loss:.6f}",
                              f"{scheduler.get_last_lr()[0]:.2e}"])
            log_fh.flush()

    log_fh.close()
    return best_val_loss
