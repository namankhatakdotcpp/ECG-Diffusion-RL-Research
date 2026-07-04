"""
Stage 2 / Tier 0 Item 4 -- Gradient Competitiveness Probe.

Implements the locked pre-registration
(Roadmap/Stage_2_Architecture_Investigation/Reports/Item4_PreRegistration.md).
Reproduces the EXACT training step (real X_train batches, real q_sample,
real per-sample CFG dropout, model.train() mode) to get gradients
representative of actual training dynamics -- NOT Items 1-3's synthetic
torch.randn pseudo-x_t convention, which was calibrated for a different
question (forward-pass response), not gradient dynamics during training.

No optimizer.step() is ever called -- gradient computation only, weights
never update. Verified via a state-dict checksum before/after the full
sweep (STOP CONDITION if it doesn't match) and a zero-grad/no-accumulation
reproducibility check (two identical-seed draws must give bit-identical
per-tensor gradient norms; STOP CONDITION if they don't).
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

CODE_DIR = Path(__file__).resolve().parents[1]  # Roadmap/.../Code/
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.io import load_config, get_logger, load_model_checkpoint  # noqa: E402
from step04_transformer_diffusion import (  # noqa: E402
    ECGDataset, _make_weighted_sampler, _load_class_labels, GaussianDiffusion,
)

OUT_DIR = (
    REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Outputs"
    / "stage2_tier0_item4_gradient_competitiveness"
)
FIG_DIR = (
    REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Figures"
    / "stage2_tier0_item4_gradient_competitiveness"
)

N_DRAWS = 10
FIXED_TIMESTEPS = [100, 500, 900]

TYPE_BUCKETS = {
    "embeddings": ["class_emb.weight", "patch_embed.lead_emb.weight"],
    "norms": ["norm1", "norm2", "final_norm"],
    "attention": ["attn."],
    "ffn": ["ff."],
    "adaLN": ["adaLN"],
    "projection": ["patch_embed.proj", "time_mlp", "unproj"],
}


def bucket_of(name: str) -> str:
    for bucket, patterns in TYPE_BUCKETS.items():
        for p in patterns:
            if p in name:
                return bucket
    return "other"


def state_dict_checksum(model) -> str:
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        h.update(name.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def load_training_data(cfg, log, device: str = "cpu"):
    """Reproduces train()'s own data-loading block exactly
    (step04_transformer_diffusion.py:707-796) -- reused, not reimplemented,
    for the parts that build X_train/train_labels/train_loader."""
    processed_dir = Path(cfg.paths.outputs.processed)
    cls_names_path = processed_dir / "class_names.json"
    cls_map_path = processed_dir / "class_mapping.json"
    with open(cls_names_path) as f:
        class_names = json.load(f)
    with open(cls_map_path) as f:
        class_mapping = json.load(f)

    X_train = np.load(str(processed_dir / "X_train.npy"))
    rec_ids_train = np.load(str(processed_dir / "record_ids_train.npy"))

    db_path = Path(cfg.paths.data.ptbxl) / "ptbxl_database.csv"
    ptbxl_db = pd.read_csv(str(db_path), index_col="ecg_id")

    vi_train, train_labels = _load_class_labels(rec_ids_train, ptbxl_db, class_mapping, class_names, log)
    X_train = X_train[vi_train]

    train_ds = ECGDataset(X_train, train_labels)
    sampler = _make_weighted_sampler(train_labels)
    d = cfg.diffusion
    train_loader = DataLoader(
        train_ds, batch_size=int(d.batch_size), sampler=sampler,
        num_workers=0, pin_memory=(device == "cuda"), drop_last=True,
    )
    return train_loader


def run_one_draw(model, diffusion, batch_x, batch_cls, device, p_uncond, t_diff_override=None):
    """One forward+backward pass, exactly reproducing the training step
    (step04_transformer_diffusion.py:863-887) -- model.train() mode,
    real per-sample CFG dropout, real q_sample noising, real MSE loss.
    NO optimizer.step() -- gradients computed and read, never applied."""
    model.zero_grad(set_to_none=True)
    B = batch_x.shape[0]

    if p_uncond > 0.0:
        null_mask = torch.bernoulli(torch.full((B,), p_uncond, device=device)).bool()
        batch_cls = batch_cls.clone()
        batch_cls[null_mask] = model.null_class_index

    if t_diff_override is not None:
        t_diff = torch.full((B,), t_diff_override, device=device, dtype=torch.long)
    else:
        t_diff = torch.randint(0, diffusion.T, (B,), device=device)

    noise = torch.randn_like(batch_x)
    x_t, _ = diffusion.q_sample(batch_x, t_diff, noise)

    eps_pred = model(x_t, t_diff, batch_cls)
    loss = F.mse_loss(eps_pred, noise)
    loss.backward()

    grad_norms = {name: float(p.grad.detach().norm().item())
                  for name, p in model.named_parameters() if p.grad is not None}
    return grad_norms, float(loss.item())


def main() -> None:
    cfg = load_config()
    log = get_logger("item4_gradient_probe", cfg=cfg)
    torch.manual_seed(0)

    loaded = load_model_checkpoint(cfg)
    if loaded is None:
        print("[BLOCKED] Checkpoint not found. Run Experiment 1 first.")
        return

    # Device selection: CUDA on the GPU server if available, else CPU.
    # A full batch_size=32 forward+backward pass with active dropout is
    # exactly the scenario already known to OOM on MPS (8GB unified memory,
    # per this project's own established constraint) -- Items 1-3's single-
    # sample forward-only probes never hit this, but Item 4's real training-
    # mimicking backward pass does. Measured on this Mac (CPU-forced, two
    # timed runs): ~1-5.5 min one-time data load (X_train.npy + per-record
    # CSV lookup; the second run was faster due to OS file-cache reuse, so
    # this is NOT fixed regardless of device -- it is fixed regardless of
    # *this specific run*, but caching means later runs on the same machine
    # are cheaper) + ~20-25s/draw x 40 draws on CPU. Expected substantially
    # faster per-draw on a real CUDA GPU; the data-loading component is
    # disk I/O / CPU-side tensor conversion and is not expected to speed up
    # from GPU compute alone.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = loaded.model.to(device)
    d = cfg.diffusion
    p_uncond = float(getattr(d, "p_uncond", 0.10))

    diffusion = GaussianDiffusion(T=int(d.T), beta_schedule=str(d.beta_schedule), device=device)

    log.info("Loading real training data (reproducing train()'s own data-loading block) ...")
    train_loader = load_training_data(cfg, log, device=device)
    train_iter = iter(train_loader)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    checksum_before = state_dict_checksum(model)

    # --- Reproducibility / zero-grad check: two identical-seed draws must match exactly ---
    model.train()
    torch.manual_seed(42)
    batch_x, batch_cls = next(train_iter)
    batch_x, batch_cls = batch_x.to(device), batch_cls.to(device)

    torch.manual_seed(9999)
    grad_norms_repro_1, _ = run_one_draw(model, diffusion, batch_x, batch_cls, device, p_uncond)
    torch.manual_seed(9999)
    grad_norms_repro_2, _ = run_one_draw(model, diffusion, batch_x, batch_cls, device, p_uncond)

    repro_match = all(
        abs(grad_norms_repro_1[k] - grad_norms_repro_2[k]) < 1e-9 for k in grad_norms_repro_1
    )
    log.info(f"Zero-grad reproducibility check: {'PASS' if repro_match else 'FAIL'}")
    if not repro_match:
        print("[STOP] Reproducibility check FAILED -- gradient accumulation bleed suspected. "
              "Aborting per pre-registration's STOP CONDITION. Do not trust any sweep numbers.")
        return

    # --- Primary design: N_DRAWS real per-sample-random-timestep draws ---
    primary_grad_norms = {}  # name -> list of norms across draws
    primary_losses = []
    for i in range(N_DRAWS):
        torch.manual_seed(1000 + i)
        try:
            batch_x, batch_cls = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch_x, batch_cls = next(train_iter)
        batch_x, batch_cls = batch_x.to(device), batch_cls.to(device)
        grad_norms, loss_val = run_one_draw(model, diffusion, batch_x, batch_cls, device, p_uncond)
        primary_losses.append(loss_val)
        for name, norm in grad_norms.items():
            primary_grad_norms.setdefault(name, []).append(norm)
        log.info(f"Primary draw {i}: loss={loss_val:.5f}, "
                 f"class_emb.weight ||grad||={grad_norms['class_emb.weight']:.6f}")

    # --- Secondary design: fixed timesteps, N_DRAWS each ---
    secondary_grad_norms = {t: {} for t in FIXED_TIMESTEPS}
    for t_val in FIXED_TIMESTEPS:
        for i in range(N_DRAWS):
            torch.manual_seed(2000 + t_val + i)
            try:
                batch_x, batch_cls = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch_x, batch_cls = next(train_iter)
            batch_x, batch_cls = batch_x.to(device), batch_cls.to(device)
            grad_norms, _ = run_one_draw(model, diffusion, batch_x, batch_cls, device, p_uncond,
                                          t_diff_override=t_val)
            for name, norm in grad_norms.items():
                secondary_grad_norms[t_val].setdefault(name, []).append(norm)
        log.info(f"Secondary design t={t_val}: class_emb.weight mean ||grad||="
                 f"{np.mean(secondary_grad_norms[t_val]['class_emb.weight']):.6f}")

    checksum_after = state_dict_checksum(model)
    weights_unchanged = (checksum_before == checksum_after)
    log.info(f"Weight-checksum check: {'PASS (unchanged)' if weights_unchanged else 'FAIL (CHANGED!)'}")
    if not weights_unchanged:
        print("[STOP] Weight-checksum check FAILED -- model weights changed during the sweep. "
              "This indicates an accidental optimizer.step() or in-place mutation. "
              "Aborting per pre-registration's STOP CONDITION -- do not trust any numbers.")
        return

    # --- Aggregate primary design ---
    all_names = sorted(primary_grad_norms.keys())
    pooled_mean = {name: float(np.mean(primary_grad_norms[name])) for name in all_names}
    pooled_std = {name: float(np.std(primary_grad_norms[name])) for name in all_names}

    class_emb_mean = pooled_mean["class_emb.weight"]
    other_means = sorted(v for k, v in pooled_mean.items() if k != "class_emb.weight")
    rank = sum(1 for v in other_means if v < class_emb_mean)
    percentile_rank = 100.0 * rank / len(other_means)

    bucket_summary = {}
    for name, mean_val in pooled_mean.items():
        b = bucket_of(name)
        bucket_summary.setdefault(b, []).append(mean_val)
    bucket_summary = {b: {"mean": float(np.mean(vs)), "n_tensors": len(vs)}
                       for b, vs in bucket_summary.items()}

    secondary_summary = {}
    for t_val in FIXED_TIMESTEPS:
        sm = {name: float(np.mean(vals)) for name, vals in secondary_grad_norms[t_val].items()}
        s_class_emb = sm["class_emb.weight"]
        s_others = sorted(v for k, v in sm.items() if k != "class_emb.weight")
        s_rank = sum(1 for v in s_others if v < s_class_emb)
        secondary_summary[t_val] = {
            "class_emb_mean_grad_norm": s_class_emb,
            "percentile_rank": 100.0 * s_rank / len(s_others),
        }

    raw = {
        "checksum_before": checksum_before,
        "checksum_after": checksum_after,
        "weights_unchanged": weights_unchanged,
        "reproducibility_check_passed": repro_match,
        "n_draws": N_DRAWS,
        "fixed_timesteps": FIXED_TIMESTEPS,
        "primary_losses": primary_losses,
        "primary_grad_norms_per_draw": primary_grad_norms,
        "secondary_grad_norms_per_draw": {str(t): v for t, v in secondary_grad_norms.items()},
    }
    with open(OUT_DIR / "gradient_probe_raw.json", "w") as f:
        json.dump(raw, f, indent=2)

    summary_df = pd.DataFrame({
        "parameter": all_names,
        "mean_grad_norm": [pooled_mean[n] for n in all_names],
        "std_grad_norm": [pooled_std[n] for n in all_names],
        "type_bucket": [bucket_of(n) for n in all_names],
    }).sort_values("mean_grad_norm", ascending=False).reset_index(drop=True)
    summary_df.to_csv(OUT_DIR / "gradient_probe_summary.csv", index=False)

    result = {
        "class_emb_mean_grad_norm": class_emb_mean,
        "class_emb_std_grad_norm": pooled_std["class_emb.weight"],
        "percentile_rank_within_other_83_tensors": percentile_rank,
        "n_other_tensors": len(other_means),
        "other_tensors_min": min(other_means),
        "other_tensors_median": float(np.median(other_means)),
        "other_tensors_max": max(other_means),
        "type_bucket_summary": bucket_summary,
        "secondary_fixed_timestep_summary": secondary_summary,
    }
    with open(OUT_DIR / "gradient_probe_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nclass_emb.weight percentile rank among {len(other_means)} other tensors: "
          f"{percentile_rank:.1f}%")


if __name__ == "__main__":
    main()
