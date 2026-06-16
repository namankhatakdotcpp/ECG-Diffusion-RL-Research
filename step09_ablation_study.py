"""
step09_ablation_study.py — Ablation study for RL reward component analysis.

PURPOSE
-------
Prove to reviewers that every reward component contributes.
Runs 6 RL fine-tuning variants (different reward weight configs / algorithms),
evaluates each with TSTR, morphology validity, and HRV consistency, then
produces Table 3 (LaTeX) for the paper.

ABLATION VARIANTS (run sequentially to avoid GPU OOM)
------------------------------------------------------
  1. full       — composite weights [0.3, 0.3, 0.2, 0.2]  (PPO) — our method
  2. diag_only  — weights [0.0, 0.0, 0.0, 1.0]            (PPO) — reward hacking demo
  3. no_diag    — weights [0.4, 0.4, 0.2, 0.0]            (PPO)
  4. no_morph   — weights [0.0, 0.4, 0.3, 0.3]            (PPO)
  5. no_hrv     — weights [0.4, 0.0, 0.3, 0.3]            (PPO)
  6. grpo_full  — composite weights [0.3, 0.3, 0.2, 0.2]  (GRPO)

RECORDED METRICS (mean ± std across 3 seeds)
--------------------------------------------
  TSTR F1     — macro F1 on real test set, classifier trained on synthetic
  Morph%      — % generated ECGs with PQRST intervals within mean ± 2σ reference
  HRV%        — % generated ECGs with SDNN/RMSSD within NORM reference range
  DTW         — nearest-neighbour DTW distance vs real test ECGs (Lead II)

OUTPUTS
-------
  outputs/results/ablation_table.tex      — LaTeX Table 3
  outputs/results/ablation_results.json   — full numeric results
  outputs/models/ablation_{name}.pt       — per-ablation best checkpoint
  outputs/models/rl_diag_only.pt          — alias for step08 optional model
  logs/ablation_{name}_log.csv            — per-iteration RL metrics
"""

from __future__ import annotations

import copy
import csv
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, get_logger, set_seed
from step04_transformer_diffusion import ECGTransformerDiffusion, GaussianDiffusion, EMA
from step05_baseline_eval import (
    _metric_dtw,
    _metric_morphology,
    _metric_tstr_trtr,
    _load_class_labels,
    _load_real_data,
)
from step06_reward_function import (
    ABLATION_CONFIGS,
    get_reward,
    _extract_hrv,
)
from step07_rl_finetuning import DDPOTrainer, GRPOTrainer, _ddim_sample


# ──────────────────────────────────────────────────────────────────────────────
# Ablation definitions
# ──────────────────────────────────────────────────────────────────────────────

ABLATION_DEFS: list[dict] = [
    {
        "name":          "full",
        "reward_config": "full",
        "algorithm":     "ppo",
        "label":         "Ours (composite)",
    },
    {
        "name":          "diag_only",
        "reward_config": "diag_only",
        "algorithm":     "ppo",
        "label":         "Diag-only (hacking)",
    },
    {
        "name":          "no_diag",
        "reward_config": "no_diag",
        "algorithm":     "ppo",
        "label":         "No diagnostic",
    },
    {
        "name":          "no_morph",
        "reward_config": "no_morph",
        "algorithm":     "ppo",
        "label":         "No morphology",
    },
    {
        "name":          "no_hrv",
        "reward_config": "no_hrv",
        "algorithm":     "ppo",
        "label":         "No HRV",
    },
    {
        "name":          "grpo_full",
        "reward_config": "full",
        "algorithm":     "grpo",
        "label":         "GRPO (full)",
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# RL training for one ablation variant
# ──────────────────────────────────────────────────────────────────────────────

def _run_rl_ablation(
    name:          str,
    reward_config: str,
    algorithm:     str,
    cfg,
    X_train:       np.ndarray,
    class_names:   list[str],
    device:        str,
    log,
) -> Path:
    """
    Run RL fine-tuning for one ablation variant.

    Reuses DDPOTrainer / GRPOTrainer from step07 with a fresh copy of the
    pre-trained baseline loaded every call.  Saves best checkpoint to
    outputs/models/ablation_{name}.pt and returns that path.
    """
    models_dir = Path(cfg.paths.outputs.models)
    logs_dir   = Path(cfg.paths.logs)
    ckpt_out   = models_dir / f"ablation_{name}.pt"

    # ── Load pre-trained weights ──────────────────────────────────────────────
    best_path = models_dir / "diffusion_best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"{best_path} not found. Run step04 first.")
    ckpt = torch.load(str(best_path), map_location=device)

    n_classes = len(class_names)

    policy_model = ECGTransformerDiffusion(cfg, n_classes=n_classes).to(device)
    policy_model.load_state_dict(ckpt["model"])

    frozen_model = ECGTransformerDiffusion(cfg, n_classes=n_classes).to(device)
    frozen_model.load_state_dict(ckpt["model"])
    for p in frozen_model.parameters():
        p.requires_grad_(False)
    frozen_model.eval()

    frozen_checksum = float(sum(p.data.sum().item() for p in frozen_model.parameters()))

    # ── Diffusion schedule ────────────────────────────────────────────────────
    diffusion = GaussianDiffusion(
        T=int(cfg.diffusion.T),
        beta_schedule=str(cfg.diffusion.beta_schedule),
        device=device,
    )

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema = EMA(policy_model, decay=float(cfg.diffusion.ema_decay))
    if "ema_shadow" in ckpt:
        ema.shadow = {k: v.to(device) for k, v in ckpt["ema_shadow"].items()}

    # ── Reward function with this variant's weights ───────────────────────────
    reward_fn = get_reward(
        config_name=reward_config,
        cfg=cfg,
        X_train=X_train,
        class_names=class_names,
        device=device,
    )

    # ── Optimiser ─────────────────────────────────────────────────────────────
    rl  = cfg.rl
    opt = torch.optim.AdamW(
        policy_model.parameters(),
        lr=float(rl.rl_lr),
        weight_decay=float(cfg.diffusion.weight_decay),
    )
    current_lr = float(rl.rl_lr)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer_kwargs = dict(
        policy_model=policy_model,
        frozen_model=frozen_model,
        diffusion=diffusion,
        reward_fn=reward_fn,
        ema=ema,
        class_names=class_names,
        cfg=cfg,
        device=device,
        opt=opt,
    )
    if algorithm == "ppo":
        trainer = DDPOTrainer(**trainer_kwargs)
    elif algorithm == "grpo":
        trainer = GRPOTrainer(**trainer_kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm!r}")

    n_rollouts   = int(rl.n_rollouts)
    n_ppo_epochs = int(rl.n_ppo_epochs)
    n_iterations = int(rl.rl_iterations)

    # ── CSV log ───────────────────────────────────────────────────────────────
    log_path = logs_dir / f"ablation_{name}_log.csv"
    log_fh   = open(log_path, "w", newline="")
    csv_w    = csv.writer(log_fh)
    csv_w.writerow([
        "iter", "class",
        "reward_total", "r_morph", "r_hrv", "r_real", "r_diag",
        "kl", "loss", "lr",
    ])

    # ── RL loop ───────────────────────────────────────────────────────────────
    KL_THRESH    = 1.0
    MORPH_THRESH = 0.3
    morph_low    = 0
    best_reward  = -float("inf")
    rng          = np.random.default_rng(int(cfg.seeds[0]))

    for it in range(1, n_iterations + 1):
        class_idx  = int(rng.integers(0, n_classes))
        class_name = class_names[class_idx]

        if algorithm == "ppo":
            rollouts = trainer.collect_rollouts(n_rollouts, class_idx)
            update   = trainer.ppo_update(rollouts, n_epochs=n_ppo_epochs)
            mean_rd  = {
                "total":   float(np.mean([ro["reward"]["total"]   for ro in rollouts])),
                "r_morph": float(np.mean([ro["reward"]["r_morph"] for ro in rollouts])),
                "r_hrv":   float(np.mean([ro["reward"]["r_hrv"]   for ro in rollouts])),
                "r_real":  float(np.mean([ro["reward"]["r_real"]  for ro in rollouts])),
                "r_diag":  float(np.mean([ro["reward"]["r_diag"]  for ro in rollouts])),
            }
            kl, loss = update["kl"], update["loss"]
        else:
            update  = trainer.grpo_update(class_idx)
            mean_rd = update["reward"]
            kl, loss = update["kl"], update["loss"]

        csv_w.writerow([
            it, class_name,
            f"{mean_rd['total']:.5f}", f"{mean_rd['r_morph']:.5f}",
            f"{mean_rd['r_hrv']:.5f}",  f"{mean_rd['r_real']:.5f}",
            f"{mean_rd['r_diag']:.5f}", f"{kl:.5f}", f"{loss:.5f}",
            f"{current_lr:.2e}",
        ])
        log_fh.flush()

        if it % 10 == 0 or it == 1:
            log.info(
                f"    [{it:03d}/{n_iterations}] r={mean_rd['total']:.4f} "
                f"kl={kl:.4f} loss={loss:.5f}"
            )

        # KL safety alarm
        if kl > KL_THRESH:
            current_lr *= 0.5
            for pg in opt.param_groups:
                pg["lr"] = current_lr
            log.warning(f"    ⚠ KL={kl:.4f} > {KL_THRESH} — halving lr → {current_lr:.2e}")

        # Morph collapse alarm
        if mean_rd["r_morph"] < MORPH_THRESH:
            morph_low += 1
            if morph_low >= 3:
                log.warning(f"    ⚠ ALERT: r_morph collapsed for {name} — possible reward hacking")
                morph_low = 0
        else:
            morph_low = 0

        # Best checkpoint
        if mean_rd["total"] > best_reward:
            best_reward = mean_rd["total"]
            torch.save(
                {
                    "ablation":    name,
                    "iter":        it,
                    "model":       policy_model.state_dict(),
                    "ema_shadow":  ema.shadow,
                    "best_reward": best_reward,
                    "class_names": class_names,
                    "n_classes":   n_classes,
                    "algorithm":   algorithm,
                    "reward_config": reward_config,
                },
                str(ckpt_out),
            )

    log_fh.close()

    # ── Frozen model integrity assertion ──────────────────────────────────────
    frozen_final = float(sum(p.data.sum().item() for p in frozen_model.parameters()))
    delta = abs(frozen_checksum - frozen_final)
    assert delta < 1e-3, (
        f"[{name}] Frozen model changed during RL! Δ={delta:.6f}"
    )
    log.info(f"    ✓ Frozen model unchanged (Δ={delta:.2e}). Best reward: {best_reward:.4f}")

    # ── Copy diag_only to rl_diag_only.pt for step08 ─────────────────────────
    if name == "diag_only":
        import shutil
        alias = models_dir / "rl_diag_only.pt"
        shutil.copy2(str(ckpt_out), str(alias))
        log.info(f"    → Copied to {alias.name} (for step08)")

    return ckpt_out


# ──────────────────────────────────────────────────────────────────────────────
# ECG generation from an ablation checkpoint
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _generate_ecgs(
    ckpt_path:   Path,
    class_names: list[str],
    n_per_class: int,
    cfg,
    device:      str,
    log,
) -> dict[str, np.ndarray]:
    """
    Load an ablation checkpoint and generate n_per_class synthetic ECGs per class.
    Returns dict {class_name: (n_per_class, 1000, 12)}.
    Uses EMA weights when available.
    """
    ckpt = torch.load(str(ckpt_path), map_location=device)
    n_classes = len(class_names)

    model = ECGTransformerDiffusion(cfg, n_classes=n_classes).to(device)
    model.load_state_dict(ckpt["model"])

    # Apply EMA weights if present
    if "ema_shadow" in ckpt:
        shadow = {k: v.to(device) for k, v in ckpt["ema_shadow"].items()}
        param_names = {name for name, _ in model.named_parameters()}
        for k, v in shadow.items():
            if k in param_names:
                dict(model.named_parameters())[k].data.copy_(v)
    model.eval()

    diffusion = GaussianDiffusion(
        T=int(cfg.diffusion.T),
        beta_schedule=str(cfg.diffusion.beta_schedule),
        device=device,
    )

    n_ddim_steps = int(cfg.diffusion.ddim_steps)
    batch_size   = 16
    generated    = {}

    for cls_idx, cls_name in enumerate(class_names):
        log.info(f"    Generating {n_per_class} × {cls_name} …")
        samples: list[np.ndarray] = []
        remaining = n_per_class

        while remaining > 0:
            B          = min(batch_size, remaining)
            cls_tensor = torch.full((B,), cls_idx, dtype=torch.long, device=device)
            x0         = _ddim_sample(model, diffusion, cls_tensor, n_ddim_steps, cfg, device)
            samples.append(x0.cpu().numpy().transpose(0, 2, 1))   # (B, 1000, 12)
            remaining -= B

        generated[cls_name] = np.concatenate(samples, axis=0)[:n_per_class]

    return generated


# ──────────────────────────────────────────────────────────────────────────────
# HRV consistency metric
# ──────────────────────────────────────────────────────────────────────────────

def _metric_hrv_pct(
    gen:        dict[str, np.ndarray],
    hrv_stats:  dict,
    class_names: list[str],
    cfg,
    log,
) -> float:
    """
    % of generated ECGs whose SDNN and RMSSD fall within mean ± 2σ of the
    NORM reference (since autonomic HRV should be present in all classes).
    Returns overall % across all classes (float in [0, 100]).
    """
    fs  = float(cfg.ptbxl.sampling_rate)
    ref = hrv_stats.get("NORM", {})

    if not ref:
        log.warning("    No HRV reference (NORM) — HRV% set to 0.0")
        return 0.0

    n_ok  = 0
    n_tot = 0
    LEAD_II = 1

    for cls_name in class_names:
        ecgs = gen[cls_name]    # (N, 1000, 12)
        n_subsample = min(50, len(ecgs))
        rng  = np.random.default_rng(42)
        idxs = rng.choice(len(ecgs), size=n_subsample, replace=False)

        for i in idxs:
            result = _extract_hrv(ecgs[i, :, LEAD_II], fs)
            if result is None:
                n_tot += 1
                continue

            ok = True
            for key in ("sdnn_ms", "rmssd_ms"):
                if key not in ref or key not in result:
                    continue
                lo = ref[key]["mean"] - 2.0 * ref[key]["std"]
                hi = ref[key]["mean"] + 2.0 * ref[key]["std"]
                if not (lo <= result[key] <= hi):
                    ok = False
                    break
            n_ok  += int(ok)
            n_tot += 1

    if n_tot == 0:
        return 0.0
    return 100.0 * n_ok / n_tot


# ──────────────────────────────────────────────────────────────────────────────
# Evaluate one ablation checkpoint (single seed)
# ──────────────────────────────────────────────────────────────────────────────

def _eval_one_seed(
    gen:         dict[str, np.ndarray],
    X_train:     np.ndarray,
    y_train:     np.ndarray,
    X_test:      np.ndarray,
    y_test:      np.ndarray,
    class_names: list[str],
    morph_stats: dict,
    hrv_stats:   dict,
    cfg,
    seed:        int,
    device:      str,
    log,
) -> dict:
    """Run all ablation metrics for one pre-generated dict of synthetic ECGs."""
    rng = np.random.default_rng(seed)

    # DTW
    log.info("      DTW …")
    dtw = _metric_dtw(
        gen, X_test, y_test, class_names,
        n_subsample=int(cfg.eval.dtw_subsample),
        rng=rng,
    )

    # Morphology %
    log.info("      Morphology% …")
    morph = _metric_morphology(
        gen, morph_stats, class_names,
        fs=float(cfg.ptbxl.sampling_rate),
        n_eval=int(cfg.eval.n_morphology_eval),
        rng=rng,
        log=log,
    )

    # HRV %
    log.info("      HRV% …")
    hrv_pct = _metric_hrv_pct(gen, hrv_stats, class_names, cfg, log)

    # TSTR
    log.info("      TSTR …")
    set_seed(seed)
    tstr, _ = _metric_tstr_trtr(
        gen, X_train, y_train, X_test, y_test, class_names,
        n_per_class=int(cfg.eval.n_synthetic_per_class),
        cfg=cfg,
        device=device,
        log=log,
    )

    return {
        "dtw":     dtw["overall"],
        "morph":   morph["overall"],
        "hrv_pct": hrv_pct,
        "tstr_f1": tstr["macro_f1"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# LaTeX Table 3
# ──────────────────────────────────────────────────────────────────────────────

def _make_ablation_table(
    all_results: dict[str, list[dict]],
    ablation_defs: list[dict],
    results_dir: Path,
    log,
) -> str:
    """
    Generate LaTeX Table 3 — ablation study.

    Columns: Config | r_morph | r_HRV | r_real | r_diag | TSTR F1 | Morph% | HRV% | DTW
    Checkmarks indicate which reward components are active (weight > 0).
    Best value in each numeric column is bolded.
    """
    CHECK = r"\checkmark"
    CROSS = r"$\times$"

    # Collect per-ablation mean/std from 3 seeds
    summaries: list[dict] = []
    for abl in ablation_defs:
        name    = abl["name"]
        results = all_results[name]      # list of seed dicts

        def _ms(key: str) -> tuple[float, float]:
            vals = [r[key] for r in results if not np.isnan(r.get(key, float("nan")))]
            if not vals:
                return float("nan"), 0.0
            return float(np.mean(vals)), float(np.std(vals))

        w = ABLATION_CONFIGS[abl["reward_config"]]
        summaries.append({
            "label":    abl["label"],
            "w_morph":  w.get("morph", 0.0),
            "w_hrv":    w.get("hrv",   0.0),
            "w_real":   w.get("real",  0.0),
            "w_diag":   w.get("diag",  0.0),
            "tstr_mean":  _ms("tstr_f1")[0],
            "tstr_std":   _ms("tstr_f1")[1],
            "morph_mean": _ms("morph")[0],
            "morph_std":  _ms("morph")[1],
            "hrv_mean":   _ms("hrv_pct")[0],
            "hrv_std":    _ms("hrv_pct")[1],
            "dtw_mean":   _ms("dtw")[0],
            "dtw_std":    _ms("dtw")[1],
        })

    # Find best per numeric metric (↑ = higher better; ↓ = lower better)
    tstr_vals  = [s["tstr_mean"]  for s in summaries]
    morph_vals = [s["morph_mean"] for s in summaries]
    hrv_vals   = [s["hrv_mean"]   for s in summaries]
    dtw_vals   = [s["dtw_mean"]   for s in summaries]

    def _best_idx(vals: list[float], higher_is_better: bool) -> int:
        clean = [v for v in vals if not np.isnan(v)]
        if not clean:
            return -1
        target = max(clean) if higher_is_better else min(clean)
        for i, v in enumerate(vals):
            if not np.isnan(v) and abs(v - target) < 1e-9:
                return i
        return -1

    best_tstr  = _best_idx(tstr_vals,  True)
    best_morph = _best_idx(morph_vals, True)
    best_hrv   = _best_idx(hrv_vals,   True)
    best_dtw   = _best_idx(dtw_vals,   False)

    def _fmt(mean: float, std: float, is_best: bool, pct: bool = False) -> str:
        if np.isnan(mean):
            return "—"
        unit  = "\\%" if pct else ""
        inner = f"{mean:.3f}{{\\scriptsize$\\pm${std:.3f}}}{unit}"
        return f"\\textbf{{{inner}}}" if is_best else inner

    def _check(w: float) -> str:
        return CHECK if w > 0.0 else CROSS

    # Build table rows
    body = ""
    for i, s in enumerate(summaries):
        row_parts = [
            s["label"].replace("_", r"\_"),
            _check(s["w_morph"]),
            _check(s["w_hrv"]),
            _check(s["w_real"]),
            _check(s["w_diag"]),
            _fmt(s["tstr_mean"],  s["tstr_std"],  i == best_tstr),
            _fmt(s["morph_mean"], s["morph_std"], i == best_morph, pct=True),
            _fmt(s["hrv_mean"],   s["hrv_std"],   i == best_hrv,   pct=True),
            _fmt(s["dtw_mean"],   s["dtw_std"],   i == best_dtw),
        ]
        body += " & ".join(row_parts) + " \\\\\n"

    header_cols = [
        "Config",
        "$r_{\\text{morph}}$",
        "$r_{\\text{HRV}}$",
        "$r_{\\text{real}}$",
        "$r_{\\text{diag}}$",
        "TSTR F1 $\\uparrow$",
        "Morph\\% $\\uparrow$",
        "HRV\\% $\\uparrow$",
        "DTW $\\downarrow$",
    ]
    header_row = " & ".join(header_cols) + " \\\\\n"
    col_align  = "l" + "c" * 4 + "r" * 4

    caption = (
        "Ablation study: effect of each reward component on synthetic ECG quality. "
        "\\checkmark\\ = component active (weight $>0$); "
        "$\\times$ = component disabled. "
        "Full composite: $w_{\\mathrm{morph}}\\!=\\!0.3$, "
        "$w_{\\mathrm{HRV}}\\!=\\!0.3$, "
        "$w_{\\mathrm{real}}\\!=\\!0.2$, "
        "$w_{\\mathrm{diag}}\\!=\\!0.2$. "
        "Results are mean ${\\scriptstyle\\pm}$ std across 3 seeds. "
        "\\textbf{Bold} = best per column."
    )

    latex = (
        "\\begin{table}[t]\n"
        + "\\centering\n"
        + f"\\caption{{{caption}}}\n"
        + "\\label{tab:ablation}\n"
        + f"\\begin{{tabular}}{{{col_align}}}\n"
        + "\\toprule\n"
        + header_row
        + "\\midrule\n"
        + body
        + "\\bottomrule\n"
        + "\\end{tabular}\n"
        + "\\end{table}\n"
    )

    out = results_dir / "ablation_table.tex"
    out.write_text(latex)
    log.info(f"Ablation table → {out}")
    return latex


# ──────────────────────────────────────────────────────────────────────────────
# Main ablation loop
# ──────────────────────────────────────────────────────────────────────────────

def ablate(cfg, log) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # ── Paths ─────────────────────────────────────────────────────────────────
    processed_dir = Path(cfg.paths.outputs.processed)
    models_dir    = Path(cfg.paths.outputs.models)
    results_dir   = Path(cfg.paths.outputs.results)
    logs_dir      = Path(cfg.paths.logs)
    for d in (models_dir, results_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Load class info ───────────────────────────────────────────────────────
    with open(processed_dir / "class_names.json") as f:
        class_names = json.load(f)
    log.info(f"Classes ({len(class_names)}): {class_names}")

    # ── Load shared data ──────────────────────────────────────────────────────
    log.info("Loading data …")
    # Raw unfiltered X_train — used for reward PCA fitting (matches step07 behaviour)
    X_train_raw = np.load(str(processed_dir / "X_train.npy"))

    # Labelled splits — used for TSTR and metric evaluation
    data    = _load_real_data(cfg, log)
    X_train = data["X_train"]   # filtered to records with valid labels
    y_train = data["y_train"]
    X_test  = data["X_test"]
    y_test  = data["y_test"]

    # ── Load reference stats ──────────────────────────────────────────────────
    def _load_json(path: Path) -> dict:
        return json.load(open(path)) if path.exists() else {}

    morph_stats = _load_json(processed_dir / "morphology_stats.json")
    hrv_stats   = _load_json(processed_dir / "hrv_stats.json")

    n_per_class = int(cfg.eval.n_synthetic_per_class)
    seeds       = list(cfg.eval.seeds)

    # ── Run all ablations sequentially ────────────────────────────────────────
    all_results:    dict[str, list[dict]] = {}
    ckpt_paths:     dict[str, Path]       = {}

    n_ablations = len(ABLATION_DEFS)
    for abl_idx, abl in enumerate(ABLATION_DEFS, start=1):
        name          = abl["name"]
        reward_config = abl["reward_config"]
        algorithm     = abl["algorithm"]
        label         = abl["label"]

        print(f"Running ablation {abl_idx}/{n_ablations}: {name} …")
        log.info("=" * 60)
        log.info(
            f"Ablation {abl_idx}/{n_ablations}: {label}"
            f"  (reward={reward_config}, algorithm={algorithm})"
        )
        log.info("=" * 60)

        # ── RL fine-tuning ────────────────────────────────────────────────────
        ckpt_path = _run_rl_ablation(
            name=name,
            reward_config=reward_config,
            algorithm=algorithm,
            cfg=cfg,
            X_train=X_train_raw,   # unfiltered, for reward PCA (matches step07)
            class_names=class_names,
            device=device,
            log=log,
        )
        ckpt_paths[name] = ckpt_path

        # ── Evaluate across seeds ─────────────────────────────────────────────
        seed_results: list[dict] = []

        for seed_idx, seed in enumerate(seeds):
            log.info(f"  Seed {seed_idx + 1}/{len(seeds)} (seed={seed}) …")
            set_seed(seed)

            # Generate synthetic ECGs (same model, different RNG)
            gen = _generate_ecgs(ckpt_path, class_names, n_per_class, cfg, device, log)

            # Compute all metrics
            metrics = _eval_one_seed(
                gen=gen,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                class_names=class_names,
                morph_stats=morph_stats,
                hrv_stats=hrv_stats,
                cfg=cfg,
                seed=seed,
                device=device,
                log=log,
            )
            log.info(
                f"    seed={seed}  TSTR={metrics['tstr_f1']:.4f}  "
                f"Morph={metrics['morph']:.1f}%  "
                f"HRV={metrics['hrv_pct']:.1f}%  "
                f"DTW={metrics['dtw']:.4f}"
            )
            seed_results.append(metrics)

        all_results[name] = seed_results

        # Summary for this ablation
        mean_tstr  = float(np.mean([r["tstr_f1"]  for r in seed_results]))
        mean_morph = float(np.mean([r["morph"]     for r in seed_results]))
        mean_hrv   = float(np.mean([r["hrv_pct"]  for r in seed_results]))
        mean_dtw   = float(np.mean([r["dtw"]       for r in seed_results]))
        log.info(
            f"  [{name}] mean: TSTR={mean_tstr:.4f}  "
            f"Morph={mean_morph:.1f}%  "
            f"HRV={mean_hrv:.1f}%  "
            f"DTW={mean_dtw:.4f}"
        )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_out = results_dir / "ablation_results.json"
    with open(json_out, "w") as f:
        json.dump(
            {
                "ablation_defs": ABLATION_DEFS,
                "seeds":         seeds,
                "n_per_class":   n_per_class,
                "results":       {
                    name: [
                        {k: (float(v) if not np.isnan(float(v)) else None) for k, v in r.items()}
                        for r in seed_results
                    ]
                    for name, seed_results in all_results.items()
                },
            },
            f, indent=2,
        )
    log.info(f"Results JSON → {json_out}")

    # ── Generate LaTeX Table 3 ─────────────────────────────────────────────────
    _make_ablation_table(all_results, ABLATION_DEFS, results_dir, log)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    log = get_logger("step09_ablation_study", cfg=cfg)
    set_seed(int(cfg.seeds[0]))

    ablate(cfg, log)
    print("✓ Ablation complete. See outputs/results/ablation_table.tex")


if __name__ == "__main__":
    main()
