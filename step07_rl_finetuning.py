"""
step07_rl_finetuning.py — PPO / GRPO RL fine-tuning of the ECG diffusion model.

PURPOSE
-------
Fine-tune the pre-trained baseline diffusion model (diffusion_best.pt) with
reinforcement learning to maximise the clinical reward (step06) while staying
close to the pre-trained distribution via KL regularisation.

MDP FRAMING
-----------
  State   : x_t  — noisy ECG at diffusion timestep t
  Action  : ε_θ(x_t, t, c) — noise predicted by the policy model
  Reward  : ClinicalReward(x₀) at the END of the denoising chain (episodic)
  Policy  : diffusion model θ (updated) vs frozen reference θ₀

POLICY LOG-PROBABILITY PROXY
-----------------------------
Computing exact log probabilities along the full denoising chain is numerically
unstable at high noise levels (ᾱ_t ≈ 0 ⟹ x̂₀ blows up).  We instead use
the diffusion ELBO (score-matching objective) as a differentiable proxy:

    log π_θ(x₀) ≈ −E_{t ~ U[0,T], ε ~ N(0,I)}[‖ε_θ(x_t, t, c) − ε‖²]

This is equivalent to −ELBO and has well-behaved gradients at all noise levels.
The same proxy is used in DPOK, RAFT, and SPIN.

ALGORITHMS
----------
  A — DDPO (PPO variant, default):
        ratio = exp(log_π_θ_new − log_π_θ_old)
        A     = r − EMA_baseline
        loss  = −min(ratio·A, clip(ratio, 1−ε, 1+ε)·A) + β·KL

  B — GRPO (group-relative advantage):
        Generate G ECGs for the same condition
        A_i   = (r_i − mean(r)) / (std(r) + 1e-8)
        loss  = −(1/G) Σᵢ Aᵢ·log π_θ(xᵢ) + β·KL

KL REGULARISATION
-----------------
KL(π_θ ‖ π_θ₀) ≈ E_t[‖ε_θ(x_t, t, c) − ε_θ₀(x_t, t, c)‖²]
(L2 distance between noise predictions, averaged over random t and x₀).

OUTPUTS
-------
  outputs/models/diffusion_rl_best.pt         — best policy checkpoint
  outputs/models/rl_ckpt_iter{N:04d}.pt       — periodic checkpoints
  logs/rl_training_log.csv                    — per-iteration metrics
  outputs/results/rl_progress_iter{N:04d}.png — visual progress snapshots
"""

from __future__ import annotations

import csv
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
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, get_logger, set_seed
from step04_transformer_diffusion import ECGTransformerDiffusion, GaussianDiffusion, EMA
from step06_reward_function import ClinicalReward, get_reward

# ──────────────────────────────────────────────────────────────────────────────
# Publication figure style
# ──────────────────────────────────────────────────────────────────────────────

PUBSTYLE: dict = {
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "font.size":          9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
}


# ──────────────────────────────────────────────────────────────────────────────
# Core RL primitives
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _ddim_sample(
    model:       nn.Module,
    diffusion:   GaussianDiffusion,
    class_label: Tensor,      # (B,) long
    n_steps:     int,
    cfg,
    device:      str,
) -> Tensor:
    """Fast deterministic DDIM sampling (η=0) used for rollout generation."""
    B       = class_label.shape[0]
    sig_len = int(cfg.ptbxl.signal_length)
    T       = int(cfg.diffusion.T)

    ts = torch.linspace(T - 1, 0, n_steps, dtype=torch.long, device=device)
    x  = torch.randn(B, 12, sig_len, device=device)

    for i, t_curr in enumerate(ts):
        t_batch = t_curr.expand(B)
        eps     = model(x, t_batch, class_label)
        ab_t    = diffusion.alpha_bar[t_curr]
        x0_hat  = (x - (1.0 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp(min=1e-8)
        x0_hat  = x0_hat.clamp(-4.0, 4.0)
        ab_prev = (
            diffusion.alpha_bar[ts[i + 1]] if i + 1 < n_steps
            else torch.ones(1, device=device)
        )
        x = ab_prev.sqrt() * x0_hat + (1.0 - ab_prev).sqrt() * eps

    return x   # (B, 12, sig_len)


def _score_log_prob(
    model:       nn.Module,
    x0:          Tensor,      # (1, 12, sig_len)  — detach from rollout graph
    diffusion:   GaussianDiffusion,
    class_label: Tensor,
    device:      str,
    K:           int   = 10,
    requires_grad: bool = False,
) -> Tensor:
    """
    Diffusion ELBO proxy for log π_θ(x₀):

        log π_θ(x₀) ≈ −(1/K) Σ_{k=1}^{K} ‖ε_θ(x_{t_k}, t_k, c) − ε_k‖²

    where t_k ~ Uniform[0, T), ε_k ~ N(0,I), x_{t_k} = q_sample(x₀, t_k, ε_k).

    Gradients flow cleanly through the standard forward pass.  Returns a scalar
    Tensor; differentiable when requires_grad=True.
    """
    B = x0.shape[0]
    T = diffusion.T
    x0_detach = x0.detach()   # ECG is a fixed "outcome" for this log-prob computation

    total = torch.zeros(1, device=device)

    ts      = torch.randint(0, T, (K,), device=device)
    noises  = [torch.randn_like(x0_detach) for _ in range(K)]

    ctx = torch.enable_grad() if requires_grad else torch.no_grad()
    with ctx:
        for k in range(K):
            t_b     = ts[k:k+1].expand(B)
            noise   = noises[k]
            x_t, _  = diffusion.q_sample(x0_detach, t_b, noise)
            eps_pred = model(x_t, t_b, class_label)
            mse      = F.mse_loss(eps_pred, noise, reduction="mean")
            total    = total - mse        # higher log prob ↔ lower MSE

    return total / K                      # per-element, ≈ −0.5 for perfect model


def _score_kl(
    policy:      nn.Module,
    frozen:      nn.Module,
    x0:          Tensor,
    diffusion:   GaussianDiffusion,
    class_label: Tensor,
    device:      str,
    K:           int = 10,
) -> Tensor:
    """
    KL(π_θ ‖ π_θ₀) ≈ (1/K) Σ_k ‖ε_θ(x_t_k, t_k) − ε_θ₀(x_t_k, t_k)‖²

    Differentiable w.r.t. policy parameters.
    """
    B = x0.shape[0]
    T = diffusion.T
    x0_detach = x0.detach()

    ts     = torch.randint(0, T, (K,), device=device)
    noises = [torch.randn_like(x0_detach) for _ in range(K)]

    total = torch.zeros(1, device=device)

    for k in range(K):
        t_b      = ts[k:k+1].expand(B)
        noise    = noises[k]
        x_t, _   = diffusion.q_sample(x0_detach, t_b, noise)
        eps_pol  = policy(x_t, t_b, class_label)
        with torch.no_grad():
            eps_frz = frozen(x_t, t_b, class_label)
        total = total + F.mse_loss(eps_pol, eps_frz, reduction="mean")

    return total / K


# ──────────────────────────────────────────────────────────────────────────────
# PPO trainer (DDPO variant)
# ──────────────────────────────────────────────────────────────────────────────

class DDPOTrainer:
    """
    Denoising Diffusion Policy Optimization — PPO for diffusion fine-tuning.

    Rollout:  deterministic DDIM → x₀ → clinical reward + score-matching log prob
    Update:   clipped PPO objective + KL regularisation (n_ppo_epochs passes)
    """

    def __init__(
        self,
        policy_model:  nn.Module,
        frozen_model:  nn.Module,
        diffusion:     GaussianDiffusion,
        reward_fn:     ClinicalReward,
        ema:           EMA,
        class_names:   list[str],
        cfg,
        device:        str,
        opt:           torch.optim.Optimizer,
    ) -> None:
        self.policy      = policy_model
        self.frozen      = frozen_model
        self.diffusion   = diffusion
        self.reward_fn   = reward_fn
        self.ema         = ema
        self.class_names = class_names
        self.cfg         = cfg
        self.device      = device
        self.opt         = opt

        rl = cfg.rl
        self.ppo_clip       = float(rl.ppo_clip)
        self.kl_beta        = float(rl.kl_beta)
        self.n_steps        = int(rl.ddim_rollout_steps)
        self.grad_clip      = float(rl.grad_clip)
        self.baseline       = 0.0
        self.baseline_decay = float(rl.baseline_decay)
        self.lp_K           = 10    # timesteps sampled for log-prob proxy

    @torch.no_grad()
    def collect_rollouts(self, n_rollouts: int, class_idx: int) -> list[dict]:
        """
        Generate n_rollouts ECGs with DDIM, compute rewards and old log probs.
        Old log probs are pre-computed under the current policy (before update).
        """
        class_label = torch.tensor([class_idx], dtype=torch.long, device=self.device)
        class_name  = self.class_names[class_idx]

        self.policy.eval()
        rollouts = []

        for _ in range(n_rollouts):
            # ── Generate x₀ ──────────────────────────────────────────────────
            x0 = _ddim_sample(
                self.policy, self.diffusion, class_label,
                self.n_steps, self.cfg, self.device,
            )

            # ── Clinical reward ────────────────────────────────────────────────
            ecg_np = x0.squeeze(0).T.cpu().numpy()   # (sig_len, 12)
            try:
                reward_dict = self.reward_fn.compute(ecg_np, class_name, class_idx)
            except Exception:
                reward_dict = {
                    "total": 0.0, "r_morph": 0.0,
                    "r_hrv": 0.0, "r_real": 0.0, "r_diag": 0.5,
                }

            # ── Old log prob (score-matching proxy, no grad) ──────────────────
            lp_old = _score_log_prob(
                self.policy, x0, self.diffusion, class_label, self.device,
                K=self.lp_K, requires_grad=False,
            ).item()

            # ── Frozen log prob (for KL estimate) ─────────────────────────────
            lp_frozen = _score_log_prob(
                self.frozen, x0, self.diffusion, class_label, self.device,
                K=self.lp_K, requires_grad=False,
            ).item()

            rollouts.append({
                "x0":         x0.detach(),
                "lp_old":     lp_old,
                "lp_frozen":  lp_frozen,
                "reward":     reward_dict,
                "class_idx":  class_idx,
                "class_name": class_name,
            })

        return rollouts

    def ppo_update(self, rollouts: list[dict], n_epochs: int = 4) -> dict[str, float]:
        """Clipped PPO loss + KL regularisation; returns mean metrics."""
        class_idx   = rollouts[0]["class_idx"]
        class_label = torch.tensor([class_idx], dtype=torch.long, device=self.device)
        clip        = self.ppo_clip

        agg = {"loss": 0.0, "kl": 0.0, "ratio": 0.0}
        n   = 0

        self.policy.train()

        for _ in range(n_epochs):
            for ro in rollouts:
                x0     = ro["x0"]
                lp_old = ro["lp_old"]     # float (snapshot at rollout time)
                r      = ro["reward"]["total"]

                # EMA baseline
                self.baseline = (
                    self.baseline_decay * self.baseline
                    + (1.0 - self.baseline_decay) * r
                )
                advantage = float(r - self.baseline)

                # Log prob under current policy (differentiable)
                lp_new = _score_log_prob(
                    self.policy, x0, self.diffusion, class_label, self.device,
                    K=self.lp_K, requires_grad=True,
                )

                # Clipped PPO importance ratio
                log_ratio = lp_new - lp_old
                ratio     = log_ratio.exp()
                ppo_1     = ratio * advantage
                ppo_2     = ratio.clamp(1.0 - clip, 1.0 + clip) * advantage
                loss_ppo  = -torch.min(ppo_1, ppo_2)

                # KL divergence: ε_θ vs ε_θ₀ on this x₀
                kl = _score_kl(
                    self.policy, self.frozen, x0, self.diffusion,
                    class_label, self.device, K=self.lp_K,
                )

                loss = loss_ppo + self.kl_beta * kl

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
                self.opt.step()
                self.ema.update(self.policy)

                agg["loss"]  += loss.item()
                agg["kl"]    += kl.item()
                agg["ratio"] += ratio.item()
                n += 1

        if n > 0:
            for k in agg:
                agg[k] /= n
        return agg


# ──────────────────────────────────────────────────────────────────────────────
# GRPO trainer
# ──────────────────────────────────────────────────────────────────────────────

class GRPOTrainer:
    """
    Group Relative Policy Optimisation adapted for diffusion fine-tuning.

    Each call to grpo_update():
      1. Generate G ECGs for the same class with DDIM.
      2. Compute rewards r_1, …, r_G.
      3. Group-relative advantage: A_i = (r_i − mean(r)) / (std(r) + 1e-8)
      4. Gradient: −(1/G) Σ Aᵢ · log π_θ(xᵢ) + β · KL
    """

    def __init__(
        self,
        policy_model:  nn.Module,
        frozen_model:  nn.Module,
        diffusion:     GaussianDiffusion,
        reward_fn:     ClinicalReward,
        ema:           EMA,
        class_names:   list[str],
        cfg,
        device:        str,
        opt:           torch.optim.Optimizer,
    ) -> None:
        self.policy      = policy_model
        self.frozen      = frozen_model
        self.diffusion   = diffusion
        self.reward_fn   = reward_fn
        self.ema         = ema
        self.class_names = class_names
        self.cfg         = cfg
        self.device      = device
        self.opt         = opt

        rl = cfg.rl
        self.G         = int(rl.grpo_groups)
        self.kl_beta   = float(rl.kl_beta)
        self.n_steps   = int(rl.ddim_rollout_steps)
        self.grad_clip = float(rl.grad_clip)
        self.lp_K      = 10

    def grpo_update(self, class_idx: int) -> dict[str, float]:
        """Generate G trajectories, compute group-relative advantages, update θ."""
        class_label = torch.tensor([class_idx], dtype=torch.long, device=self.device)
        class_name  = self.class_names[class_idx]

        # ── Generate G ECGs (no gradient) ─────────────────────────────────────
        self.policy.eval()
        x0_list      = []
        reward_dicts = []

        with torch.no_grad():
            for _ in range(self.G):
                x0 = _ddim_sample(
                    self.policy, self.diffusion, class_label,
                    self.n_steps, self.cfg, self.device,
                )
                x0_list.append(x0.detach())

        # ── Rewards ────────────────────────────────────────────────────────────
        for g in range(self.G):
            ecg_np = x0_list[g].squeeze(0).T.cpu().numpy()
            try:
                rd = self.reward_fn.compute(ecg_np, class_name, class_idx)
            except Exception:
                rd = {"total": 0.0, "r_morph": 0.0, "r_hrv": 0.0,
                      "r_real": 0.0, "r_diag": 0.5}
            reward_dicts.append(rd)

        rewards    = torch.tensor([rd["total"] for rd in reward_dicts], device=self.device)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # ── Policy gradient + KL ───────────────────────────────────────────────
        self.policy.train()
        total_loss = torch.zeros(1, device=self.device)
        total_kl   = torch.zeros(1, device=self.device)

        for g in range(self.G):
            lp_new = _score_log_prob(
                self.policy, x0_list[g], self.diffusion, class_label, self.device,
                K=self.lp_K, requires_grad=True,
            )
            kl = _score_kl(
                self.policy, self.frozen, x0_list[g], self.diffusion,
                class_label, self.device, K=self.lp_K,
            )
            total_loss = total_loss + (-advantages[g] * lp_new + self.kl_beta * kl)
            total_kl   = total_kl + kl.detach()

        total_loss = total_loss / self.G

        self.opt.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.opt.step()
        self.ema.update(self.policy)

        mean_rd = {
            "total":   float(rewards.mean().item()),
            "r_morph": float(np.mean([rd["r_morph"] for rd in reward_dicts])),
            "r_hrv":   float(np.mean([rd["r_hrv"]   for rd in reward_dicts])),
            "r_real":  float(np.mean([rd["r_real"]  for rd in reward_dicts])),
            "r_diag":  float(np.mean([rd["r_diag"]  for rd in reward_dicts])),
        }
        return {
            "loss":   float(total_loss.item()),
            "kl":     float(total_kl.item() / self.G),
            "reward": mean_rd,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _make_progress_plot(
    policy_model: nn.Module,
    frozen_model: nn.Module,
    diffusion:    GaussianDiffusion,
    class_names:  list[str],
    cfg,
    iteration:    int,
    results_dir:  Path,
    device:       str,
    log,
) -> None:
    """
    Side-by-side Lead-II comparison: frozen baseline (blue) vs RL policy (red).
    Up to 3 classes × 5 samples, 2 rows per class.
    """
    n_show  = min(3, len(class_names))
    n_cols  = 5
    n_rows  = n_show * 2
    sig_len = int(cfg.ptbxl.signal_length)
    fs      = float(cfg.ptbxl.sampling_rate)
    t_axis  = np.arange(sig_len) / fs
    n_steps = int(cfg.diffusion.ddim_steps)

    with plt.rc_context(PUBSTYLE):
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3 * n_cols, 2.5 * n_rows),
            constrained_layout=True,
        )
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        for ci, cls_name in enumerate(class_names[:n_show]):
            cls_label = torch.full((n_cols,), ci, dtype=torch.long, device=device)
            ts        = torch.linspace(
                diffusion.T - 1, 0, n_steps, dtype=torch.long, device=device
            )

            for model_obj, row_off, color, label_str in [
                (frozen_model, ci * 2,     "#1f77b4", "Baseline"),
                (policy_model, ci * 2 + 1, "#d62728", f"RL iter {iteration:04d}"),
            ]:
                model_obj.eval()
                x = torch.randn(n_cols, 12, sig_len, device=device)
                for i, t_curr in enumerate(ts):
                    t_b    = t_curr.expand(n_cols)
                    eps    = model_obj(x, t_b, cls_label)
                    ab_t   = diffusion.alpha_bar[t_curr]
                    x0_hat = (x - (1.0-ab_t).sqrt()*eps) / ab_t.sqrt().clamp(min=1e-8)
                    x0_hat = x0_hat.clamp(-4.0, 4.0)
                    ab_p   = (
                        diffusion.alpha_bar[ts[i+1]] if i+1 < n_steps
                        else torch.ones(1, device=device)
                    )
                    x = ab_p.sqrt() * x0_hat + (1.0 - ab_p).sqrt() * eps

                samples = x.cpu().numpy()  # (n_cols, 12, sig_len)

                for col in range(n_cols):
                    ax = axes[row_off, col]
                    ax.plot(t_axis, samples[col, 1, :], color=color, lw=0.7, alpha=0.9)
                    ax.set_ylim(-4.5, 4.5)
                    ax.tick_params(labelsize=7)
                    if col == 0:
                        ax.set_ylabel(f"{cls_name}\n{label_str}", fontsize=8, color=color)
                    if row_off == n_rows - 1:
                        ax.set_xlabel("Time (s)", fontsize=7)

        fig.suptitle(
            f"RL Training Progress — Iteration {iteration:04d}  (Lead II)",
            fontsize=10,
        )

    out = results_dir / f"rl_progress_iter{iteration:04d}.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Progress plot → {out.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg, log) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # ── Directories ──────────────────────────────────────────────────────────
    processed_dir = Path(cfg.paths.outputs.processed)
    models_dir    = Path(cfg.paths.outputs.models)
    results_dir   = Path(cfg.paths.outputs.results)
    logs_dir      = Path(cfg.paths.logs)
    for d in (models_dir, results_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Class names ──────────────────────────────────────────────────────────
    class_names_path = processed_dir / "class_names.json"
    if not class_names_path.exists():
        raise FileNotFoundError(f"{class_names_path} not found. Run step03 first.")
    with open(class_names_path) as f:
        class_names = json.load(f)
    n_classes = len(class_names)
    log.info(f"Classes ({n_classes}): {class_names}")

    # ── Load pre-trained checkpoint ───────────────────────────────────────────
    best_path = models_dir / "diffusion_best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"{best_path} not found. Run step04 first.")
    log.info(f"Loading {best_path.name} …")
    ckpt = torch.load(str(best_path), map_location=device)

    # Policy model — updated during RL
    policy_model = ECGTransformerDiffusion(cfg, n_classes=n_classes).to(device)
    policy_model.load_state_dict(ckpt["model"])

    # Frozen reference — NEVER updated
    frozen_model = ECGTransformerDiffusion(cfg, n_classes=n_classes).to(device)
    frozen_model.load_state_dict(ckpt["model"])
    for p in frozen_model.parameters():
        p.requires_grad_(False)
    frozen_model.eval()

    frozen_checksum = float(
        sum(p.data.sum().item() for p in frozen_model.parameters())
    )
    log.info(f"Frozen model checksum: {frozen_checksum:.6f}")

    # ── Diffusion schedule ────────────────────────────────────────────────────
    diffusion = GaussianDiffusion(
        T=int(cfg.diffusion.T),
        beta_schedule=str(cfg.diffusion.beta_schedule),
        device=device,
    )

    # ── EMA (restore from checkpoint) ────────────────────────────────────────
    ema = EMA(policy_model, decay=float(cfg.diffusion.ema_decay))
    if "ema_shadow" in ckpt:
        ema.shadow = {k: v.to(device) for k, v in ckpt["ema_shadow"].items()}

    # ── Reward function ───────────────────────────────────────────────────────
    log.info("Building ClinicalReward …")
    X_train_path = processed_dir / "X_train.npy"
    if not X_train_path.exists():
        raise FileNotFoundError(f"{X_train_path} not found. Run step02 first.")
    X_train   = np.load(str(X_train_path))
    reward_fn = get_reward(
        "full", cfg=cfg, X_train=X_train,
        class_names=class_names, device=device,
    )
    log.info("Reward function ready.")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    rl  = cfg.rl
    opt = torch.optim.AdamW(
        policy_model.parameters(),
        lr=float(rl.rl_lr),
        weight_decay=float(cfg.diffusion.weight_decay),
    )
    current_lr = float(rl.rl_lr)

    # ── Trainer ───────────────────────────────────────────────────────────────
    algorithm = str(rl.algorithm).lower()
    log.info(f"RL algorithm: {algorithm}")

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
        raise ValueError(f"Unknown rl.algorithm: {algorithm!r}")

    n_rollouts   = int(rl.n_rollouts)
    n_ppo_epochs = int(rl.n_ppo_epochs)
    n_iterations = int(rl.rl_iterations)

    # ── CSV log ───────────────────────────────────────────────────────────────
    log_path = logs_dir / "rl_training_log.csv"
    log_fh   = open(log_path, "w", newline="")
    csv_w    = csv.writer(log_fh)
    csv_w.writerow([
        "iter", "class",
        "reward_total", "r_morph", "r_hrv", "r_real", "r_diag",
        "kl", "loss", "lr",
    ])
    log_fh.flush()

    # ── Safety alarm state ────────────────────────────────────────────────────
    morph_low_count = 0
    MORPH_THRESH    = 0.3
    KL_THRESH       = 1.0

    # ── Training ──────────────────────────────────────────────────────────────
    best_reward = -float("inf")
    rng = np.random.default_rng(int(cfg.seeds[0]))

    log.info(f"Starting {n_iterations} RL iterations …")

    for it in range(1, n_iterations + 1):
        class_idx  = int(rng.integers(0, n_classes))
        class_name = class_names[class_idx]

        # ── RL step ───────────────────────────────────────────────────────────
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
            kl   = update["kl"]
            loss = update["loss"]
        else:  # grpo
            update  = trainer.grpo_update(class_idx)
            mean_rd = update["reward"]
            kl      = update["kl"]
            loss    = update["loss"]

        # ── CSV log ───────────────────────────────────────────────────────────
        csv_w.writerow([
            it, class_name,
            f"{mean_rd['total']:.5f}",
            f"{mean_rd['r_morph']:.5f}",
            f"{mean_rd['r_hrv']:.5f}",
            f"{mean_rd['r_real']:.5f}",
            f"{mean_rd['r_diag']:.5f}",
            f"{kl:.5f}",
            f"{loss:.5f}",
            f"{current_lr:.2e}",
        ])
        log_fh.flush()

        if it == 1 or it % 5 == 0:
            log.info(
                f"[{it:04d}/{n_iterations}] class={class_name} | "
                f"r={mean_rd['total']:.4f} "
                f"(morph={mean_rd['r_morph']:.3f} "
                f"hrv={mean_rd['r_hrv']:.3f} "
                f"real={mean_rd['r_real']:.3f} "
                f"diag={mean_rd['r_diag']:.3f}) | "
                f"kl={kl:.4f} | loss={loss:.5f} | lr={current_lr:.1e}"
            )

        # ── Safety alarm: KL divergence ───────────────────────────────────────
        if kl > KL_THRESH:
            current_lr *= 0.5
            for pg in opt.param_groups:
                pg["lr"] = current_lr
            log.warning(
                f"  ⚠ KL={kl:.4f} > {KL_THRESH} — possible reward hacking; "
                f"halving lr → {current_lr:.2e}"
            )

        # ── Safety alarm: morphology collapse ────────────────────────────────
        if mean_rd["r_morph"] < MORPH_THRESH:
            morph_low_count += 1
            if morph_low_count >= 3:
                log.warning(
                    f"  ⚠ ALERT: r_morph < {MORPH_THRESH} for 3 consecutive "
                    f"iterations — reward hacking detected. Saving alert checkpoint."
                )
                torch.save(
                    {
                        "iter":       it,
                        "model":      policy_model.state_dict(),
                        "ema_shadow": ema.shadow,
                        "r_morph":    mean_rd["r_morph"],
                    },
                    str(models_dir / "reward_hacking_alert.pt"),
                )
                morph_low_count = 0   # reset to avoid flooding disk
        else:
            morph_low_count = 0

        # ── Best model ────────────────────────────────────────────────────────
        if mean_rd["total"] > best_reward:
            best_reward = mean_rd["total"]
            torch.save(
                {
                    "iter":        it,
                    "model":       policy_model.state_dict(),
                    "ema_shadow":  ema.shadow,
                    "best_reward": best_reward,
                    "class_names": class_names,
                    "n_classes":   n_classes,
                    "algorithm":   algorithm,
                },
                str(models_dir / "diffusion_rl_best.pt"),
            )

        # ── Periodic progress plot ────────────────────────────────────────────
        if it % int(rl.plot_every_iters) == 0:
            _make_progress_plot(
                policy_model, frozen_model, diffusion,
                class_names, cfg, it, results_dir, device, log,
            )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if it % int(rl.save_every_iters) == 0:
            torch.save(
                {
                    "iter":        it,
                    "model":       policy_model.state_dict(),
                    "ema_shadow":  ema.shadow,
                    "opt":         opt.state_dict(),
                    "best_reward": best_reward,
                    "class_names": class_names,
                    "n_classes":   n_classes,
                    "current_lr":  current_lr,
                },
                str(models_dir / f"rl_ckpt_iter{it:04d}.pt"),
            )
            log.info(f"Checkpoint → rl_ckpt_iter{it:04d}.pt")

    log_fh.close()

    # ── Final assertion: frozen model unchanged ───────────────────────────────
    frozen_checksum_final = float(
        sum(p.data.sum().item() for p in frozen_model.parameters())
    )
    delta = abs(frozen_checksum - frozen_checksum_final)
    assert delta < 1e-3, (
        f"Frozen model weights changed during RL training! "
        f"Δchecksum = {delta:.6f}  "
        f"(initial={frozen_checksum:.6f}, final={frozen_checksum_final:.6f})"
    )
    log.info(
        f"✓ Frozen model integrity verified — weights unchanged "
        f"(checksum Δ = {delta:.2e})"
    )

    return best_reward


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    log = get_logger("step07_rl_finetuning", cfg=cfg)
    set_seed(int(cfg.seeds[0]))

    best_reward = train(cfg, log)
    print(f"✓ RL fine-tuning complete. Best total reward: {best_reward:.3f}")


if __name__ == "__main__":
    main()
