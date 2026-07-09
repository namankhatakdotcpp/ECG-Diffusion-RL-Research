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
  outputs/models/diffusion_rl_best.pt         — best training-reward checkpoint
  outputs/models/diffusion_rl_selected.pt     — best checkpoint by independent
                                                 Mentor Classifier eval (heavy
                                                 checkpoints only; see
                                                 rl.checkpoint_selection)
  outputs/models/rl_ckpt_iter{N:04d}.pt       — periodic checkpoints
  logs/rl_training_log.csv                    — per-iteration metrics
  outputs/results/rl_progress_iter{N:04d}.png — visual progress snapshots
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

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
        self.baseline_step  = 0     # Adam-style bias-correction counter, see
                                     # _bias_corrected_baseline. Starts at 0
                                     # (incremented to 1 before first use) and
                                     # is NOT persisted across checkpoints --
                                     # this trainer object is reconstructed
                                     # fresh on every run (self.baseline
                                     # itself is not saved/restored in any
                                     # checkpoint dict either), so a resumed
                                     # run correctly restarts the correction
                                     # from t=0 along with a fresh baseline,
                                     # not a stale t paired with an old one.
        self.lp_K           = 10    # timesteps sampled for log-prob proxy

    def _update_and_bias_correct_baseline(self, r: float) -> float:
        """
        Update the EMA baseline with reward r, then return the Adam-style
        bias-corrected estimate for use in this step's advantage.

        Quantified problem this fixes (Roadmap/Stage_4_Optimization/
        Decisions.md): with baseline initialized to 0.0 and no correction,
        the weight still resting on that 0.0 init after k updates is
        `decay^k` -- 85% after 16 updates (1 smoke-test iteration), 20%
        still at the end of a 10-iteration smoke test, 37% at k=100 and
        ~8% at k=250 (squarely inside the planned Gate 3 validation range)
        -- not a cold-start artifact a longer run washes out on its own.

        baseline_hat = baseline / (1 - decay^t), t starting at 1 (matches
        Adam's bias correction exactly). At t=1 this fully cancels the
        (1-decay) weight the first sample would otherwise be down-weighted
        by; as t -> large, decay^t -> 0 and baseline_hat -> baseline, so
        long-run behaviour (decay=0.99, unchanged per this fix's scope) is
        identical to before -- only the early/mid-run bias is removed.
        """
        self.baseline = (
            self.baseline_decay * self.baseline
            + (1.0 - self.baseline_decay) * r
        )
        self.baseline_step += 1
        correction = 1.0 - self.baseline_decay ** self.baseline_step
        return self.baseline / correction

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
                    "r_hrv": 0.0, "r_real": 0.0, "r_diag": 0.5, "r_a3": 0.0,
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
        """
        Clipped PPO loss + KL regularisation; returns mean metrics.

        No critic/value network in this DDPO setup (advantage comes from an
        EMA scalar baseline, not a learned value function) — 'value_loss' is
        therefore not applicable and is omitted rather than faked. Likewise
        there's no closed-form policy entropy for the score-matching log-prob
        proxy; `logprob_std` (spread of lp_new across the batch) is logged as
        a stand-in stochasticity signal instead of a real entropy number.
        """
        class_idx   = rollouts[0]["class_idx"]
        class_label = torch.tensor([class_idx], dtype=torch.long, device=self.device)
        clip        = self.ppo_clip

        rewards = np.array([ro["reward"]["total"] for ro in rollouts], dtype=float)

        agg = {
            "loss": 0.0, "policy_loss": 0.0, "kl": 0.0, "ratio": 0.0,
            "clip_fraction": 0.0, "grad_norm": 0.0,
        }
        advantages: list[float] = []
        logprobs:   list[float] = []
        n = 0

        self.policy.train()

        for _ in range(n_epochs):
            for ro in rollouts:
                x0     = ro["x0"]
                lp_old = ro["lp_old"]     # float (snapshot at rollout time)
                r      = ro["reward"]["total"]

                # Bias-corrected EMA baseline (see _update_and_bias_correct_baseline)
                baseline_hat = self._update_and_bias_correct_baseline(r)
                advantage = float(r - baseline_hat)

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
                # clip_grad_norm_ returns the pre-clip total gradient norm —
                # capture it rather than discarding it, so a vanishing/
                # exploding gradient signal doesn't have to be inferred
                # indirectly from the loss/KL curves alone.
                grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
                self.opt.step()
                self.ema.update(self.policy)

                ratio_val = ratio.item()
                agg["loss"]          += loss.item()
                agg["policy_loss"]   += loss_ppo.item()
                agg["kl"]            += kl.item()
                agg["ratio"]         += ratio_val
                agg["clip_fraction"] += float(abs(ratio_val - 1.0) > clip)
                agg["grad_norm"]     += float(grad_norm)
                advantages.append(advantage)
                logprobs.append(lp_new.item())
                n += 1

        if n > 0:
            for k in agg:
                agg[k] /= n
        agg["advantage_mean"] = float(np.mean(advantages)) if advantages else 0.0
        agg["advantage_std"]  = float(np.std(advantages))  if advantages else 0.0
        agg["logprob_std"]    = float(np.std(logprobs))    if logprobs else 0.0
        agg["reward_mean"]    = float(rewards.mean())
        agg["reward_std"]     = float(rewards.std())
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
                      "r_real": 0.0, "r_diag": 0.5, "r_a3": 0.0}
            reward_dicts.append(rd)

        rewards    = torch.tensor([rd["total"] for rd in reward_dicts], device=self.device)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # ── Policy gradient + KL ───────────────────────────────────────────────
        # No importance-sampling ratio here (GRPO recomputes lp_new fresh each
        # call rather than reusing an "old" snapshot), so ratio/clip_fraction
        # are PPO-only diagnostics — reported as N/A for this algorithm rather
        # than a fabricated value of 1.0/0.0.
        self.policy.train()
        total_loss = torch.zeros(1, device=self.device)
        total_kl   = torch.zeros(1, device=self.device)
        policy_loss_sum = 0.0
        logprobs: list[float] = []

        for g in range(self.G):
            lp_new = _score_log_prob(
                self.policy, x0_list[g], self.diffusion, class_label, self.device,
                K=self.lp_K, requires_grad=True,
            )
            kl = _score_kl(
                self.policy, self.frozen, x0_list[g], self.diffusion,
                class_label, self.device, K=self.lp_K,
            )
            pg_term = -advantages[g] * lp_new
            total_loss = total_loss + (pg_term + self.kl_beta * kl)
            total_kl   = total_kl + kl.detach()
            policy_loss_sum += pg_term.item()
            logprobs.append(lp_new.item())

        total_loss = total_loss / self.G

        self.opt.zero_grad()
        total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.opt.step()
        self.ema.update(self.policy)

        mean_rd = {
            "total":   float(rewards.mean().item()),
            "r_morph": float(np.mean([rd["r_morph"] for rd in reward_dicts])),
            "r_hrv":   float(np.mean([rd["r_hrv"]   for rd in reward_dicts])),
            "r_real":  float(np.mean([rd["r_real"]  for rd in reward_dicts])),
            "r_diag":  float(np.mean([rd["r_diag"]  for rd in reward_dicts])),
            "r_a3":    float(np.mean([rd["r_a3"]    for rd in reward_dicts])),
        }
        adv_np = advantages.detach().cpu().numpy()
        return {
            "loss":            float(total_loss.item()),
            "policy_loss":     policy_loss_sum / self.G,
            "kl":              float(total_kl.item() / self.G),
            "ratio":           None,
            "clip_fraction":   None,
            "grad_norm":       float(grad_norm),
            "advantage_mean":  float(adv_np.mean()),
            "advantage_std":   float(adv_np.std()),
            "logprob_std":     float(np.std(logprobs)) if logprobs else 0.0,
            "reward_mean":     float(rewards.mean().item()),
            "reward_std":      float(rewards.std().item()),
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
# Checkpoint evaluation (cheap per-checkpoint monitoring vs. full Phase-4 suite)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _lightweight_checkpoint_eval(
    policy_model: nn.Module,
    diffusion:    GaussianDiffusion,
    reward_fn:    ClinicalReward,
    class_names:  list[str],
    cfg,
    iteration:    int,
    kl_to_base:   float,
    results_dir:  Path,
    device:       str,
    log,
    n_per_class:  int = 5,
) -> dict:
    """
    Cheap, every-checkpoint monitoring: frozen TRTR classifier confidence +
    A3Reward score (both from `reward_fn`, already loaded — no retrain, no
    subprocess) + KL-to-base (already computed this iteration by the
    trainer's KL regulariser). Intended to catch plateauing/regression
    between the coarser full-suite checkpoints without paying the Mentor
    Classifier's retrain cost.

    `a3_reward_mean` here is THIS reward function's own A3Reward score
    (Mahalanobis distance to the real per-class A3 reference, mapped through
    exp(-d/scale)) — a cheap trend signal, not the same thing as the
    Mahalanobis/Bhattacharyya numbers `mentor_eval/subband_similarity_
    metrics.py` reports (which need real generated-data comparison via the
    full-eval path). Do not conflate the two when reading this file.
    """
    n_steps    = int(cfg.diffusion.ddim_steps)
    diag_confs: list[float] = []
    a3_scores:  list[float] = []

    for ci, cname in enumerate(class_names):
        class_label = torch.full((n_per_class,), ci, dtype=torch.long, device=device)
        x0 = _ddim_sample(policy_model, diffusion, class_label, n_steps, cfg, device)
        for s in range(n_per_class):
            ecg_np = x0[s].T.cpu().numpy()
            try:
                rd = reward_fn.compute(ecg_np, cname, ci)
                diag_confs.append(rd["r_diag"])
                a3_scores.append(rd["r_a3"])
            except Exception:
                continue

    report = {
        "iteration":        iteration,
        "trtr_diag_conf_mean": float(np.mean(diag_confs)) if diag_confs else None,
        "trtr_diag_conf_std":  float(np.std(diag_confs))  if diag_confs else None,
        "a3_reward_mean":   float(np.mean(a3_scores)) if a3_scores else None,
        "a3_reward_std":    float(np.std(a3_scores))  if a3_scores else None,
        "kl_to_base":       kl_to_base,
    }
    out_path = results_dir / f"checkpoint_eval_lightweight_iter{iteration:04d}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(
        f"  [lightweight eval @ {iteration}] "
        f"trtr_diag_conf={report['trtr_diag_conf_mean']:.3f} "
        f"a3_reward={report['a3_reward_mean']:.3f} "
        f"kl_to_base={kl_to_base:.4f} → {out_path.name}"
    )
    # Trend-only: appended to checkpoint_metrics.json for visibility, but never
    # consulted by early stopping or checkpoint selection (see Decisions.md —
    # those require the independent Mentor Classifier, not this TRTR signal).
    _append_checkpoint_metrics(results_dir, {"kind": "lightweight", **report})
    return report


def _run_classification_validation_once(
    ckpt_path: Path, out_dir: Path, seed: int, log,
) -> Optional[dict]:
    """One classification_validation.py run → parsed classifier_generated_eval.json, or None."""
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "mentor_eval.classification_validation",
                "--ckpt", str(ckpt_path), "--out-dir", str(out_dir), "--seed", str(seed),
            ],
            capture_output=True, text=True, timeout=3600,
        )
        if result.returncode != 0:
            log.warning(
                f"    classification_validation.py (seed={seed}) exited "
                f"{result.returncode}: {result.stderr[-2000:]}"
            )
            return None

        gen_eval_path = out_dir / "classifier_generated_eval.json"
        if not gen_eval_path.exists():
            log.warning(
                f"    {gen_eval_path} not written (seed={seed}) — Stage 2 was "
                "likely [BLOCKED] (see stdout)."
            )
            return None

        gen_metrics = json.load(open(gen_eval_path))
        return {
            "mentor_accuracy": gen_metrics.get("accuracy"),
            "mentor_macro_f1": gen_metrics.get("macro_f1"),
            "mentor_excluded_classes": gen_metrics.get("excluded_classes", []),
            "out_dir": str(out_dir),
        }
    except Exception as exc:
        log.warning(f"    classification_validation.py (seed={seed}) failed to launch: {exc}")
        return None


def _full_checkpoint_eval(
    ckpt_path: Path, iteration: int, log, n_repeats: int = 1,
) -> Optional[dict]:
    """
    Full Stage-3-equivalent suite: Mentor Classifier retrain + disease-wise
    metrics, via `mentor_eval.classification_validation` (accepts --ckpt/
    --out-dir/--seed, so it can be pointed at this specific RL checkpoint
    without touching the frozen `diffusion_best.pt` baseline). Reserved for
    the coarse `rl.full_eval_checkpoints` subset — this is a several-GPU-
    minute retrain, not something to repeat at every lightweight checkpoint.

    n_repeats > 1 (only used at the FIRST full-eval checkpoint, driven by
    `rl.mentor_stability_check_repeats`) re-runs the whole thing with
    different seeds — same pattern as the earlier n_seeds=3 baseline run —
    to measure mentor_macro_f1's own run-to-run spread (Stage 1 retrains the
    classifier AND Stage 2 regenerates synthetic samples each call, so a
    single run's number could be noise rather than signal; this project has
    direct prior evidence of that at n=200 sample scale, see Decisions.md).
    Reports mentor_macro_f1_std alongside the mean; does NOT compare it
    against `early_stopping.min_delta` itself — that comparison is logged
    explicitly at the call site so it isn't silently buried in this return
    value.

    Returns the Mentor Classifier's metrics on THIS checkpoint's generated
    ECGs, or None if every run failed/was blocked. This is the metric early
    stopping and checkpoint selection should use — never the TRTR classifier
    that IS the training reward signal (see Decisions.md).

    Subband breakdown (`mentor_eval.run_subband_analysis`) and the
    conditioning diagnostic (`mentor_eval.run_disease_conditioning_analysis`)
    are NOT wired in here: both hardcode `outputs/models/diffusion_best.pt`
    with no --ckpt override, so pointing them at an RL checkpoint would
    require adding that override first (or risk clobbering the baseline
    checkpoint other tooling depends on). Left as a follow-up rather than
    faked — see Decisions.md.
    """
    base_out_dir = Path("outputs/mentor_review") / f"rl_checkpoint_iter{iteration:04d}"
    seeds = [42 + i for i in range(max(1, n_repeats))]
    log.info(
        f"  [full eval @ {iteration}] launching classification_validation.py "
        f"--ckpt {ckpt_path} x{len(seeds)} run(s) (this retrains the Mentor "
        f"Classifier — several GPU-minutes EACH) …"
    )

    runs: list[dict] = []
    for i, seed in enumerate(seeds):
        out_dir = base_out_dir if len(seeds) == 1 else base_out_dir.with_name(
            f"{base_out_dir.name}_rep{i}"
        )
        r = _run_classification_validation_once(ckpt_path, out_dir, seed, log)
        if r is not None:
            runs.append(r)

    if not runs:
        return None

    macro_f1s   = [r["mentor_macro_f1"] for r in runs if r["mentor_macro_f1"] is not None]
    accuracies  = [r["mentor_accuracy"] for r in runs if r["mentor_accuracy"] is not None]
    metrics = {
        "mentor_macro_f1":     float(np.mean(macro_f1s)) if macro_f1s else None,
        "mentor_macro_f1_std": float(np.std(macro_f1s)) if len(macro_f1s) > 1 else None,
        "mentor_accuracy":     float(np.mean(accuracies)) if accuracies else None,
        "mentor_excluded_classes": runs[0].get("mentor_excluded_classes", []),
        "n_repeats": len(runs),
        "out_dirs": [r["out_dir"] for r in runs],
    }
    log.info(
        f"  [full eval @ {iteration}] mentor_macro_f1="
        f"{metrics['mentor_macro_f1']}"
        + (f" (std={metrics['mentor_macro_f1_std']:.4f} over {len(runs)} runs)"
           if metrics["mentor_macro_f1_std"] is not None else "")
        + f" mentor_accuracy={metrics['mentor_accuracy']}"
    )
    return metrics


def _append_checkpoint_metrics(results_dir: Path, record: dict) -> None:
    """Append one record to the consolidated, machine-readable checkpoint log."""
    path = results_dir / "checkpoint_metrics.json"
    records = json.load(open(path)) if path.exists() else []
    records.append(record)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)


def _is_better(value: float, best: Optional[float], mode: str, min_delta: float = 0.0) -> bool:
    if best is None:
        return True
    return (value > best + min_delta) if mode == "max" else (value < best - min_delta)


@torch.no_grad()
def _same_seed_before_after_comparison(
    frozen_model: nn.Module,
    policy_model: nn.Module,
    diffusion:    GaussianDiffusion,
    class_names:  list[str],
    reward_fn:    ClinicalReward,
    cfg,
    device:       str,
    log,
    class_idx:    int = 0,
) -> dict:
    """
    Same latent seed, frozen vs. post-smoke-test policy: if the policy
    genuinely updated, the two generated ECGs should measurably differ
    (L2, cosine, and A3-reward delta) — a numeric version of the visual
    frozen-vs-policy progress plot, not a replacement for it.
    """
    n_steps = int(cfg.diffusion.ddim_steps)
    class_label = torch.full((1,), class_idx, dtype=torch.long, device=device)
    cname = class_names[class_idx]

    torch.manual_seed(1234)
    x0_frozen = _ddim_sample(frozen_model, diffusion, class_label, n_steps, cfg, device)
    torch.manual_seed(1234)
    x0_policy = _ddim_sample(policy_model, diffusion, class_label, n_steps, cfg, device)

    a = x0_frozen.flatten().cpu().numpy()
    b = x0_policy.flatten().cpu().numpy()
    l2 = float(np.linalg.norm(a - b))
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    ecg_frozen = x0_frozen.squeeze(0).T.cpu().numpy()
    ecg_policy = x0_policy.squeeze(0).T.cpu().numpy()
    try:
        r_frozen = reward_fn.compute(ecg_frozen, cname, class_idx)
        r_policy = reward_fn.compute(ecg_policy, cname, class_idx)
    except Exception:
        r_frozen = r_policy = {"total": None}

    result = {
        "class": cname,
        "l2_distance": l2,
        "cosine_similarity": cosine,
        "reward_frozen": r_frozen.get("total"),
        "reward_policy": r_policy.get("total"),
        "samples_measurably_differ": l2 > 1e-4,
    }
    log.info(
        f"  [same-seed before/after] class={cname} l2={l2:.4f} "
        f"cosine={cosine:.5f} reward_frozen={r_frozen.get('total')} "
        f"reward_policy={r_policy.get('total')}"
    )
    if not result["samples_measurably_differ"]:
        log.warning(
            "  ⚠ Same-seed frozen vs. policy samples are numerically "
            "identical (l2 ~ 0) — the policy did not change what it "
            "generates. Consistent with a dead policy-gradient update."
        )
    return result


def _plot_smoke_test_diagnostics(
    kl_history: list[float], reward_history: list[float],
    grad_norm_history: list[float], results_dir: Path, log,
) -> None:
    """Stacked KL / reward / grad_norm-vs-iteration plot for the smoke test."""
    n = len(kl_history)
    if n == 0:
        return
    x = list(range(1, n + 1))
    with plt.rc_context(PUBSTYLE):
        fig, axes = plt.subplots(3, 1, figsize=(6, 7), sharex=True, constrained_layout=True)
        axes[0].plot(x, kl_history, marker="o", color="#d62728")
        axes[0].set_ylabel("KL")
        axes[1].plot(x, reward_history, marker="o", color="#2ca02c")
        axes[1].set_ylabel("reward_total")
        axes[2].plot(x, grad_norm_history, marker="o", color="#1f77b4")
        axes[2].set_ylabel("grad_norm")
        axes[2].set_xlabel("smoke-test iteration")
        fig.suptitle("Smoke Test Diagnostics", fontsize=11)
    out = results_dir / "smoke_test_diagnostics.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Diagnostics plot → {out.name}")


def _report_smoke_test_results(
    policy_model: nn.Module,
    frozen_model: nn.Module,
    diffusion:    GaussianDiffusion,
    class_names:  list[str],
    cfg,
    device:       str,
    checksum_before: float,
    kl_history:     list[float],
    reward_history: list[float],
    grad_norm_history: list[float],
    reward_fn:      ClinicalReward,
    results_dir:    Path,
    log,
) -> dict:
    """
    Explicit pass/warn checks for the failure mode this smoke test exists to
    catch: a policy-gradient wiring bug that runs without crashing but never
    actually updates the policy. Every check here is a documented, measurable
    condition — not a subjective "looks fine" read of a training curve.
    """
    checksum_after = float(sum(p.data.sum().item() for p in policy_model.parameters()))
    params_changed = abs(checksum_after - checksum_before) > 1e-8

    n = len(kl_history)
    half = max(1, n // 2)
    kl_nonzero      = any(k > 1e-6 for k in kl_history)
    grad_never_zero = all(g > 1e-10 for g in grad_norm_history) if grad_norm_history else False
    reward_first_half  = float(np.mean(reward_history[:half])) if reward_history else None
    reward_second_half = float(np.mean(reward_history[half:])) if reward_history[half:] else None
    reward_trend_up = (
        reward_second_half is not None and reward_first_half is not None
        and reward_second_half >= reward_first_half
    )

    same_seed = _same_seed_before_after_comparison(
        frozen_model, policy_model, diffusion, class_names, reward_fn, cfg, device, log,
    )

    checks = {
        "policy_params_changed": params_changed,
        "kl_nonzero_at_some_point": kl_nonzero,
        "grad_norm_never_zero": grad_never_zero,
        "reward_did_not_degrade_first_vs_second_half": reward_trend_up,
        "same_seed_samples_measurably_differ": same_seed["samples_measurably_differ"],
    }
    all_pass = all(checks.values())

    timing = reward_fn.get_timing_summary()
    _plot_smoke_test_diagnostics(kl_history, reward_history, grad_norm_history, results_dir, log)

    report = {
        "n_iterations":        n,
        "checksum_before":     checksum_before,
        "checksum_after":      checksum_after,
        "kl_history":          kl_history,
        "reward_history":      reward_history,
        "grad_norm_history":   grad_norm_history,
        "reward_first_half":   reward_first_half,
        "reward_second_half":  reward_second_half,
        "reward_component_latency_ms": timing,
        "same_seed_before_after": same_seed,
        "checks":              checks,
        "all_checks_passed":   all_pass,
    }
    out_path = results_dir / "smoke_test_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # baseline_reward.json — nice-to-have for later delta reporting against a
    # real training run, not blocking anything here.
    if reward_history:
        baseline_path = results_dir / "baseline_reward.json"
        with open(baseline_path, "w") as f:
            json.dump({
                "reward_total_iter1": reward_history[0],
                "reward_component_latency_ms": timing,
            }, f, indent=2)

    log.info("=" * 60)
    log.info("SMOKE TEST RESULTS")
    log.info("=" * 60)
    for name, passed in checks.items():
        log.info(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    log.info(
        f"  reward: first-half mean={reward_first_half} → "
        f"second-half mean={reward_second_half}"
    )
    log.info("  reward-component latency (ms/ECG, mean):")
    for comp, stats in timing.items():
        log.info(f"    {comp}: {stats['mean_ms']}")
    log.info(
        "  reward_a3 IS included in the timing above (a3 row) — implemented "
        "and wired into the composite reward as of this smoke test. This "
        "does NOT numerically validate reward_a3 against Stage 3's subband "
        "metrics; run validate_a3_reward.py separately for that (see "
        "Decisions.md — Stage3_Subband_Master_Comparison.md has no real "
        "numbers to compare against yet, 0/72 rows evaluated)."
    )
    a3_stats = timing.get("a3", {})
    if a3_stats.get("mean_ms") and timing.get("total", {}).get("mean_ms"):
        a3_pct = 100.0 * a3_stats["mean_ms"] / timing["total"]["mean_ms"]
        log.info(f"  a3 is {a3_pct:.1f}% of total reward-computation latency.")
        if a3_pct > 50.0:
            log.warning(
                f"  ⚠ a3 accounts for {a3_pct:.1f}% of reward latency — it "
                "may be the PPO rollout bottleneck. Do not optimize "
                "prematurely, but document this before a long GPU run."
            )
    if all_pass:
        log.info("✓ All smoke-test checks passed. Policy is updating under RL.")
    else:
        log.warning(
            "⚠ One or more smoke-test checks FAILED. Do not proceed to a real "
            "GPU run until this is diagnosed — a documented failure mode in "
            "diffusion-RL is a policy-gradient loop that runs without "
            "crashing but never actually updates the policy."
        )
    log.info(f"Report → {out_path}")
    return report


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(
    cfg, log, smoke_test: bool = False,
    n_iterations_override: Optional[int] = None,
    run_tag: Optional[str] = None,
) -> float:
    """
    smoke_test=True runs a short (`rl.smoke_test_iterations`, default 10)
    correctness check instead of the full schedule: does the policy actually
    change under PPO/GRPO before any GPU-hours are committed to a real run?
    Skips periodic checkpointing and the eval-checkpoint schedule (nothing
    to select/stop on in 10 iterations); collects parameter-checksum,
    KL/reward trend, grad_norm, and reward-latency diagnostics and writes
    them to outputs/results/smoke_test_report.json.

    n_iterations_override: run exactly this many iterations instead of
    `rl.rl_iterations` (config.yaml) or `rl.smoke_test_iterations` — for a
    validation-length run (e.g. Gate 3's 100-250 updates) that's neither
    the smoke test nor a full training run's default length. Ignored if
    smoke_test=True (that always uses rl.smoke_test_iterations).

    run_tag: if set, checkpoints/logs/results this run WRITES (not the
    diffusion_best.pt it reads) go under a subdirectory named `run_tag` —
    see the "Output isolation" block below for exactly what that covers.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    if smoke_test:
        log.info("=" * 60)
        log.info("SMOKE TEST MODE — short run to verify PPO/GRPO wiring, "
                  "not a real training run.")
        log.info("=" * 60)

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

    # ── Output isolation (--run-tag) ────────────────────────────────────────
    # Re-point WRITE paths only, after reading the frozen baseline above --
    # diffusion_best.pt always comes from the untagged models dir. Every
    # checkpoint/log/result this run writes (rl_ckpt_iter*.pt,
    # diffusion_rl_best.pt, diffusion_rl_selected.pt, reward_hacking_alert.pt,
    # rl_training_log.csv, progress plots, smoke_test_report.json,
    # checkpoint_metrics.json) lands under the tag instead, so a longer
    # validation run's checkpoints can't collide with / be silently
    # overwritten by a later run that reuses the same iteration numbers.
    if run_tag:
        models_dir  = models_dir / run_tag
        results_dir = results_dir / run_tag
        logs_dir    = logs_dir / run_tag
        for d in (models_dir, results_dir, logs_dir):
            d.mkdir(parents=True, exist_ok=True)
        log.info(
            f"--run-tag={run_tag!r}: writing checkpoints/logs/results under "
            f"this tag ({models_dir}, {results_dir}, {logs_dir}). Reads "
            f"diffusion_best.pt from the untagged models dir (already loaded above)."
        )

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
    if smoke_test:
        n_iterations = int(rl.get("smoke_test_iterations", 10))
    elif n_iterations_override is not None:
        n_iterations = int(n_iterations_override)
    else:
        n_iterations = int(rl.rl_iterations)

    # ── CSV log ───────────────────────────────────────────────────────────────
    log_path = logs_dir / "rl_training_log.csv"
    log_fh   = open(log_path, "w", newline="")
    csv_w    = csv.writer(log_fh)
    csv_w.writerow([
        "iter", "class",
        "reward_total", "r_morph", "r_hrv", "r_real", "r_diag", "r_a3",
        "contrib_morph", "contrib_hrv", "contrib_real", "contrib_diag", "contrib_a3",
        "kl", "loss", "lr",
        # Standard PPO/GRPO diagnostics. ratio/clip_fraction are PPO-only
        # (importance-sampling artifacts) — blank for GRPO, not 0/1.
        "policy_loss", "ratio", "clip_fraction", "grad_norm",
        "advantage_mean", "advantage_std", "logprob_std",
        "reward_mean", "reward_std",
    ])
    log_fh.flush()

    # Weighted contribution = weight * raw component score, i.e. how much of
    # `reward_total` each term is actually responsible for at this iteration
    # (not just its configured weight). Logged per the Stage 4 decision to
    # measure real per-term contribution before revisiting reward weights.
    reward_weights = reward_fn.weights

    # Checkpoint eval schedule (see config.yaml rl.eval_checkpoints /
    # rl.full_eval_checkpoints): lightweight eval is cheap and runs at every
    # entry; the full Mentor-Classifier-retrain suite only runs at the coarser
    # full_eval_checkpoints subset to avoid paying that multi-minute retrain
    # cost repeatedly.
    eval_checkpoints      = set() if smoke_test else set(int(x) for x in rl.get("eval_checkpoints", []))
    full_eval_checkpoints = set() if smoke_test else set(int(x) for x in rl.get("full_eval_checkpoints", []))

    # ── Smoke-test diagnostics ───────────────────────────────────────────────
    smoke_kl_history:        list[float] = []
    smoke_reward_history:    list[float] = []
    smoke_grad_norm_history: list[float] = []
    smoke_param_checksum_before = float(
        sum(p.data.sum().item() for p in policy_model.parameters())
    ) if smoke_test else None

    # Checkpoint selection + early stopping: BOTH gated on full-eval (Mentor
    # Classifier) checkpoints only. Never on TRTR-based lightweight
    # checkpoints — TRTR is the training reward signal itself, so stopping or
    # selecting on it would just reconfirm training-reward convergence, not
    # generalisable quality. See config.yaml rl.early_stopping /
    # rl.checkpoint_selection and Decisions.md.
    sel_cfg          = rl.get("checkpoint_selection", {})
    sel_metric_name  = str(sel_cfg.get("metric", "mentor_macro_f1"))
    sel_mode         = str(sel_cfg.get("mode", "max"))
    best_selected_metric: Optional[float] = None

    es_cfg        = rl.get("early_stopping", {})
    es_enabled    = bool(es_cfg.get("enabled", False))
    es_metric     = str(es_cfg.get("metric", "mentor_macro_f1"))
    es_mode       = str(es_cfg.get("mode", "max"))
    es_patience   = int(es_cfg.get("patience", 2))
    es_min_delta  = float(es_cfg.get("min_delta", 0.0))
    es_best: Optional[float] = None
    es_bad_count  = 0
    stop_early    = False
    first_full_eval_done = False
    mentor_stability_repeats = int(rl.get("mentor_stability_check_repeats", 3))

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
                "r_a3":    float(np.mean([ro["reward"]["r_a3"]    for ro in rollouts])),
            }
            kl   = update["kl"]
            loss = update["loss"]
        else:  # grpo
            update  = trainer.grpo_update(class_idx)
            mean_rd = update["reward"]
            kl      = update["kl"]
            loss    = update["loss"]

        diag = {
            "policy_loss":    update.get("policy_loss", 0.0),
            "ratio":          update.get("ratio"),           # None for GRPO
            "clip_fraction":  update.get("clip_fraction"),   # None for GRPO
            "grad_norm":      update.get("grad_norm", 0.0),
            "advantage_mean": update.get("advantage_mean", 0.0),
            "advantage_std":  update.get("advantage_std", 0.0),
            "logprob_std":    update.get("logprob_std", 0.0),
            "reward_mean":    update.get("reward_mean", mean_rd["total"]),
            "reward_std":     update.get("reward_std", 0.0),
        }

        if smoke_test:
            smoke_kl_history.append(kl)
            smoke_reward_history.append(mean_rd["total"])
            smoke_grad_norm_history.append(diag["grad_norm"])

        # ── Reward-contribution logging ─────────────────────────────────────────
        contrib = {
            k: reward_weights.get(k, 0.0) * mean_rd[f"r_{k}"]
            for k in ("morph", "hrv", "real", "diag", "a3")
        }

        # ── CSV log ───────────────────────────────────────────────────────────
        csv_w.writerow([
            it, class_name,
            f"{mean_rd['total']:.5f}",
            f"{mean_rd['r_morph']:.5f}",
            f"{mean_rd['r_hrv']:.5f}",
            f"{mean_rd['r_real']:.5f}",
            f"{mean_rd['r_diag']:.5f}",
            f"{mean_rd['r_a3']:.5f}",
            f"{contrib['morph']:.5f}",
            f"{contrib['hrv']:.5f}",
            f"{contrib['real']:.5f}",
            f"{contrib['diag']:.5f}",
            f"{contrib['a3']:.5f}",
            f"{kl:.5f}",
            f"{loss:.5f}",
            f"{current_lr:.2e}",
            f"{diag['policy_loss']:.5f}",
            "" if diag["ratio"] is None else f"{diag['ratio']:.5f}",
            "" if diag["clip_fraction"] is None else f"{diag['clip_fraction']:.5f}",
            f"{diag['grad_norm']:.5f}",
            f"{diag['advantage_mean']:.5f}",
            f"{diag['advantage_std']:.5f}",
            f"{diag['logprob_std']:.5f}",
            f"{diag['reward_mean']:.5f}",
            f"{diag['reward_std']:.5f}",
        ])
        log_fh.flush()

        if it == 1 or it % 5 == 0:
            total_contrib = sum(contrib.values()) + 1e-8
            log.info(
                f"[{it:04d}/{n_iterations}] class={class_name} | "
                f"r={mean_rd['total']:.4f} "
                f"(morph={mean_rd['r_morph']:.3f} "
                f"hrv={mean_rd['r_hrv']:.3f} "
                f"real={mean_rd['r_real']:.3f} "
                f"diag={mean_rd['r_diag']:.3f} "
                f"a3={mean_rd['r_a3']:.3f}) | "
                f"kl={kl:.4f} | loss={loss:.5f} | lr={current_lr:.1e}"
            )
            log.info(
                "  contribution% "
                + " ".join(
                    f"{k}={100*v/total_contrib:.1f}%" for k, v in contrib.items()
                )
            )
            ratio_str = "n/a" if diag["ratio"] is None else f"{diag['ratio']:.3f}"
            clipf_str = "n/a" if diag["clip_fraction"] is None else f"{diag['clip_fraction']:.3f}"
            log.info(
                f"  diagnostics policy_loss={diag['policy_loss']:.4f} "
                f"ratio={ratio_str} clip_frac={clipf_str} "
                f"grad_norm={diag['grad_norm']:.4f} "
                f"adv={diag['advantage_mean']:.3f}±{diag['advantage_std']:.3f} "
                f"logprob_std={diag['logprob_std']:.3f} "
                f"reward={diag['reward_mean']:.3f}±{diag['reward_std']:.3f}"
            )
            if diag["grad_norm"] == 0.0:
                log.warning(
                    "  ⚠ grad_norm=0.0 — gradients are not flowing to the "
                    "policy. This is the failure mode the smoke test "
                    "(run_ppo_smoke_test) is meant to catch before a real "
                    "GPU run; do not proceed on this run without diagnosing it."
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

        # ── Best model (skipped in smoke-test mode — don't clobber a real best) ─
        if not smoke_test and mean_rd["total"] > best_reward:
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

        # ── Checkpoint (skipped in smoke-test mode — nothing worth keeping) ────
        need_eval_ckpt = it in eval_checkpoints or it in full_eval_checkpoints
        if not smoke_test and (it % int(rl.save_every_iters) == 0 or need_eval_ckpt):
            ckpt_path = models_dir / f"rl_ckpt_iter{it:04d}.pt"
            if not ckpt_path.exists() or it % int(rl.save_every_iters) == 0:
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
                    str(ckpt_path),
                )
                log.info(f"Checkpoint → rl_ckpt_iter{it:04d}.pt")

        # ── Lightweight checkpoint eval ─────────────────────────────────────────
        if it in eval_checkpoints:
            _lightweight_checkpoint_eval(
                policy_model, diffusion, reward_fn, class_names, cfg,
                it, kl, results_dir, device, log,
            )

        # ── Full Stage-3-equivalent checkpoint eval ─────────────────────────────
        if it in full_eval_checkpoints:
            ckpt_path_full = models_dir / f"rl_ckpt_iter{it:04d}.pt"

            # First full-eval checkpoint only: repeat the Mentor Classifier
            # eval `mentor_stability_check_repeats` times (different seeds) to
            # measure mentor_macro_f1's own run-to-run spread before trusting
            # a single reading for early-stop/selection decisions. See
            # Decisions.md — this project already has direct evidence
            # (S3-002's n=200 accuracy discrepancy) that small-sample
            # generated-data metrics can be noisy run-to-run.
            n_repeats = 1 if first_full_eval_done else mentor_stability_repeats
            full_metrics = _full_checkpoint_eval(ckpt_path_full, it, log, n_repeats=n_repeats)

            if not first_full_eval_done and full_metrics is not None:
                std = full_metrics.get("mentor_macro_f1_std")
                if std is not None:
                    log.info(
                        f"  [mentor stability check] mentor_macro_f1 spread over "
                        f"{full_metrics['n_repeats']} runs: std={std:.4f} "
                        f"(configured early_stopping.min_delta={es_min_delta})"
                    )
                    if std >= es_min_delta:
                        log.warning(
                            f"  ⚠ mentor_macro_f1's run-to-run std ({std:.4f}) is >= "
                            f"the configured min_delta ({es_min_delta:.4f}). This "
                            "metric cannot currently distinguish real improvement "
                            "from sampling noise for early stopping / checkpoint "
                            "selection at this checkpoint count. Widen min_delta or "
                            "increase classification_validation.py's samples/class "
                            "before trusting a stop/selection decision from a single "
                            "reading — do not proceed on this as just a caveat."
                        )
                    else:
                        log.info(
                            "  ✓ mentor_macro_f1 spread is below min_delta — "
                            "single-reading decisions at later full-eval "
                            "checkpoints are meaningful relative to this threshold."
                        )
                else:
                    log.warning(
                        "  [mentor stability check] could not compute std "
                        f"(only {full_metrics.get('n_repeats', 0)} successful run(s) "
                        "out of the requested repeats) — stability unverified."
                    )
                first_full_eval_done = True

            if full_metrics is not None:
                _append_checkpoint_metrics(
                    results_dir, {"iteration": it, "kind": "full", **full_metrics}
                )

                # ── Checkpoint selection (heavy metric only) ────────────────────
                sel_val = full_metrics.get(sel_metric_name)
                if sel_val is not None and _is_better(sel_val, best_selected_metric, sel_mode):
                    best_selected_metric = sel_val
                    sel_ckpt_path = models_dir / "diffusion_rl_selected.pt"
                    shutil.copy(str(ckpt_path_full), str(sel_ckpt_path))
                    log.info(
                        f"  ✓ New selected checkpoint (by {sel_metric_name}="
                        f"{sel_val:.4f}) → {sel_ckpt_path.name}"
                    )
                elif sel_val is None:
                    log.warning(
                        f"  checkpoint_selection metric {sel_metric_name!r} not found "
                        f"in full-eval metrics {list(full_metrics)} — skipping selection "
                        f"at iteration {it}."
                    )

                # ── Early stopping (heavy metric only) ──────────────────────────
                if es_enabled:
                    es_val = full_metrics.get(es_metric)
                    if es_val is None:
                        log.warning(
                            f"  early_stopping metric {es_metric!r} not found in "
                            f"full-eval metrics — cannot evaluate stop condition "
                            f"at iteration {it}."
                        )
                    elif _is_better(es_val, es_best, es_mode, min_delta=es_min_delta):
                        es_best = es_val
                        es_bad_count = 0
                    else:
                        es_bad_count += 1
                        log.info(
                            f"  early_stopping: no improvement in {es_metric} "
                            f"({es_bad_count}/{es_patience} full-eval checkpoints)"
                        )
                        if es_bad_count >= es_patience:
                            log.warning(
                                f"  ⚠ Early stopping triggered at iteration {it}: "
                                f"{es_metric} has not improved for {es_patience} "
                                f"consecutive full-eval checkpoints."
                            )
                            stop_early = True
            else:
                log.warning(
                    f"  Full eval at iteration {it} produced no metrics — "
                    "checkpoint selection and early stopping skipped for this checkpoint."
                )

        if stop_early:
            break

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

    if smoke_test:
        _report_smoke_test_results(
            policy_model, frozen_model, diffusion, class_names, cfg, device,
            smoke_param_checksum_before,
            smoke_kl_history, smoke_reward_history, smoke_grad_norm_history,
            reward_fn, results_dir, log,
        )

    return best_reward


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="RL fine-tuning of the ECG diffusion model.")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run a short (rl.smoke_test_iterations) correctness check instead "
             "of the full training schedule — verifies the policy actually "
             "updates under PPO/GRPO before committing GPU-hours to a real run. "
             "See outputs/results/smoke_test_report.json for the result.",
    )
    parser.add_argument(
        "--n-iterations", type=int, default=None,
        help="Run exactly this many iterations instead of rl.rl_iterations "
             "(config.yaml) — for a validation-length run (e.g. Gate 3's "
             "100-250 updates) shorter than a full training run but longer "
             "than --smoke-test. Ignored if --smoke-test is also passed.",
    )
    parser.add_argument(
        "--run-tag", type=str, default=None,
        help="Write this run's checkpoints/logs/results (rl_ckpt_iter*.pt, "
             "diffusion_rl_best.pt, rl_training_log.csv, progress plots, "
             "smoke_test_report.json, checkpoint_metrics.json) under a "
             "subdirectory with this name inside each of outputs/models/, "
             "logs/, and outputs/results/ — so a validation run's outputs "
             "can't collide with / be silently overwritten by a later run "
             "that reuses the same iteration numbers. Still reads "
             "diffusion_best.pt from the untagged models dir.",
    )
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("step07_rl_finetuning", cfg=cfg)
    set_seed(int(cfg.seeds[0]))

    best_reward = train(
        cfg, log, smoke_test=args.smoke_test,
        n_iterations_override=args.n_iterations, run_tag=args.run_tag,
    )
    if args.smoke_test:
        print("✓ Smoke test complete — see outputs/results/smoke_test_report.json")
    else:
        print(f"✓ RL fine-tuning complete. Best total reward: {best_reward:.3f}")


if __name__ == "__main__":
    main()
