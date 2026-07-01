"""
Roadmap/Stage_1_Diagnosis/Code/common_probes.py — small, cheap diagnostic
helpers shared across Stage 1 experiments (1.5 checkpoint verification, 2.5
training curves, 3.5 layer-wise probe). Kept here instead of duplicated
because the same magnitude-only conditioning measure is used at three
different granularities (per-checkpoint, per-training-step, per-layer).

Nothing here needs a checkpoint saved to disk — it takes an already-loaded
model, so callers control exactly which checkpoint/training-step state is
being probed.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from sklearn.metrics import f1_score


def sensitivity_metric(model, device: str, n_classes: int, cfg, timesteps: list[int] | None = None) -> float:
    """Magnitude-only class-conditioning effect: mean ||eps_A - eps_B|| / eps_scale,
    class 0 vs every other class, at a few fixed timesteps. Same measure as
    mentor_eval/conditioning_sensitivity_probe.py, factored out so it can be
    computed repeatedly (per-checkpoint, per-epoch, per-layer) without
    duplicating the forward-pass bookkeeping each time.
    """
    torch.manual_seed(0)
    n_leads = 12
    seq_len = int(cfg.ptbxl.signal_length)
    batch = 8
    x_t = torch.randn(batch, n_leads, seq_len, device=device)
    T = int(cfg.diffusion.T)
    if timesteps is None:
        timesteps = [T - 1, int(T * 0.5), 0]

    diffs = []
    with torch.no_grad():
        for t_val in timesteps:
            t = torch.full((batch,), t_val, device=device, dtype=torch.long)
            y_a = torch.full((batch,), 0, device=device, dtype=torch.long)
            eps_a = model(x_t, t, y_a)
            eps_scale = eps_a.flatten(1).norm(dim=1).mean().item()
            for cls_idx in range(1, n_classes):
                y_b = torch.full((batch,), cls_idx, device=device, dtype=torch.long)
                eps_b = model(x_t, t, y_b)
                diff = (eps_a - eps_b).flatten(1).norm(dim=1).mean().item()
                diffs.append(diff / (eps_scale + 1e-8))
    return float(np.mean(diffs))


def collapse_and_macro_f1(
    model, diffusion, class_names: list[str], clf, device: str, cfg, prep_stats,
    mentor_classes: list[str], mentor_to_trained_class: dict[str, str | None],
    n_gen_per_class: int, seed: int,
) -> tuple[float, float]:
    """Generate n_gen_per_class samples per generatable class, classify with
    the fixed MentorClassifier `clf`, return (collapse_frac, macro_f1).
    Shared by Experiment 1.5 (per-checkpoint) and Experiment 2/2.5
    (per-dataset-size and per-training-curve-point) so the metric definition
    can't silently drift between experiments.
    """
    from step04_transformer_diffusion import generate_ecg

    trained_to_mentor = {v: k for k, v in mentor_to_trained_class.items() if v is not None}
    mentor_name_to_idx = {n: i for i, n in enumerate(mentor_classes)}
    gen_X, gen_y = [], []
    for cls_name in class_names:
        mentor_name = trained_to_mentor.get(cls_name)
        if mentor_name is None:
            continue
        samples = generate_ecg(
            model, diffusion, class_label=class_names.index(cls_name),
            n_samples=n_gen_per_class, device=device, cfg=cfg, seed=seed,
            stats=prep_stats,
        )
        gen_X.append(samples)
        gen_y.append(np.full(len(samples), mentor_name_to_idx[mentor_name]))
    gen_X = np.concatenate(gen_X, axis=0)
    gen_y = np.concatenate(gen_y, axis=0)

    Xt = torch.from_numpy(gen_X.transpose(0, 2, 1)).float().to(device)
    clf.eval()
    with torch.no_grad():
        pred = clf(Xt).argmax(dim=1).cpu().numpy()
    macro_f1 = float(f1_score(gen_y, pred, average="macro", zero_division=0))
    counts = Counter(pred.tolist())
    collapse_frac = float(max(counts.values())) / len(pred)
    return collapse_frac, macro_f1
