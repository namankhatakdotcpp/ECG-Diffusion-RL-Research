"""
Stage 2 / Tier 0 Item 8 -- Representation Collapse Analysis.

Implements the locked pre-registration
(Roadmap/Stage_2_Architecture_Investigation/Reports/Item8_PreRegistration.md).
Uses Roadmap/_infra/representation_metrics.py directly (fisher_ratio,
linear_probe_accuracy), per the master prompt's explicit instruction --
not reimplemented in common/. Runs BOTH at EVERY block, deliberately
overriding the module's own "probe only at 1-2 flagged blocks" cost-
saving suggestion, per the master prompt's explicit instruction.

Same cheap single-timestep synthetic-noise forward-pass convention as
Item 1/3/6 -- NOT full reverse-diffusion generation. "Hidden states"
are mean-pooled per-block activations from one forward pass per draw,
at a FIXED timestep per computation (never mixed across timesteps,
per the module's own docstring requirement).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
INFRA_DIR = REPO_ROOT / "Roadmap" / "_infra"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(INFRA_DIR) not in sys.path:
    sys.path.insert(0, str(INFRA_DIR))

from common.io import load_config, get_logger, load_model_checkpoint  # noqa: E402
from common.hooks import register_layer_hooks  # noqa: E402
from common.plotting import plot_representation_collapse  # noqa: E402
from representation_metrics import fisher_ratio, linear_probe_accuracy  # noqa: E402

OUT_DIR = (
    REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Outputs"
    / "stage2_tier0_item8_representation_collapse"
)
FIG_DIR = (
    REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Figures"
    / "stage2_tier0_item8_representation_collapse"
)
REPORT_DIR = REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Reports"

TIMESTEPS = [100, 500, 900]
N_GEN_PER_CLASS = 20
VERIFIED_MARGIN_ABOVE_CHANCE = 0.15
N_SPLITS = 3  # independent random train/test splits, per validity-check review
PERMUTATION_SEED = 12345


def main() -> None:
    cfg = load_config()
    log = get_logger("item8_representation_collapse", cfg=cfg)
    torch.manual_seed(0)

    loaded = load_model_checkpoint(cfg)
    if loaded is None:
        print("[BLOCKED] Checkpoint not found. Run Experiment 1 first.")
        return

    model = loaded.model
    device = loaded.device
    n_classes = loaded.n_classes
    n_layers = len(model.blocks)
    n_leads = 12
    seq_len = int(cfg.ptbxl.signal_length)
    model.eval()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    chance = 1.0 / n_classes
    log.info(f"Item 8 sweep: timesteps={TIMESTEPS}, n_gen_per_class={N_GEN_PER_CLASS}, "
             f"n_classes={n_classes}, n_layers={n_layers}, model.eval() mode, chance={chance:.4f}")

    rows = []
    raw = {"timesteps": TIMESTEPS, "n_gen_per_class": N_GEN_PER_CLASS, "n_classes": n_classes,
           "per_timestep": {}}

    for t_val in TIMESTEPS:
        # features[block] -> list of (feature_vec, label)
        features_by_block = [[] for _ in range(n_layers)]
        labels_all = []

        draw_idx = 0
        for cls in range(n_classes):
            for _ in range(N_GEN_PER_CLASS):
                torch.manual_seed(3000 + draw_idx)
                draw_idx += 1
                x_t = torch.randn(1, n_leads, seq_len, device=device)
                t = torch.full((1,), t_val, device=device, dtype=torch.long)
                y = torch.full((1,), cls, device=device, dtype=torch.long)

                handles, captured = register_layer_hooks(model)
                with torch.no_grad():
                    model(x_t, t, y)
                for h in handles:
                    h.remove()

                for layer in range(n_layers):
                    features_by_block[layer].append(captured[layer][0].numpy())
                labels_all.append(cls)

        labels_arr = np.array(labels_all)
        raw["per_timestep"][str(t_val)] = {}

        for layer in range(n_layers):
            X = np.stack(features_by_block[layer])
            d_hidden = X.shape[1]
            fr = fisher_ratio(X, labels_arr)
            lp = linear_probe_accuracy(X, labels_arr)

            # --- Validity check 1: n_train vs. d, reported explicitly, not just implied ---
            n_train_le_d = lp.n_train_samples <= d_hidden

            # --- Validity check 2: multiple random splits (not just the default seed=42) ---
            split_accuracies = [lp.accuracy]  # seed=42 result already computed above
            split_train_accuracies = [lp.train_accuracy]
            for split_seed in range(1, N_SPLITS):
                lp_split = linear_probe_accuracy(X, labels_arr, seed=42 + split_seed)
                split_accuracies.append(lp_split.accuracy)
                split_train_accuracies.append(lp_split.train_accuracy)

            # --- Validity check 3: label-permutation control ---
            rng = np.random.RandomState(PERMUTATION_SEED + layer + t_val)
            shuffled_labels = labels_arr.copy()
            rng.shuffle(shuffled_labels)
            lp_shuffled = linear_probe_accuracy(X, shuffled_labels)

            rows.append({
                "timestep": t_val, "block": layer + 1,
                "fisher_ratio": fr.fisher_ratio,
                "fisher_n_classes_used": fr.n_classes_used,
                "fisher_classes_excluded": fr.classes_excluded_low_n,
                "probe_accuracy": lp.accuracy,
                "probe_chance_accuracy": lp.chance_accuracy,
                "probe_train_accuracy": lp.train_accuracy,
                "probe_n_train_samples": lp.n_train_samples,
                "probe_n_test_samples": len(labels_arr) - lp.n_train_samples,
                "hidden_dim": d_hidden,
                "n_train_le_d": n_train_le_d,
                "probe_pca_components": lp.pca_components,
                "probe_classes_excluded": lp.classes_excluded_low_n,
                "multi_split_accuracies": split_accuracies,
                "multi_split_mean": float(np.mean(split_accuracies)),
                "multi_split_range": float(max(split_accuracies) - min(split_accuracies)),
                "permutation_test_accuracy": lp_shuffled.accuracy,
                "permutation_train_accuracy": lp_shuffled.train_accuracy,
                "permutation_at_or_near_chance": lp_shuffled.accuracy < chance + VERIFIED_MARGIN_ABOVE_CHANCE,
            })
            raw["per_timestep"][str(t_val)][str(layer + 1)] = {
                "fisher_ratio": fr.fisher_ratio, "fisher_warning": fr.warning,
                "probe_accuracy": lp.accuracy, "probe_warning": lp.warning,
                "hidden_dim": d_hidden, "n_train": lp.n_train_samples,
                "n_train_le_d": n_train_le_d,
                "multi_split_accuracies": split_accuracies,
                "permutation_test_accuracy": lp_shuffled.accuracy,
                "permutation_train_accuracy": lp_shuffled.train_accuracy,
            }
            log.info(f"t={t_val} block={layer+1}: Fisher={fr.fisher_ratio:.4f} "
                     f"probe_acc={lp.accuracy:.4f} (chance={lp.chance_accuracy:.4f}, "
                     f"pca_components={lp.pca_components}, train_acc={lp.train_accuracy:.4f}, "
                     f"n_train={lp.n_train_samples}, d={d_hidden}, n_train<=d={n_train_le_d}) | "
                     f"multi-split accs={[round(a,3) for a in split_accuracies]} | "
                     f"PERMUTATION test_acc={lp_shuffled.accuracy:.4f} "
                     f"(chance={chance:.4f}, train_acc={lp_shuffled.train_accuracy:.4f})")
            if fr.warning:
                log.info(f"  Fisher warning: {fr.warning}")
            if lp.warning:
                log.info(f"  Probe warning: {lp.warning}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "representation_collapse.csv", index=False)
    with open(OUT_DIR / "representation_collapse_raw.json", "w") as f:
        json.dump(raw, f, indent=2)

    fig_path = plot_representation_collapse(df, FIG_DIR, chance=chance)

    df["verdict"] = df.apply(
        lambda r: "VERIFIED" if r["probe_accuracy"] > chance + VERIFIED_MARGIN_ABOVE_CHANCE
        else "FALSIFIED", axis=1
    )

    print(df.to_string(index=False))
    print(f"\nChance accuracy: {chance:.4f}, VERIFIED margin threshold: {chance + VERIFIED_MARGIN_ABOVE_CHANCE:.4f}")
    print(f"Wrote: {fig_path}")


if __name__ == "__main__":
    main()
