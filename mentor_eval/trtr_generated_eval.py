"""
mentor_eval/trtr_generated_eval.py — Condition 3 evaluation (reliability-alpha
sweep): score a diffusion/RL checkpoint's GENERATED samples with the frozen,
real-data-trained TRTR classifier (outputs/models/trtr_classifier.pt), in the
native 6-class taxonomy (NORM/MI/STTC/CD/HYP/OTHER, per
outputs/processed/class_names.json).

This is distinct from mentor_eval/classification_validation.py, which trains a
FRESH 4-class mentor-proxy classifier (Normal/STEMI/NSTEMI/AFIB) every
invocation and never touches trtr_classifier.pt. Decisions.md's Condition 3
("r_diag increases >5% while TRTR macro-F1 drops or stays flat") is defined in
terms of the frozen 6-class classifier trtr_classifier_eval.json already
documents -- so this script reuses that exact classifier (loaded once,
never retrained here) and asks: what does IT think of THIS checkpoint's
generated samples?

Generation uses LIVE policy weights, not EMA -- matching how
step07_rl_finetuning.py's PPO rollouts and reward function actually score the
policy during training (collect_rollouts uses self.policy directly; EMA is
updated but never sampled from). This also matches
mentor_eval/checkpoint_utils.py's generate_for_class(use_ema=False) default,
which exists because EMA shadow weights were diagnosed (2026-06-25) as
severely undertrained relative to live weights, producing pure noise when
sampled from. Do NOT switch this to EMA without re-checking that diagnosis --
it would silently evaluate different (and likely broken) weights than what
was actually trained.

Sample count defaults to cfg.eval.n_synthetic_per_class (500), this
project's standard full-evaluation sample count (used by
step05_baseline_eval.py's own TSTR/TRTR/DTW/etc. pipeline) -- NOT
mentor_eval/classification_validation.py's smaller n=100 default, which is
specific to that script's separate 4-class proxy and not this project's
main convention.

Usage:
    python -m mentor_eval.trtr_generated_eval --ckpt PATH --out-dir PATH
        [--n-gen-samples N] [--seed S]

Writes:
    <out-dir>/trtr_generated_eval.json
        {accuracy, macro_f1, per_class_f1, class_names, n_gen_samples,
         excluded_classes, ckpt_path, seed}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_config, get_logger
from step05_baseline_eval import _load_diffusion_model, Simple1DCNN
from step04_transformer_diffusion import generate_ecg


def _load_trtr_classifier(cfg, device: str, log):
    """Load the frozen, real-data-trained TRTR classifier. Fails loudly --
    no fallback to a neutral/default score -- since a silent fallback here
    would misrepresent Condition 3 as 'evaluated' when it wasn't.
    """
    path = Path(cfg.paths.outputs.models) / "trtr_classifier.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- Condition 3 requires the frozen TRTR "
            f"classifier already used by DiagnosticUtilityReward. Run "
            f"step05_baseline_eval.py first to produce it."
        )
    ckpt = torch.load(str(path), map_location=device)
    nc = ckpt.get("n_classes")
    if nc is None:
        raise KeyError(f"{path}: checkpoint missing 'n_classes' key.")
    clf = Simple1DCNN(n_classes=nc).to(device)
    clf.load_state_dict(ckpt["state_dict"])
    clf.eval()
    log.info(f"Loaded frozen TRTR classifier from {path} (n_classes={nc}).")
    return clf


def run(ckpt_path: Path, out_dir: Path, cfg, seed: int, log, n_gen_samples: int) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading diffusion/RL checkpoint from {ckpt_path} ...")
    model, diffusion, ema, class_names, _device = _load_diffusion_model(
        cfg, log, ckpt_path=str(ckpt_path)
    )
    n_classes = len(class_names)
    del ema  # unused -- generation below uses live weights, not EMA (see module docstring)

    clf = _load_trtr_classifier(cfg, device, log)

    log.info(
        f"Generating {n_gen_samples}/class x {n_classes} classes "
        f"(seed={seed}, live weights, DDIM) ..."
    )
    gen_X_list, gen_y_list, excluded_classes = [], [], []
    for cls_idx, cls_name in enumerate(class_names):
        try:
            samples = generate_ecg(
                model=model, diffusion=diffusion, class_label=cls_idx,
                n_samples=n_gen_samples, device=device, cfg=cfg,
                seed=seed + cls_idx, stats=None,  # z-score space, matches classifier training space
            )  # (n_gen_samples, 1000, 12)
        except Exception as exc:
            log.warning(f"Generation failed for class {cls_name!r}: {exc}. Excluding from macro-F1.")
            excluded_classes.append(cls_name)
            continue
        gen_X_list.append(samples)
        gen_y_list.append(np.full(len(samples), cls_idx, dtype=np.int64))

    if not gen_X_list:
        raise RuntimeError("No classes could be generated -- cannot compute Condition 3.")

    gen_X = np.concatenate(gen_X_list, axis=0)   # (N, 1000, 12)
    gen_y = np.concatenate(gen_y_list, axis=0)   # (N,)

    log.info(f"Classifying {len(gen_X)} generated samples with the frozen TRTR classifier ...")
    x = torch.from_numpy(gen_X.transpose(0, 2, 1)).float()  # (N, 12, 1000), matches DiagnosticUtilityReward's ecg.T convention
    preds = []
    batch = 256
    with torch.no_grad():
        for i in range(0, len(x), batch):
            logits = clf(x[i:i + batch].to(device))
            preds.append(logits.argmax(dim=1).cpu().numpy())
    preds = np.concatenate(preds)

    eval_labels = [i for i in range(n_classes) if class_names[i] not in excluded_classes]
    accuracy = float(accuracy_score(gen_y, preds))
    macro_f1 = float(f1_score(gen_y, preds, average="macro", zero_division=0, labels=eval_labels))
    per_class_f1_vals = f1_score(gen_y, preds, average=None, zero_division=0, labels=eval_labels)
    per_class_f1 = {class_names[i]: float(f) for i, f in zip(eval_labels, per_class_f1_vals)}

    result = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "class_names": class_names,
        "n_gen_samples": n_gen_samples,
        "excluded_classes": excluded_classes,
        "ckpt_path": str(ckpt_path),
        "seed": seed,
    }
    out_path = out_dir / "trtr_generated_eval.json"
    out_path.write_text(json.dumps(result, indent=2))
    log.info(f"Wrote {out_path}")
    print(
        f"accuracy={accuracy:.4f}  macro_f1={macro_f1:.4f}  "
        f"excluded={excluded_classes}  n_gen_samples={n_gen_samples}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Condition 3 eval: score a checkpoint's generated samples with the frozen 6-class TRTR classifier."
    )
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--n-gen-samples", type=int, default=None,
                         help="Default: cfg.eval.n_synthetic_per_class (this project's standard full-eval sample count).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not Path(args.ckpt).exists():
        print(f"[FATAL] --ckpt path does not exist: {args.ckpt}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config()
    log = get_logger(__name__)
    n_gen_samples = args.n_gen_samples if args.n_gen_samples is not None else int(cfg.eval.n_synthetic_per_class)

    run(Path(args.ckpt), Path(args.out_dir), cfg, args.seed, log, n_gen_samples)


if __name__ == "__main__":
    main()
