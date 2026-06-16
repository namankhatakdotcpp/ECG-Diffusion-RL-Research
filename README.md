# ECG Diffusion + RL Research
**HCLTech Internship | Naman Khatak | June 2026**

## Project in One Sentence

Train a Transformer-backbone Diffusion Model to generate realistic synthetic
ECGs for multiple disease classes, then fine-tune it with PPO/GRPO
Reinforcement Learning using a clinically grounded reward function to make the
generated signals more physiologically valid.

## Architecture

```
PTB-XL Dataset (multiple disease classes)
        │
        ▼
  Preprocessing
        │
        ▼
  Transformer-Diffusion Model
  ┌─────────────────────────────────────────┐
  │  Class Label Embedding (disease type)    │
  │          +                               │
  │  Time Step Embedding (diffusion step t)  │
  │          │                               │
  │          ▼                               │
  │  Transformer Encoder Backbone            │
  │  (replaces the standard UNet)            │
  │  - Multi-head self-attention             │
  │  - Captures long-range ECG dependencies  │
  │          │                               │
  │          ▼                               │
  │  Predicts noise ε at each timestep       │
  └─────────────────────────────────────────┘
        │
        ▼
  Generate synthetic ECG for any disease class
        │
        ▼
  Clinical Reward Function
  (morphology + HRV + realism + diagnostic utility)
        │
        ▼
  PPO / GRPO RL Fine-Tuning
        │
        ▼
  RL-Optimised ECG Generation
        │
        ▼
  Evaluation (DTW, MMD, Morphological Validity, TSTR) + Ablation Study
```

## Directory Layout

```
ECG/
├── config.yaml                            # single source of truth — all params/paths
├── requirements.txt
├── README.md
├── utils/                                  # shared helpers
│   ├── config.py    — load_config()
│   ├── logger.py    — get_logger()
│   ├── seed.py       — set_seed()
│   └── metrics.py    — dtw_distance, mmd_score, per_class_f1, morphological_validity
│
├── step01_data_load_and_visualise.py       # download, verify, visualise PTB-XL
├── step02_preprocessing.py                 # filter, normalise, fold split, save arrays
├── step03_eda_and_class_mapping.py         # decide final classes, morphology/HRV stats
├── step04_transformer_diffusion.py         # train the Transformer-backbone diffusion model
├── step05_baseline_eval.py                 # DTW/MMD/FED/Morph/TSTR — Table 1
├── step06_reward_function.py               # clinical reward: morphology+HRV+realism+diag
├── step07_rl_finetuning.py                  # PPO/GRPO RL fine-tuning with KL regularisation
├── step08_final_evaluation.py               # baseline vs RL head-to-head — Table 2 + figures
├── step09_ablation_study.py                 # 6-variant reward-component ablation — Table 3
│
├── data/                                    (gitignored)
│   ├── ptbxl/
│   └── mitbih/                              (secondary, not used by the core pipeline)
└── outputs/                                 (gitignored)
    ├── processed/
    ├── models/
    ├── generated/
    └── results/
```

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline in order
python step01_data_load_and_visualise.py   # ~10 min (download + visualise)
python step02_preprocessing.py             # ~20 min
python step03_eda_and_class_mapping.py     # ~30 min (morphology extraction)
python step04_transformer_diffusion.py     # ~hours (GPU needed)
python step05_baseline_eval.py             # ~1 hour
python step06_reward_function.py           # ~30 min
python step07_rl_finetuning.py             # ~hours (GPU needed)
python step08_final_evaluation.py          # ~1 hour
python step09_ablation_study.py            # ~hours (6 RL runs)
```

## Hardware Requirements

- GPU strongly recommended for step04 (diffusion training) and step07
  (RL fine-tuning) — both run in minutes on a modern GPU vs. hours on CPU.
- ~2 GB disk for PTB-XL raw data, a few hundred MB for processed arrays and
  checkpoints.

## Configuration

All hyperparameters live in **`config.yaml`** — never hard-code paths or
settings in step files. Load it from any script with:

```python
from utils import load_config
cfg = load_config()
print(cfg.diffusion.T)            # 1000
print(cfg.paths.data.ptbxl)       # resolved path to PTB-XL
```

## Reproducibility

Three seeds are defined in `config.yaml` (`seeds: [42, 123, 456]`).
Each step file calls `set_seed(seed)` at the top:

```python
from utils import set_seed
set_seed(cfg.seeds[0])
```

Evaluation steps (05, 08, 09) report metrics as mean ± std across all three
seeds.

## Experiment Tracking

W&B is off by default. Enable in `config.yaml`:

```yaml
logging:
  use_wandb: true
  wandb_project: ecg-diffusion-rl
  wandb_entity: your-username
```
