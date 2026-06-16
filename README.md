# ECG Diffusion + RL Research
**HCLTech Internship | Naman Khatak | June 2026**

## Project Structure

Two parallel tracks targeting a single publication:

```
TRACK A — GENERATION (this repo's primary focus)
  PTB-XL → Preprocessing → Baseline Diffusion (SSSD-ECG style)
         → Disease-to-Healthy Translation (MI → NORM)
         → PPO/GRPO RL Fine-Tuning with Clinical Reward
         → RL-Optimised ECG Generation → TSTR Evaluation

TRACK B — CLASSIFICATION (teammate's track, shared data pipeline)
  PTB-XL → Preprocessing → Multimodal Transformer
         → 7-class multi-label ECG classification
         → Doubles as the TSTR evaluator for Track A
```

## Directory Layout

```
ecg_research/
├── config.yaml                    # ALL shared hyperparameters and paths
├── requirements.txt
├── utils/                         # shared helpers (config, logging, seeding, metrics)
│
├── step01_data_download.py        # download PTB-XL (and MIT-BIH)
├── step02_preprocessing.py        # normalise, segment, split
├── step03_eda_and_validation.py   # class distribution, signal quality checks
├── step04_baseline_diffusion.py   # train SSSD-ECG style diffusion model
├── step05_diffusion_eval.py       # FID-style, DTW, MMD evaluation
├── step06_classifier_track_b.py   # (teammate) multimodal Transformer classifier
├── step07_reward_function.py      # clinical reward: PQRST + classifier confidence
├── step08_rl_finetuning.py        # PPO / GRPO fine-tuning of diffusion model
├── step09_tstr_evaluation.py      # Train-on-Synthetic Test-on-Real evaluation
└── step10_ablation_study.py       # ablation over reward weights / RL algorithms
│
├── data/
│   ├── ptbxl/                     # raw PTB-XL download
│   └── mitbih/                    # raw MIT-BIH download
└── outputs/
    ├── processed/                 # preprocessed tensors / DataLoaders
    ├── models/                    # saved checkpoints
    ├── generated/                 # synthetic ECG waveforms
    └── results/                   # evaluation tables, plots
```

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline in order
python step01_data_download.py
python step02_preprocessing.py
python step03_eda_and_validation.py
python step04_baseline_diffusion.py
python step05_diffusion_eval.py
# step06 is the teammate's classifier — needed before step07+
python step07_reward_function.py
python step08_rl_finetuning.py
python step09_tstr_evaluation.py
python step10_ablation_study.py
```

## Configuration

All hyperparameters live in **`config.yaml`** — never hard-code paths or
settings in step files. Load it from any script with:

```python
from utils import load_config
cfg = load_config()
print(cfg.diffusion.T)       # 1000
print(cfg.paths.data.ptbxl)  # absolute path resolved at load time
```

## Reproducibility

Three seeds are defined in `config.yaml` (`seeds: [42, 123, 456]`).
Each step file calls `set_seed(seed)` at the top:

```python
from utils import set_seed
set_seed(cfg.seeds[0])
```

## Experiment Tracking

W&B is off by default. Enable in `config.yaml`:

```yaml
logging:
  use_wandb: true
  wandb_project: ecg-diffusion-rl
  wandb_entity: your-username
```
