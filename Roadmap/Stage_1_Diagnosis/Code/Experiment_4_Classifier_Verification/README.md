# Experiment 4 — MentorClassifier Verification (AFIB reject-class test)

## Why this matters before trusting Experiments 1-3

Every other Stage 1 experiment uses MentorClassifier's predictions as the
ground-truth signal for "did conditioning work." If AFIB behaves as a
catch-all bucket for anything the classifier doesn't recognize — rather
than a genuine learned AFIB concept — then any conditioning-failure
conclusion involving AFIB (or any noisy/edge-case sample landing there)
would be an artifact of the classifier, not the diffusion model.

## Why this experiment needs no GPU and no diffusion checkpoint

It never touches the diffusion model. It only corrupts REAL PTB-XL ECGs
with Gaussian noise and watches what MentorClassifier does — this is
exactly why it could be run directly on the development laptop rather than
waiting for the GPU server round-trip.

## What it measures (see script docstring for full method)

1. **Noise robustness** — per-class accuracy as Gaussian noise increases.
2. **Prediction drift** — what fraction of predictions flip as noise
   increases.
3. **AFIB attraction ratio** — of predictions that flip due to noise, what
   fraction land on AFIB, relative to a 1-in-4 chance baseline. A ratio
   that grows above 1 as noise increases is direct evidence AFIB is
   absorbing corrupted/unrecognizable input rather than representing a
   learned rhythm concept.
4. **Confidence calibration** — mean softmax confidence of predictions, by
   predicted class, as noise increases. A well-calibrated classifier should
   get less confident as its input is destroyed; if AFIB predictions stay
   confident under heavy noise while other classes' confidence drops, the
   model is confidently wrong on AFIB specifically.

## How to run

```bash
python Roadmap/Stage_1_Diagnosis/Code/Experiment_4_Classifier_Verification/classifier_verification.py
```

No arguments — always evaluates on the official PTB-XL test fold, all 4
mentor classes, `NOISE_LEVELS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]` (z-score
units, matching the `clip_range=[-4,4]` used everywhere else in this repo).

## Experiment 4.5 — Feature Drift Visualization

`feature_drift_visualization.py` makes Experiment 4's numeric AFIB-attraction
finding visual: it projects clean real, progressively-noised real (same
underlying ECGs, so individual samples' drift paths can be traced), and —
once Experiment 1's checkpoint exists — generated samples into a single
shared PCA space. The real+noise half needs no checkpoint and was run
locally alongside Experiment 4; the generated half is added automatically
once a checkpoint is found at `outputs/models/diffusion_best.pt` (no flag
needed — just re-run after Experiment 1 completes).

```bash
python Roadmap/Stage_1_Diagnosis/Code/Experiment_4_Classifier_Verification/feature_drift_visualization.py
```

Hand back: `feature_drift_features.csv`, `feature_drift_pca.png`.

## Status

Run locally on the development machine on 2026-07-02 (in progress/complete
— see `Experiment_Log.md` for the actual outcome and
`Reports/classifier_validation_report.md` for the write-up).
