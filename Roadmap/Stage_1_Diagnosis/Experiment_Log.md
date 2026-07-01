# Stage 1 Experiment Log

## 2026-07-02 — Initial repo audit

- Confirmed outputs/models/ is empty (no diffusion_best.pt, no diffusion_ckpt_ep*.pt) on this machine.
- Confirmed logs/ is empty (no diffusion_training_log.csv).
- Confirmed outputs/generated/ is empty.
- Confirmed torch.cuda.is_available() == False, torch.backends.mps.is_available() == True on this machine.
- Confirmed config.yaml device: auto only checks cuda, never mps -> silently trains on CPU on this Mac unless fixed.
- Confirmed outputs/conditioning_analysis/cfg_sweep_result.txt claims CFG is unimplemented; contradicts current step04_transformer_diffusion.py (null class token, p_uncond dropout, two-pass guided ddim_sample all present). Report predates commits 3a2b035..231703f. Flagged as stale in MASTER_PROGRESS.md.
- outputs/processed/ already contains full preprocessed PTB-XL split: X_train (17418,1000,12), X_val (2183,...), X_test (2198,...). Class counts: NORM 7386/928/932, MI 3374/401/411, STTC 2651/338/350, CD 2630/340/351, HYP 1036/139/113, OTHER 254/29/30 (train/val/test).

## 2026-07-02 — Compute benchmark and decision

- Benchmarked step04's ECGTransformerDiffusion (8.43M params) training step on this machine:
  - CPU: 23,057.6 ms/step at batch=32 -> ~29 days for 200 epochs x 544 steps/epoch. Infeasible.
  - MPS: RuntimeError, out of memory (allocated 8.43 GiB, max allowed 9.07 GiB) on the FIRST step at batch=32. This machine has only 8GB total unified memory (Apple M3, 8 cores).
- Fixed step04_transformer_diffusion.py:_resolve_device() to check torch.backends.mps.is_available() (previously only checked cuda, silently falling back to cpu on this Mac even with device: auto).
- Asked the user how Stage 1 compute should be sourced. Answer: "I have a GPU server" -- user will run prepared scripts there and hand back outputs.
- Prepared, but did not run (no GPU/checkpoint on this machine), Experiment 1 (run_experiment_1.sh, orchestrates existing step04 + mentor_eval scripts), Experiment 2 (new run_dataset_scaling.py: stratified subsets at 380/1000/2500/5000/10000/full, fixed classifier across sizes, records time/GPU-mem/accuracy/macro-F1/collapse), and Experiment 3 (new directional_conditioning_probe.py: paired same-seed generation across class labels, cosine similarity between generated displacement and real centroid displacement in MentorClassifier embedding space).

## 2026-07-02 — Experiment 4 run locally (no GPU/checkpoint needed)

- Ran Code/Experiment_4_Classifier_Verification/classifier_verification.py directly on this machine (real-data-only, no diffusion model involved).
- MentorClassifier trained fresh: best val macro F1 = 0.7243.
- Test-fold per-class counts: Normal 963, STEMI 516, NSTEMI 185, AFIB 44.
- Key result: AFIB attraction ratio (share of noise-driven misclassifications landing on AFIB, relative to 1/4 chance) peaks at 3.58x at sigma=0.5 (89.5% of all flips -> AFIB), then falls below chance (0.93x) at sigma=2.0 as NSTEMI takes over as the dominant flip target (76.8% of flips at sigma=2.0).
- AFIB is already the weakest class on clean data (accuracy=0.455, lowest confidence=0.653 at sigma=0).
- Full writeup: Reports/classifier_validation_report.md.

## 2026-07-02 — Experiment 4.5 (real+noise half) run locally

- Ran Code/Experiment_4_Classifier_Verification/feature_drift_visualization.py.
- Reused the cached MentorClassifier from Experiment 4 (outputs/conditioning_analysis/mentor_classifier.pt).
- Real+noise half complete: feature_drift_pca.png, feature_drift_features.csv in place. Generated-sample half is [PARTIAL] pending Experiment 1's checkpoint -- script auto-detects outputs/models/diffusion_best.pt and adds the generated overlay with no extra flags once it exists; just re-run after Experiment 1 completes on the GPU server.
