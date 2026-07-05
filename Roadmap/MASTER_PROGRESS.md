# Master Progress

Current status (updated 2026-07-06) -- see each stage's own status file for
detail, since that is where status is actually kept current; this table is
a top-level index, not the source of truth:

| Stage | Status | Detail |
|---|---|---|
| Stage 1 — Diagnosis | **Complete** | `Stage_1_Diagnosis/Decisions.md` |
| Stage 2 — Architecture Investigation | **Complete** | `Stage_2_Architecture_Investigation/STAGE2_STATUS.md` |
| Stage 3 — Architecture Improvements | **In progress** — Phase 0 complete, candidates S3-001..006 implemented, GPU training underway | `Stage_3_Architecture_Improvements/Stage3_Status.md` |
| Stage 4 — Optimization | Not started | Gated on Stage 3 |
| Stage 5 — Final Model | Not started | Gated on Stage 4 |

---

## Historical snapshot below (as of 2026-07-02) -- superseded by the table above

The section below was accurate on 2026-07-02, while Stage 1 was still
in progress. Left in place as a dated log entry (this project's own
provenance convention), not updated in place -- do not read it as
current status; the table above and each stage's own status file are
current.

## Stage 1 status detail

- **Architecture.md** (project root) — complete. Key findings: no trained
  checkpoint/logs exist on the dev laptop; `cfg_sweep_result.txt` is stale
  and contradicts the current CFG implementation (now fixed-in-place, code
  is trusted); AFIB has no trained diffusion class (merged into OTHER,
  103 records < `min_class_samples=200`).
- **Compute plan** — this laptop cannot train the diffusion model (CPU:
  23s/step, ~29 days for 200 epochs; MPS: OOMs at batch=32 on 8GB unified
  memory). Fixed a real bug along the way: `_resolve_device()` in
  `step04_transformer_diffusion.py` never checked MPS, only cuda/cpu.
  User confirmed a GPU server is available; Experiments 1-3 will run there.
- **Experiment 1 (Baseline Reproduction)** — code ready
  (`Stage_1_Diagnosis/Code/Experiment_1_Baseline/run_experiment_1.sh`).
  Not yet run.
- **Experiment 2 (Dataset Scaling)** — new code written
  (`Stage_1_Diagnosis/Code/Experiment_2_Dataset_Scaling/run_dataset_scaling.py`):
  stratified subsets at 380/1000/2500/5000/10000/full, one fixed
  MentorClassifier reused across all sizes to isolate the diffusion
  model's data size as the only variable. Not yet run.
- **Experiment 3 (Directional Conditioning Analysis)** — new code written
  (`Stage_1_Diagnosis/Code/Experiment_3_Directional_Probe/directional_conditioning_probe.py`):
  paired same-seed generation across class labels, cosine similarity
  between the generated embedding displacement and the real-centroid
  displacement in MentorClassifier's 128-dim feature space. Not yet run.
- **Experiment 4 (MentorClassifier Verification)** — **COMPLETE.** Needed
  no GPU/checkpoint, so it ran directly on this machine. Headline result:
  AFIB behaves as an out-of-distribution reject class, but only at
  moderate noise (sigma≈0.25-0.5: absorbs 80-90% of all misclassifications,
  3.2-3.6x chance rate); at extreme noise the same role shifts to NSTEMI.
  AFIB is also the weakest class on clean data (accuracy 0.455 vs.
  Normal's 0.974). Full detail:
  `Stage_1_Diagnosis/Reports/classifier_validation_report.md`.
- **Experiment 5 (Decision Report)** — blocked on Experiments 1-3.

## Sub-experiments added (2026-07-02, per user request)

Four additional low-cost experiments now exist, each closing a specific
gap the corresponding core experiment leaves open — see
`Stage_1_Diagnosis/README.md` for the full rationale:
- **1.5 Checkpoint Verification** — conditioning vs. epoch, using
  Experiment 1's already-saved per-epoch checkpoints (near-zero extra cost).
- **2.5 Training Curves** — built into `run_dataset_scaling.py`
  (`--curve-every`); loss/sensitivity/collapse/F1 over training per size.
- **3.5 Layer-wise Direction Probe** — conditioning magnitude + direction
  consistency at each of the 6 Transformer blocks (pure forward passes,
  cheap).
- **4.5 Feature Drift Visualization** — real→noise→generated in one shared
  PCA space. **Real+noise half already run locally** (no checkpoint
  needed); generated-sample overlay auto-added once Experiment 1 completes.

## One-command reproducibility

`bash Roadmap/Stage_1_Diagnosis/run_stage1.sh` runs Experiments
1→1.5→2(+2.5)→3→3.5→4(skipped, already done)→4.5 in dependency order, then
runs `collect_stage1_results.py`, which writes
`Reports/Stage1_Results_Digest.md` — a mechanical table of every result
found on disk (tested now against the currently-available Experiment 4/4.5
outputs; works correctly). This digest is not a substitute for the
narrative reports — interpreting what the numbers mean still requires
Claude to read them and write `Stage1_Final_Report.md`.

## Next action

Run `bash Roadmap/Stage_1_Diagnosis/run_stage1.sh` on the GPU server (single
command, per-experiment READMEs still document manual steps if preferred).
Once it completes, hand back the `Roadmap/Stage_1_Diagnosis/Outputs/` and
`Figures/` trees plus `Reports/Stage1_Results_Digest.md`. Claude then fills
in `baseline_report.md`, `dataset_scaling_report.md`, and
`directional_conditioning_report.md` (currently stubs — no numbers
fabricated) and writes `Stage1_Final_Report.md`, incorporating Experiment
4's already-complete finding.
