# Experiment 2 — Dataset Scaling

No new dataset is downloaded. This trains 6 independent diffusion models —
identical architecture and hyperparameters, only the training-data size
differs — on stratified subsets of the PTB-XL data already in
`outputs/processed/`.

## Design choices (documented so results are interpretable, not just numbers)

- **Sizes:** 380, 1000, 2500, 5000, 10000, full (~17.4k after class mapping).
  Each subset preserves the full set's per-class proportions (stratified,
  not uniform random) — otherwise a small subset could accidentally starve
  a rare class (OTHER has only 254 records total) and confound "less data"
  with "no data for this class at all."
- **Fixed classifier, not one-per-size:** a single MentorClassifier is
  trained once on the *full* real dataset and reused to score every dataset
  size's generated samples. If we retrained a classifier per size instead,
  a change in accuracy could come from a worse classifier rather than a
  worse generator — this design isolates the diffusion model's training
  data size as the only variable under test.
- **Same epoch count across all 6 runs** (`config.yaml`'s
  `diffusion.n_epochs`, override with `--epochs`): dataset size is the only
  thing that should vary between runs; changing epochs per run would
  confound "more data" with "more/less optimization."
- **Collapse fraction:** the fraction of ALL generated samples (pooled
  across every requested class) that the classifier assigns to whichever
  single mentor-class it predicts most often. High collapse (near 1.0)
  means the generator produces near-identical output regardless of the
  requested class — the core symptom Stage 1 exists to diagnose.

## How to run

```bash
python Roadmap/Stage_1_Diagnosis/Code/Experiment_2_Dataset_Scaling/run_dataset_scaling.py
# or, to control compute budget:
python Roadmap/Stage_1_Diagnosis/Code/Experiment_2_Dataset_Scaling/run_dataset_scaling.py --epochs 100 --sizes 380,1000,2500,5000,10000,full
```

This does not touch `outputs/models/diffusion_best.pt` (the Experiment 1
checkpoint) — every checkpoint here is written under
`Roadmap/Stage_1_Diagnosis/Outputs/Experiment_2_Dataset_Scaling/checkpoints/size_{N}/`.

## What to hand back

- `Roadmap/Stage_1_Diagnosis/Outputs/Experiment_2_Dataset_Scaling/dataset_scaling_metrics.csv`
  (and the `.json` copy)
- `Roadmap/Stage_1_Diagnosis/Figures/Experiment_2_Dataset_Scaling/*.png`
- Optionally the 6 checkpoints, if you want Claude to run further diagnostics
  (sensitivity probe, directional probe from Experiment 3) per size rather
  than just reading the summary metrics.

## Experiment 2.5 — training curves, not just final numbers

Every dataset size also records a `training_curves_size_{N}.csv` (loss,
sensitivity_metric, collapse_frac, macro_f1 every `--curve-every` epochs,
default 25) plus a combined `training_curves_all_sizes.png` at the end.
This exists for the same reason as Experiment 1.5: a dataset size that
looks bad in the final-epoch table might just need more epochs, and a flat
curve across sizes (all plateauing at the same collapse level regardless of
epoch) is much stronger evidence for "architecture is the bottleneck" than
final numbers alone. Control the cost with `--curve-every` (0 disables) and
`--curve-n-gen` (samples/class per snapshot, default 20 — small because
this runs many times per size).

## Time/memory note

This runs 6 full training loops sequentially. If GPU time is limited,
`--epochs` can be reduced for a faster first pass — the scaling *trend*
(does accuracy/collapse plateau or keep improving with more data?) is the
answer Stage 1 needs, not publication-grade absolute numbers. Record
whatever `--epochs` value was actually used in `Experiment_Log.md` so the
trend is interpreted against a fixed optimization budget.
