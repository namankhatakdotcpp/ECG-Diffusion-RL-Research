# Experiment 2 — Dataset Scaling

**Status: NOT YET RUN.** Code is ready at
`Roadmap/Stage_1_Diagnosis/Code/Experiment_2_Dataset_Scaling/run_dataset_scaling.py`
— to be executed on the GPU server. This file will be filled in from
`Outputs/Experiment_2_Dataset_Scaling/dataset_scaling_metrics.csv` once
that run completes.

## Question this experiment answers

Does increasing PTB-XL training data alone (380 → 1000 → 2500 → 5000 →
10000 → full ≈17.4k), with everything else held fixed (architecture,
epochs, a single fixed MentorClassifier used to score every size), improve
class-conditioning fidelity? Or does it plateau — which would be evidence
that the bottleneck is architectural rather than data volume, and Stage 2/3
(more data collection or a joint-dataset approach) would not be justified
by this evidence alone.

## Planned analysis (to be filled in after the GPU run)

- Table: dataset_size, n_train_records, accuracy, macro_f1, collapse_frac,
  train_time_sec, peak_gpu_mem_gb, final_train_loss.
- Plot interpretation: is the accuracy/macro-F1 curve still rising at
  "full", or has it visibly plateaued by 2500-5000? A plateau well before
  "full" is the strongest evidence for "architecture is the bottleneck, not
  data."
- Collapse-fraction trend: does collapse decrease monotonically with more
  data, or does it stay high (near the ~0.33-1.0 range, given 3
  generatable mentor classes) regardless of size? A flat, high collapse
  curve across all sizes would suggest the conditioning pathway itself
  (not insufficient examples) is broken.
- Cross-reference with Experiment 3: does the directional conditioning
  score improve with dataset size, run per checkpoint size using
  `directional_conditioning_probe.py --ckpt .../size_{N}/diffusion_best.pt`?
