# Stage 3 -- Cross-Candidate Comparison

**CAVEAT added retroactively (Stage 4 investigation, see
`Roadmap/Stage_4_Optimization/Decisions.md` -- "Second, separate bug..."
and its blast-radius follow-up entries)**: every "Generated Macro-F1"
value in this table comes from `mentor_eval.classification_validation`'s
`evaluate_classifier`, which prior to that fix had no explicit guard
excluding AFIB (never generated -- no trained-model class exists for it)
from the macro-F1 average. Whether any specific value below was actually
numerically affected depends on whether that candidate's classifier ever
predicted the AFIB index for at least one generated sample -- not
verified for any of these five runs (would need each run's raw confusion
matrix, most of which predate this investigation and may not be
preserved). **Since every candidate went through the identical pipeline,
the RELATIVE ranking between candidates is plausibly more robust than any
single absolute value** -- but this is not verified, only plausible.
Treat every number in the "Generated Macro-F1" column as provisional
pending recomputation with the fixed pipeline, and treat the S3-001
selection decision in `Roadmap/Stage_4_Optimization/Decisions.md` (made
partly on this column) accordingly.

All candidates evaluated via the identical `mentor_eval.classification_validation` pipeline against the frozen baseline manifest (checksum `16ac1715ac90ecb3db119de5611a3d2fff2cdc6ca82e53fb4d9c9c3a1864819d`, commit `6cf2e8b6c2352910b5b5dc23c2910a8cd0d2f7de`, source: frozen baseline_manifest.json). Optimizer-config column reflects whether the run's training commit predates the gain-parameter weight-decay fix (`0294330`/`432395c`).

Baseline (classifier trained on real data, evaluated on generated ECGs) accuracy: `0.4200`

| Candidate | Variant | Params (M) | Generated Accuracy | Generated Macro-F1 | Generated Macro-AUC | Real-data Accuracy | Delta vs. Baseline (accuracy) | Similarity Cosine (mean) | Similarity Mahalanobis (mean) | Sharma/Subband Metrics | Training Time | Optimizer Config | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S3-001 | baseline | 8.43 | 0.44 | 0.3421 | 0.8411 | 0.8331 | 0.02 | -- | -- | not computed (queue does not run subband_similarity_metrics) | 6h 21m | pre-fix | done |
| S3-002 | layerscale | 8.43 | 0.3967 | 0.2814 | 0.8382 | 0.8331 | -0.0233 | -- | -- | not computed (queue does not run subband_similarity_metrics) | -- | pre-fix | training |
| S3-003 | late_gain | 8.43 | 0.42 | 0.2406 | 0.8638 | -- | 0.0 | 0.1439 | 10.8173 | not computed (queue does not run subband_similarity_metrics) | 7h 29m | post-fix | done |
| S3-004 | residual_scaling | 8.43 | 0.3967 | 0.2863 | 0.8631 | -- | -0.0233 | 0.1437 | 8.8656 | not computed (queue does not run subband_similarity_metrics) | 6h 27m | post-fix | done |
| S3-005 | hybrid | 8.43 | 0.39 | 0.2059 | 0.8742 | -- | -0.03 | 0.1397 | 10.9448 | not computed (queue does not run subband_similarity_metrics) | 6h 44m | post-fix | done |
| S3-006 | final_norm_gain | 8.43 | -- | -- | -- | -- | -- | -- | -- | not computed (queue does not run subband_similarity_metrics) | -- | unknown | not yet evaluated -- no metadata.json found |
