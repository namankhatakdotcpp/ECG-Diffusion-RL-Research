# Stage 3 -- Cross-Candidate Comparison

All candidates evaluated via the identical `mentor_eval.classification_validation` pipeline against the frozen baseline manifest (checksum `16ac1715ac90ecb3db119de5611a3d2fff2cdc6ca82e53fb4d9c9c3a1864819d`, commit `5071ee81a481248321a222abada76b361e779279`, source: COMPUTED LIVE -- no frozen outputs/mentor_review/baseline_manifest.json found on this machine; run mentor_eval.run_all to produce one). Optimizer-config column reflects whether the run's training commit predates the gain-parameter weight-decay fix (`0294330`/`432395c`).

**Baseline generated-data classifier metrics not found** (expected at `/Users/a7206035376/Desktop/HCL_Internship/ECG/outputs/mentor_review/classification_validation/classifier_generated_eval.json`) -- Delta column cannot be computed.

| Candidate | Variant | Params (M) | Generated Accuracy | Generated Macro-F1 | Generated Macro-AUC | Real-data Accuracy | Delta vs. Baseline (accuracy) | Similarity Cosine (mean) | Similarity Mahalanobis (mean) | Sharma/Subband Metrics | Training Time | Optimizer Config | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S3-001 | baseline | 8.43 | -- | -- | -- | -- | -- | -- | -- | not computed (queue does not run subband_similarity_metrics) | -- | unknown | not yet evaluated -- no metadata.json found |
| S3-002 | layerscale | 8.43 | -- | -- | -- | -- | -- | -- | -- | not computed (queue does not run subband_similarity_metrics) | -- | unknown | not yet evaluated -- no metadata.json found |
| S3-003 | late_gain | 8.43 | -- | -- | -- | -- | -- | -- | -- | not computed (queue does not run subband_similarity_metrics) | -- | unknown | not yet evaluated -- no metadata.json found |
| S3-004 | residual_scaling | 8.43 | -- | -- | -- | -- | -- | -- | -- | not computed (queue does not run subband_similarity_metrics) | -- | unknown | not yet evaluated -- no metadata.json found |
| S3-005 | hybrid | 8.43 | -- | -- | -- | -- | -- | -- | -- | not computed (queue does not run subband_similarity_metrics) | -- | unknown | not yet evaluated -- no metadata.json found |
| S3-006 | final_norm_gain | 8.43 | -- | -- | -- | -- | -- | -- | -- | not computed (queue does not run subband_similarity_metrics) | -- | unknown | not yet evaluated -- no metadata.json found |
