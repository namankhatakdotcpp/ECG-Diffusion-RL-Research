# Stage 3 -- Cross-Candidate Comparison

All candidates evaluated via the identical `mentor_eval.classification_validation` pipeline against the frozen baseline manifest (checksum `unavailable`, commit `unavailable`, source: ERROR -- baseline manifest missing, NOT computed live (see error banner above)). Optimizer-config column reflects whether the run's training commit predates the gain-parameter weight-decay fix (`0294330`/`432395c`).

**Baseline generated-data classifier metrics not found** (expected at `/Users/a7206035376/Desktop/HCL_Internship/ECG/outputs/mentor_review/classification_validation/classifier_generated_eval.json`) -- Delta column cannot be computed.

| Candidate | Variant | Params (M) | Generated Accuracy | Generated Macro-F1 | Generated Macro-AUC | Real-data Accuracy | Delta vs. Baseline (accuracy) | Similarity Cosine (mean) | Similarity Mahalanobis (mean) | Sharma/Subband Metrics | Training Time | Optimizer Config | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S3-001 | baseline | 8.43 | -- | -- | -- | -- | -- | -- | -- | not computed (queue does not run subband_similarity_metrics) | -- | unknown | not yet evaluated -- no metadata.json found |
| S3-002 | layerscale | 8.43 | -- | -- | -- | -- | -- | -- | -- | not computed (queue does not run subband_similarity_metrics) | -- | unknown | not yet evaluated -- no metadata.json found |
| S3-003 | late_gain | 8.43 | -- | -- | -- | -- | -- | -- | -- | not computed (queue does not run subband_similarity_metrics) | -- | unknown | not yet evaluated -- no metadata.json found |
| S3-004 | residual_scaling | 8.43 | -- | -- | -- | -- | -- | -- | -- | not computed (queue does not run subband_similarity_metrics) | -- | unknown | not yet evaluated -- no metadata.json found |
| S3-005 | hybrid | 8.43 | -- | -- | -- | -- | -- | -- | -- | not computed (queue does not run subband_similarity_metrics) | -- | unknown | not yet evaluated -- no metadata.json found |
| S3-006 | final_norm_gain | 8.43 | -- | -- | -- | -- | -- | -- | -- | not computed (queue does not run subband_similarity_metrics) | -- | unknown | not yet evaluated -- no metadata.json found |
