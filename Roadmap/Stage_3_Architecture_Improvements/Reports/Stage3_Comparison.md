# Stage 3 -- Cross-Candidate Comparison

All candidates evaluated via the identical `mentor_eval.classification_validation` pipeline against the frozen baseline manifest (checksum `16ac1715ac90ecb3db119de5611a3d2fff2cdc6ca82e53fb4d9c9c3a1864819d`, commit `563214a7020b275403f3deda33f21d647406537b`, source: COMPUTED LIVE -- no frozen outputs/mentor_review/baseline_manifest.json found on this machine; run mentor_eval.run_all to produce one). Optimizer-config column reflects whether the run's training commit predates the gain-parameter weight-decay fix (`0294330`/`432395c`).

**Baseline generated-data classifier metrics not found** (expected at `/Users/a7206035376/Desktop/HCL_Internship/ECG/outputs/mentor_review/classification_validation/classifier_generated_eval.json`) -- Delta column cannot be computed.

| Candidate | Variant | Generated Accuracy | Generated Macro-F1 | Generated Macro-AUC | Real-data Accuracy | Delta vs. Baseline (accuracy) | Optimizer Config | Status |
|---|---|---|---|---|---|---|---|---|
| S3-001 | baseline | -- | -- | -- | -- | -- | unknown | not yet evaluated -- no metadata.json found |
| S3-002 | layerscale | -- | -- | -- | -- | -- | unknown | not yet evaluated -- no metadata.json found |
| S3-003 | late_gain | -- | -- | -- | -- | -- | unknown | not yet evaluated -- no metadata.json found |
| S3-004 | residual_scaling | -- | -- | -- | -- | -- | unknown | not yet evaluated -- no metadata.json found |
| S3-005 | hybrid | -- | -- | -- | -- | -- | unknown | not yet evaluated -- no metadata.json found |
