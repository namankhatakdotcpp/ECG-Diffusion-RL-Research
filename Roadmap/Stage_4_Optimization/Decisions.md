# Stage 4 -- Decisions

## RL base architecture: S3-001

S3-001 (variant "baseline" -- architecturally identical to the pre-Stage-3
model, per `run_stage3_queue.VARIANT_BY_RUN_ID`) is the frozen base for RL
fine-tuning.

Justification, from `Reports/Stage3_Comparison.md`'s actual measured values
(not asserted): highest generated-data accuracy (0.4400) and highest
generated-data macro-F1 (0.3421) of the five evaluated candidates
(S3-002: 0.3967/0.2814, S3-003: 0.42/0.2406, S3-004: 0.3967/0.2863,
S3-005: 0.39/0.2059).

Caveat: this is a decision by the two primary metrics only, not the
full weighted "Overall" score from Stage3_Comparison's Step 3 (that
weighting scheme was explicitly left undefined pending a normalization/
weighting decision -- see `Stage_3_Architecture_Improvements/Reports/`).
If Cosine/Mahalanobis/Hausdorff/Bhattacharyya/subband evidence is later
weighted formally and points to a different candidate, this decision
should be revisited before committing further GPU time to S3-001's RL
fine-tune.

## Training-time reward classifier vs. evaluation classifier: kept separate

`step06_reward_function.py`'s `DiagnosticUtilityReward` continues to use
`tstr_classifier.pt` (a `Simple1DCNN` from `step05_baseline_eval.py`) as
the in-the-loop RL reward signal -- not the Mentor Classifier used in
`mentor_eval/classification_validation.py`.

Reasoning: if the same classifier were both the reward signal and the
evaluation instrument, the RL policy could learn to satisfy that
classifier's specific decision boundary without genuinely improving
disease-discriminative morphology -- indistinguishable from reward
hacking until checked against an independent oracle. The Mentor
Classifier is that independent oracle and must stay independent.

Hard constraint: Phase 4 (post-RL re-evaluation) must use ONLY the
Mentor Classifier / `classification_validation.py`'s pipeline for any
reported accuracy/F1/AUC numbers. `tstr_classifier.pt`'s scores are a
training signal, never a reported result.

## reward_a3 (Stage 3 subband finding -> RL reward term): open

Confirmed NOT a duplicate of any existing `ClinicalReward` component:
`MorphologyReward` measures discrete PR/QRS/QT interval durations (ms,
via neurokit2 peak detection) -- not waveform energy or shape.
`RealismReward` is a generic, class-agnostic, Lead-II-only PCA check --
not frequency-decomposed or class-conditional. Neither captures what
A3-subband energy divergence (Stage 3's dominant failure mode) measures.

Decision needed, not yet made: add `reward_a3` as a genuine 5th weighted
component (alongside morph/hrv/real/diag), and re-derive the weight
split across 5 (or 6, if `reward_regularization` is also added
separately) terms -- the previously-proposed 0.55/0.25/0.10/0.10 split
was designed for 4 terms and does not carry over automatically. Requires
sign-off before Phase 2 implementation.
