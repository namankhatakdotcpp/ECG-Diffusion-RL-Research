# Investigation 02: OTHER Class Permanent r_diag Collapse — Log-Based Analysis

**Date:** 2026-07-21
**Status:** Log-analysis phase complete. Root trigger not yet determined; requires future instrumentation.

## Research Question
Why does the OTHER class undergo a permanent diagnostic reward collapse during
Stage 4 PPO fine-tuning, while other classes (including HYP, which also shows
instability, and STTC, which shows superficially similar dips) do not show the
same irreversible pattern?

## Evidence Collected
- Full 1000-iteration `rl_training_log.csv` (`logs/stage4_finetune_v1/`) analyzed
  directly, not summarized from memory.
- 88 near-zero (`r_diag < 0.05`) rows for OTHER across the run.
- 8 recoverable dips (`0.15 <= r_diag <= 0.30`) for OTHER, at iterations 6, 12,
  200, 503, 590, 829, 856, 868.
- Detailed row-by-row comparison of iterations 200, 202, 208 (recovered spike)
  vs. 375, 382, 383, 389 (permanent transition).
- Whole-run search for `ratio > 1.10` across all classes and iterations.
- Whole-run search for the joint condition `ratio > 1.1 AND clip_fraction > 0.1
  AND advantage_mean < -0.1`.
- STTC's two comparable dip events (iterations 785, 871) checked for recovery.

## Established Findings
- OTHER remains in a healthy high-`r_diag` regime (~0.45-0.57) from iteration 6
  through iteration 382.
- Iteration 383 marks a transition into a persistent low-`r_diag` regime that
  does not recover for the remaining 617 iterations of the run.
- At the transition, `r_diag` drops ~95.5% (0.42849 -> 0.01931, iter 382 -> 383),
  while `r_morph` (0.54693 -> 0.60226) and `r_a3` (0.39113 -> 0.46865) both
  *improve* over the same step -- the drop is diagnostic-specific, not a
  general model degradation.
- `ratio > 1.10` occurs exactly once in the entire 1000-iteration run, across
  all six classes: iteration 383, OTHER (`ratio = 1.12430`).
- The joint condition (`ratio > 1.1 AND clip_fraction > 0.1 AND
  advantage_mean < -0.1`) also has exactly one match in the whole run: the same
  iteration-383 row (`clip_fraction = 0.25`, `advantage_mean = -0.15528`).
- OTHER's 8 recoverable dips all show materially different PPO statistics from
  iteration 383 (max `ratio` among them is 1.05036, at iteration 200) and all
  return to the healthy regime within a few iterations.
- Of OTHER's 88 near-zero rows, only iteration 383 itself shows this extreme
  PPO signature; the other 87 are consistent with being steady-state behavior
  of the already-collapsed regime, not independent transition events.
- STTC's comparable dip events (iterations 785, 871) both recover fully within
  a handful of iterations (785 -> 792: 0.00006 -> 0.28097; 871 -> 883: 0.03733
  -> 0.22324) -- STTC does not exhibit a permanent collapse anywhere in the run.

## Rejected / Unsupported Hypotheses
- General PPO instability as a persistent explanation: rejected -- iteration
  200 shows comparable or larger `kl`/`grad_norm` with full recovery two
  iterations later, so update magnitude alone does not explain permanent
  collapse.
- General model degradation at iteration 383: rejected -- `r_morph` and `r_a3`
  both improve at the same step `r_diag` collapses.
- Treating every near-zero `r_diag` row as an independent collapse event:
  rejected -- 87 of OTHER's 88 near-zero rows lack the PPO anomaly and are
  better explained as the collapsed state's steady-state behavior.

## Interpretation (deliberately cautious)
The evidence supports a two-stage interpretation:
1. A transition occurs at iteration 383 in which the diagnostic reward
   collapses while other reward components remain healthy.
2. A unique, unusually large PPO update (the only `ratio > 1.1` event in the
   run) coincides with this transition and may contribute to the persistence
   of the collapsed regime.

The available logs do not determine whether the PPO update initiated the
transition or responded to an upstream change in the diagnostic reward signal
itself. This is an open question, not resolved by the aggregate log.

## Remaining Open Question
What made the iteration-383 rollout/minibatch different from all previous
OTHER updates? The aggregate `rl_training_log.csv` (one row per iteration)
does not contain the information needed to answer this. Missing:
- sampled ECG record IDs for that rollout,
- rollout composition per PPO batch,
- classifier logits/probabilities per sample, before reward aggregation,
- per-sample reward components (the CSV only has per-iteration aggregates).

## What Existing Artifacts Do Not Resolve
Checkpoints `rl_ckpt_iter0370.pt` and `rl_ckpt_iter0380.pt` exist on the GPU
server and bracket the transition, but no per-rollout logs, sampled ECG
records, or classifier logits/probabilities from the iteration-383 update
itself were found (`find outputs -iname "*iter38*" -o -iname "*rollout*"`
returned no rollout-level artifacts).

## Checkpoint Comparison Result (370 vs. 380) — Inconclusive for OTHER

Ran `mentor_eval/classification_validation.py --ckpt <path> --seed 42` against
both `rl_ckpt_iter0370.pt` and `rl_ckpt_iter0380.pt` (the checkpoints bracketing
the iteration-383 transition).

**Result:**

| Metric | Checkpoint 370 | Checkpoint 380 |
|---|---|---|
| Generated-data accuracy | 0.4300 | 0.3567 |
| Generated-data macro F1 | 0.3339 | 0.2149 |
| Normal F1 | 0.4219 | 0.0762 |
| STEMI F1 | 0.5405 | 0.5102 |
| NSTEMI F1 | 0.0392 | 0.0583 |

Real-data classifier metrics (Stage 1) are identical between runs, as expected
(same PTB-XL data, same seed). Generated-data macro F1 declined ~36% relative
between the two checkpoints, driven mainly by a large drop in Normal-class F1.

**Why this does not resolve the OTHER question:** `classification_validation.py`
excludes AFIB from all Stage 2 (generated-data) evaluation, regardless of
checkpoint (script line 15: "AFIB is excluded from stage 2 regardless of
checkpoint"). AFIB is the mentor-facing proxy class that OTHER maps to
(`mentor_eval/class_mapping.py`'s `MENTOR_TO_TRAINED_CLASS`, which has exactly
four keys -- Normal, STEMI, NSTEMI, AFIB -- with AFIB mapped to `None` rather
than to a trained class; OTHER has no entry at all). No OTHER-specific metric
is produced by this script under any invocation.

**Conclusion:** This comparison shows real, if unrelated, evidence that overall
generation quality (Normal/STEMI/NSTEMI) declined between iterations 370 and
380. It provides **no evidence either way** about whether OTHER's generation
quality was already degrading before the iteration-383 transition identified
in the training-log analysis. The original question this comparison was meant
to answer remains open.

**Gap identified:** The current mentor evaluation pipeline has no mechanism to
score OTHER-conditioned generated samples at all, since it operates entirely
in the 4-class mentor-facing taxonomy and structurally excludes AFIB/OTHER.
A genuinely informative checkpoint comparison for this investigation would
require either (a) an OTHER-inclusive evaluation path in the mentor pipeline,
or (b) direct use of the TRTR classifier (6-class native taxonomy, already
used elsewhere in Stage 4) against generated OTHER samples from both
checkpoints.

## Future Work
1. **Checkpoint comparison via TRTR classifier**: the mentor-pipeline
   comparison above was attempted but is structurally unable to evaluate
   OTHER (see above). Re-run the comparison using the TRTR classifier
   (6-class native taxonomy) against OTHER samples generated from
   `rl_ckpt_iter0370.pt` and `rl_ckpt_iter0380.pt`, to check whether
   degradation was already underway before iteration 383 (weakening the
   "sudden transition" framing) or whether both checkpoints look healthy on
   OTHER specifically (supporting iteration 383 as the true onset).
2. **Instrumented rerun**: add per-rollout logging (sample IDs, classifier
   logits/probabilities, reward components before aggregation) to identify
   the upstream trigger of the transition, if the checkpoint comparison above
   does not resolve it.

## Note on Decisions.md
This is an investigation record, not a project decision -- it documents what
was tested, what was learned, and what remains unresolved. A concise
conclusion should only be promoted into `Decisions.md` once either the
checkpoint comparison or an instrumented rerun confirms the underlying
mechanism.
