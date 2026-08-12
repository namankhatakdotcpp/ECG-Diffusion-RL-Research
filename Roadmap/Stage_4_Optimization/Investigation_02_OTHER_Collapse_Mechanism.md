# Investigation 02: OTHER Class Permanent r_diag Collapse — Log-Based Analysis

**Date:** 2026-07-21
**Status:** Log-analysis phase complete. Root trigger (what specifically made
the iteration-383 rollout/update different) was never determined — PPO
instability was ruled out as the cause (see "Rejected / Unsupported
Hypotheses" below), but no positive mechanism was confirmed. The only
remaining path to resolving it is the instrumented rerun (per-rollout
logging, "Future Work" item 3 below), which has not been run.

**Deprioritized (2026-07-21, follow-up), not abandoned:** this root-cause
question is explicitly deprioritized in favor of RL reward-design work,
now that Investigation_03 has established which generation-quality signal
to trust for checkpoint evaluation (see `Decisions.md`, "Gate CLOSED:
`diffusion_rl_selected_UNVALIDATED.pt` rejection hardens..."). This is a
conscious prioritization call, not an oversight — the instrumented rerun
remains available as future work if the mechanism becomes load-bearing for
the paper (e.g. if reward-design work needs to explain why a specific
class collapses to justify a reward-shaping choice).

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

## Search for an OTHER-Capable Evaluator — No Ready-to-Run Tool Exists

Searched the full evaluation surface (`mentor_eval/` and root `step*.py`
scripts) for anything that could score OTHER-conditioned generated samples,
verified directly from source rather than assumed:

- **`classification_validation.py`** (mentor pipeline): 4-class taxonomy only
  (Normal/STEMI/NSTEMI/AFIB). No OTHER. Already established above.
- **`subband_similarity_metrics.py`**: also restricted to the mentor-derived
  taxonomy -- `BOX_CLASSES = ["Normal", "STEMI", "NSTEMI"]` (line 54),
  explicitly commented "AFIB excluded - no trained model class." This tool
  does **not** bypass the taxonomy gap and cannot evaluate OTHER either.
- **`step05_baseline_eval.py`'s TRTR classifier**: this is the one genuinely
  native-taxonomy tool -- trained and evaluated over the diffusion model's own
  6-class list (`class_names.json`: NORM/MI/STTC/CD/HYP/OTHER), confirmed by
  its own code handling an explicit `"OTHER" in name_to_idx` case.
  `analyze_stage4_hyp_other.py` (lines 122-129) itself already documents this
  as the correct tool for direct HYP/OTHER evaluation, in preference to the
  mentor pipeline.

**Gap identified:** `step05_baseline_eval.py`'s `main()` takes **no `--ckpt`
argument** -- it is hardcoded to whatever checkpoint `load_config()` resolves
(the frozen baseline), unlike `classification_validation.py`, which already
supports `--ckpt`. There is currently no way to point the TRTR classifier's
generation-and-eval path at `rl_ckpt_iter0370.pt` or `rl_ckpt_iter0380.pt`
specifically without a code change.

**Conclusion:** An OTHER-capable evaluator exists in principle (the TRTR
classifier's native 6-class taxonomy), but not as a ready-to-run tool for
this comparison. The minimal path forward is a small, well-scoped change --
add a `--ckpt` parameter to `step05_baseline_eval.py`, mirroring
`classification_validation.py`'s existing pattern -- not a new experiment
design and not an instrumented rerun.

## TRTR Multi-Seed Checkpoint Comparison (370 vs. 380) — OTHER-Specific Result

With `--ckpt` support added to `step05_baseline_eval.py`, ran its native
6-class TRTR/TSTR path against both `rl_ckpt_iter0370.pt` and
`rl_ckpt_iter0380.pt`, 3 seeds each (42, 123, 456), matching the multi-seed
rep structure used elsewhere in Stage 4. Source: `baseline_metrics.json`
under `outputs/results/checkpoint_370_trtr_eval/` and
`outputs/results/checkpoint_380_trtr_eval/`, read in full and verified
directly (not derived from a summarized intermediate).

**Per-seed TSTR macro F1 and OTHER-class F1:**

| Seed | Ckpt 370 macro | Ckpt 380 macro | Ckpt 370 OTHER | Ckpt 380 OTHER |
|---|---|---|---|---|
| 42  | 0.355341 | 0.367020 | 0.235955 | 0.275862 |
| 123 | 0.338927 | 0.341882 | 0.236842 | 0.282051 |
| 456 | 0.344609 | 0.347025 | 0.387097 | 0.387097 |

**Per-seed deltas (380 − 370):**

| Seed | Macro Δ | OTHER Δ |
|---|---|---|
| 42  | +0.011679 | +0.039907 |
| 123 | +0.002955 | +0.045209 |
| 456 | +0.002416 | +0.000000 |

- Macro Δ: mean = 0.005683, population std = 0.004245 → range [0.001438, 0.009928]
- OTHER Δ: mean = 0.028372, population std = 0.020178 → range [0.008194, 0.048550]

**Result: the pre-registered separation criterion is not met.** OTHER's
delta is larger on average than the general macro delta, but the two
ranges overlap (macro's upper bound 0.009928 exceeds OTHER's lower bound
0.008194). At n=3 seeds, this does not establish that OTHER declined by
more than ordinary run-to-run noise already present across all classes --
and under TSTR, OTHER did not decline at all (it improved in every seed).
This is consistent with the documented run-to-run variance for minority
classes elsewhere in this project (e.g. NSTEMI's rep0/rep1/rep2 spread of
0.000/0.378/0.131 at iteration 1000). This is a real answer under TSTR
(see Final Conclusion below), not a data shortage requiring more seeds to
resolve; additional seeds would only be useful as optional confirmation.

One data point worth flagging without over-reading it: at seed 456,
OTHER's F1 is bit-for-bit identical between checkpoints
(`0.3870967741935484` in both files' `raw_seeds[2].tstr_per_class_f1[5]`),
while every other class's F1 for that same seed shifts between checkpoints
(e.g. NORM: 0.456731 -> 0.486445). This could mean OTHER's generation
quality was already stable/unaffected by whatever changed between 370 and
380 for that seed -- or it could be coincidence given only 3 seeds. It does
not change the overlap conclusion above.

**What this does and does not resolve:** Unlike the earlier
`classification_validation.py` comparison, this result *is* OTHER-specific
(no taxonomy exclusion). Under TSTR specifically, it rules out an
OTHER-specific decline between these checkpoints at these seeds -- OTHER
improved in all 3 seeds and its delta range is not separable from the
general macro delta range. It does not, however, reconcile this with
`classification_validation.py`'s decline on a different metric; see Final
Conclusion below for how these two results are treated jointly.

## Future Work
1. **More seeds for the 370/380 TRTR comparison** (optional confirmation,
   not a blocker -- see Final Conclusion below, which treats this
   investigation as closed under TSTR). At n=3, the OTHER delta range and
   the general macro delta range overlap; additional seeds could firm up
   that non-separation but are not required to close the OTHER-specific
   question this investigation was scoped to answer.
2. **New investigation**: reconcile why `classification_validation.py`
   shows generated-macro decline (370 -> 380) while TSTR shows improvement
   over the same interval. This is a distinct question from the original
   OTHER-collapse-mechanism investigation (see Final Conclusion below) and
   should be tracked separately rather than as follow-up here.
3. **Instrumented rerun**: add per-rollout logging (sample IDs, classifier
   logits/probabilities, reward components before aggregation) to identify
   the upstream trigger of the iteration-383 transition, if either of the
   above leaves it unresolved.

## Note on Decisions.md
This is an investigation record, not a project decision -- it documents what
was tested, what was learned, and what remains unresolved. A concise
conclusion should only be promoted into `Decisions.md` once either the
checkpoint comparison or an instrumented rerun confirms the underlying
mechanism.

## Metric Interpretation

This investigation now includes two independent evaluation paradigms,
measuring different properties of generated OTHER samples:

- **`classification_validation.py`** (train on real, test on generated):
  measures whether generated samples preserve class identity recognizable
  by a real-trained classifier. Prior result in this doc (commit 0d42ca9):
  generated macro F1 declined from checkpoint 370 (0.3339) to checkpoint
  380 (0.2149).
- **`step05_baseline_eval.py` TSTR path** (train on generated, test on
  real): measures whether generated samples are useful as training data.
  This investigation's result (above): both macro and OTHER TSTR improved
  slightly from checkpoint 370 to 380.

The TSTR evaluation did not reproduce the deterioration observed by
`classification_validation.py`. Because the two evaluation pipelines
measure different properties (generated-to-real transferability vs.
real-trained recognition of generated samples), these results should not be
interpreted as confirming or refuting one another. Disagreement between
them does not mean either metric is wrong -- they quantify different
aspects of generation quality, which can evolve differently over the course
of RL fine-tuning.

## Final Conclusion (OTHER-Specific Question)

**Established:**
- Native OTHER-capable evaluation was implemented (`--ckpt`/`--out-dir`
  patch to `step05_baseline_eval.py`, commit 7eb2469 and follow-up).
- Checkpoints 370 and 380 were evaluated successfully under the same
  3-seed protocol (seeds 42/123/456) used elsewhere in this investigation.
- Under TSTR specifically: OTHER's F1 improved from checkpoint 370 to 380
  in all 3 seeds (never declined); macro TSTR also improved in all 3 seeds.
- The pre-registered statistical-separation criterion (delta ranges must
  not overlap to claim an OTHER-specific effect) is **not met** -- the
  OTHER delta range ([0.008194, 0.048550]) and macro delta range
  ([0.001438, 0.009928]) overlap.

**Interpretation (do not overstate):** No evidence of OTHER-specific
collapse was observed **under the TSTR evaluation specifically**, on these
two checkpoints, at these three seeds. This is not a general or unqualified
claim that "OTHER does not collapse" -- the `classification_validation.py`
result showing decline on a different metric remains unreconciled, not
resolved. This finding narrows the question rather than closing it: the
original concern (an OTHER-specific failure) is unconfirmed under this
metric, not disproven overall.

**Remaining open question (new, not a continuation of Investigation_02):**
Why does the real-trained classifier (`classification_validation.py`) show
degradation from checkpoint 370 to 380 while TSTR remains stable/improves
over the same interval? This is a distinct research question about how
different generation-quality properties diverge during RL fine-tuning, and
should be tracked as a new, separate investigation rather than extending
this one indefinitely.

**Project status:** Investigation_02 (the OTHER-collapse-mechanism
investigation) is considered complete as of this entry. The remaining work
is a new hypothesis-level question, not an unfinished check within this
investigation.
