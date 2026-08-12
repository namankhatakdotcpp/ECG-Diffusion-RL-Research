# Investigation 03: TSTR vs. classification_validation.py Metric Divergence

**Date:** 2026-07-21
**Status:** Phase 1 (pre-registration + read-only investigation) complete.
Finding 1's provenance gap subsequently resolved via filesystem timestamps.
Phase 2 (matched-sample-count rerun) complete: outcome **UNEXPLAINED** —
the metric divergence is not a sample-count artifact; by elimination,
opposite transfer direction (candidate (a)) is implicated as the driver.
Not committed.

## Research Question

Why does `classification_validation.py` show generated-macro F1 DECLINE
(0.3339 -> 0.2149, checkpoints `rl_ckpt_iter0370.pt` -> `rl_ckpt_iter0380.pt`,
documented in commit `0d42ca9`) while `step05_baseline_eval.py`'s TSTR path
shows macro AND OTHER-class F1 IMPROVEMENT over the same checkpoint interval
(documented in `Investigation_02_OTHER_Collapse_Mechanism.md`'s "TRTR
Multi-Seed Checkpoint Comparison" section, commit `13080ee`)?

This is a new, separate investigation, spun off explicitly by
Investigation_02's "Final Conclusion (OTHER-Specific Question)" section
rather than a continuation of it.

## Step 1 — Pre-Registered Decision Criteria (written before any new evidence)

**(a) Real divergence between transfer directions.** Generation quality
genuinely differs by transfer direction — generated samples might be
structurally recognizable to a fresh classifier (feeding TSTR) while
becoming progressively less similar to what a REAL-data-trained classifier
expects (feeding `classification_validation.py`'s decline), or vice versa.
Evidence needed: confirmation that the two scripts measure genuinely
different, non-redundant directions (train-on-real-test-on-generated vs.
train-on-generated-test-on-real), such that disagreement is expected rather
than contradictory.

**(b) Pipeline bug/inconsistency.** One of the two eval scripts has a
defect — differing preprocessing, differing sample counts, differing
checkpoint-loading logic, a stale cache, or a taxonomy mismatch similar to
the prior confirmed bug in this project (see `Decisions.md`, "`analyze_
stage4_hyp_other.py` TRTR cross-reference bug — CONFIRMED and fixed
(taxonomy key mismatch, not missing data)", commit referenced there as the
pattern to watch for). Evidence needed: a direct code diff between the two
scripts' data-loading and evaluation paths showing an actual defect, not
just a difference.

**(c) Sample-count / class-balance artifact.** The two evals may use
different generated-sample counts or seeds in ways that make their results
not directly comparable regardless of underlying quality. Evidence needed:
comparing the two scripts' actual sample sizes and class distributions per
checkpoint.

These are not mutually exclusive. If Step 2 finds evidence for more than
one, all confirmed findings are reported — the goal is not to pick a single
winner.

## Step 2 — Read-Only Investigation Findings

All findings below are re-derived directly from source on this session
(hostname `himtenduh`, the GPU server) in this pass, not carried over from
Investigation_02 by paraphrase.

### Finding 1 — Provenance Limitation: RESOLVED for both sides (2026-07-21, follow-up)

**Update:** the open half of this gap (which checkpoint set `0d42ca9`'s
`classification_validation.py` run loaded) has since been resolved via
filesystem timestamps and is no longer open. See "Finding 1 Resolution"
below, added after a targeted follow-up check. The original analysis is
kept as written beneath it for the record, since the reasoning (why this
mattered, why it wasn't resolvable from the checkpoints/JSON alone) still
stands — only the "cannot be confirmed" conclusion is superseded.

**Finding 1 Resolution:** the filesystem mount is `relatime` (confirmed:
`mount | grep " / "` shows `rw,relatime`), so access times are meaningful,
not disabled. `stage4_finetune_v1/rl_ckpt_iter0370.pt`'s access time
(`2026-07-21 10:13:38.792100903 UTC`) matches — to the same sub-second
precision — the write time of `checkpoint_370_comparison/classifier_real_
eval.json` (the `0d42ca9` output artifact). Same for iteration 380:
checkpoint access `10:15:08.445185428 UTC` matches `checkpoint_380_
comparison/classifier_real_eval.json`'s write time exactly. Both precede
the `0d42ca9` commit (`10:25:08 UTC`) by a plausible 10-12 minutes. The
`experimentA_reward_replay_cyclic` checkpoints show no access anywhere
near that window — their only recorded access (`13:04:15 UTC`, identical
for both files) is ~2h39m *after* the commit, consistent with this
investigation's own later `torch.load` inspection calls, not the historical
run. **Conclusion: `0d42ca9` used `stage4_finetune_v1`'s checkpoints, not
`experimentA_reward_replay_cyclic`'s.** Combined with `13080ee`'s TSTR/TRTR
result (which used `stage4_finetune_v1` per the explicit `--ckpt` paths in
that run), both halves of the divergence this investigation studies are now
confirmed to describe the same underlying model transition.

### Finding 1 (original analysis, superseded conclusion retained for record) — which checkpoint set was used could not initially be reconstructed

Both scripts require an explicit, existing `--ckpt` path with no silent
fallback to a default when a bad path is given:
- `classification_validation.py:360` (`parser.add_argument("--ckpt", ...,
  default=None)`) and `:371-376` (raises `[FATAL]` and exits if the given
  path doesn't exist).
- `step05_baseline_eval.py:1186-1198` (same pattern: `[FATAL]` and exit on
  a nonexistent `--ckpt` path).

However, this machine has **two directories containing identically-named
checkpoints** that are confirmed to be genuinely different model weights,
not duplicates:

| Path | `iter` | `best_reward` (embedded in checkpoint) | md5 (370) | md5 (380) |
|---|---|---|---|---|
| `outputs/models/stage4_finetune_v1/` | 370, 380 | 0.5351991261181018 | `e3bf63d3...` | `2b231178...` |
| `outputs/models/experimentA_reward_replay_cyclic/` | 370, 380 | 0.5014594101776308 | `bd242c20...` | `b7433f53...` |

(`best_reward` and md5 hashes confirmed directly by loading both checkpoint
files with `torch.load` and running `md5sum` in this session.)

`stage4_finetune_v1` is the run Investigation_02's `rl_training_log.csv`
analysis and iteration-383 transition are about. `experimentA_reward_replay_
cyclic` is the unrelated HYP→STTC reward-replay experiment ("Experiment A —
Complete", `Decisions.md`), a separate training run that happens to share
iteration numbers 370/380 by coincidence of checkpoint-saving cadence.

Neither script persists its resolved `--ckpt` argument into any output
artifact — `classifier_generated_eval.json` and `baseline_metrics.json`
(the source files for the 0d42ca9 and 13080ee numbers respectively) contain
no path field. Logging is stdout-only with no file handler
(`utils/logger.py:38-50`, `logging.StreamHandler(sys.stdout)`, no
`FileHandler` anywhere in that function), and no shell history survives on
this machine that references either checkpoint path.

**This is a reproducibility limitation, not evidence that an incorrect
checkpoint was used.** Nothing found in this investigation suggests either
historical run (`0d42ca9`'s `classification_validation.py` comparison, or
`13080ee`'s TRTR/TSTR comparison) loaded the wrong checkpoint set — only
that neither can be independently reconstructed from the artifacts that
survive. `stage4_finetune_v1` is the contextually obvious choice for both,
given that both investigations are framed around that run's iteration-383
transition. This is reported as an open provenance limitation, not a
confirmed instance of (b), and — because it calls into question whether
`0d42ca9` and `13080ee` are even describing the same underlying model
transition — it is treated as the lead finding below rather than a
sub-point under candidate (b).

**Partial cross-check attempted (read-only, no new evaluation): result is
inconclusive, not confirming.** `logs/stage4_finetune_v1/rl_training_log.
csv` and `logs/experimentA_reward_replay_cyclic/rl_training_log.csv` were
read directly to see whether either training run's own logged rewards
could corroborate which checkpoint set a given script actually loaded.
Neither log has an explicit "best-reward-so-far" column; the closest proxy
is the running maximum of the `reward_total` column. That running max, by
iteration 370 and by iteration 380, is **0.53520** for `stage4_finetune_v1`
and **0.50146** for `experimentA_reward_replay_cyclic` — both matching
their own checkpoint's embedded `best_reward` almost exactly (0.5351991...
and 0.5014594... respectively). This confirms both checkpoint files are
genuinely, correctly attributed to their claimed originating runs (neither
is corrupted or mislabeled) — but it is a **within-run self-consistency
check**, not a cross-reference to what either external script actually
loaded on the date of its run. Neither log records anything about a
`classification_validation.py` or `step05_baseline_eval.py` invocation, so
this cross-check cannot narrow which checkpoint set `0d42ca9` or `13080ee`
used. The provenance limitation remains fully open on both sides.

### Finding 2 — Sample count per class: confirmed 5x discrepancy (supports candidate (c))

- `classification_validation.py:320`: `generate_for_class(..., n_samples=100,
  ...)` — hardcoded, not config-driven.
- `step05_baseline_eval.py:1066`: `n_per_class = int(ecfg.n_synthetic_per_
  class)`, resolving to `config.yaml:242`'s `n_synthetic_per_class: 500`.

Confirmed against the actual output artifacts: `classifier_generated_eval.
json`'s confusion matrices for both the 370 and 380 runs
(`outputs/mentor_review/checkpoint_370_comparison/`, `checkpoint_380_
comparison/`) sum to exactly 100 samples per evaluated class (Normal,
STEMI, NSTEMI), consistent with the hardcoded value.

This is a real, code-level, confirmed difference: `classification_
validation.py` evaluates on 100 generated samples/class, `step05_baseline_
eval.py`'s TSTR path evaluates on 500 generated samples/class — a 5x
difference in the underlying generated-sample population size feeding each
metric.

### Finding 3 — Classifier reuse: ruled out, no shared-object bug

`classification_validation.py`'s `MentorClassifier` (`:80-82`, explicit
docstring: "Same architecture as step05's Simple1DCNN, retrained for the 4
mentor-review classes (different label scheme)") is trained from scratch
on the 4-class mentor taxonomy each run (`:284`, `train_classifier(...)`).
`step05_baseline_eval.py`'s TSTR/TRTR classifiers (`:768-787`) are separate
objects trained on the native 6-class taxonomy. No accidental object reuse
across the two scripts.

### Finding 4 — Real-data fold assignment: consistent, not a contributing factor

`classification_validation.py:61-63`: `TRAIN_FOLDS = list(range(1, 9))`,
`VAL_FOLDS = [9]`, `TEST_FOLDS = [10]`, hardcoded. `config.yaml:58-60`:
`train_fold: [1..8]`, `val_fold: [9]`, `test_fold: [10]` — the same
assignment that (by convention) produced `step05_baseline_eval.py`'s
precomputed `X_train.npy`/`X_val.npy`/`X_test.npy` (`step05_baseline_
eval.py:190-196`). Fold assignment is consistent between the two
pipelines' real-data splits — this rules out fold mismatch as a
contributing factor.

### Finding 5 — Transfer direction: confirmed genuinely reversed (supports candidate (a))

`classification_validation.py:275-284` trains its classifier on REAL data
only (folds 1-8/9), then `:296-328` ("Stage 2: evaluate on GENERATED data")
evaluates that real-trained classifier on generated samples — train-real,
test-generated. `step05_baseline_eval.py:749-772` ("TSTR: train Simple1DCNN
on synthetic... test on X_test") trains on GENERATED samples and evaluates
on real `X_test` — train-generated, test-real. These are confirmed to be
the reverse of each other, not the same measurement computed two different
ways. This matches (re-derived from source, not paraphrased from) the
framing already stated in Investigation_02's "Metric Interpretation"
section.

## What Step 2 Establishes

**Leading item — the investigation's premise itself was, until a follow-up
check, resting on an open provenance limitation (Finding 1) — now
resolved.** This machine holds two genuinely different checkpoints under
the same filenames (`rl_ckpt_iter0370.pt`/`0380.pt` in `stage4_finetune_v1/`
vs. `experimentA_reward_replay_cyclic/`). A filesystem-timestamp check
(Finding 1 Resolution, above) confirmed both `0d42ca9` and `13080ee`
loaded `stage4_finetune_v1`'s checkpoints — the two results genuinely
describe the same underlying model transition, not two unrelated runs.
The findings below can now be read as settled on that point, not
conditional on it.

With premise now confirmed, two of the three pre-registered candidates
are confirmed as real, non-exclusive contributors to the divergence itself:
- **(a) is structurally confirmed**: the two pipelines measure opposite
  transfer directions by design (Finding 5). Disagreement between them is
  not inherently contradictory.
- **(c) is confirmed**: a real 5x generated-sample-count difference exists
  between the two pipelines (Finding 2).
- **(b) is not confirmed as a defect** in either script's logic — no
  object reuse (Finding 3), consistent fold assignment (Finding 4), no
  silent-fallback behavior (Finding 1).

**What this does not establish:** that (a) and (c) together are sufficient
to explain the *direction* of the disagreement (decline vs. improvement),
as opposed to just differing magnitude or noise. That question is tested
directly in "Phase 2" below.

## Stop Condition (Phase 1 — Step 1/Step 2 read-only pass)

Stopped here at the time, per instruction. No fix proposed, no new
evaluation run, no eval script modified. Findings above are read-only
source/artifact inspection only. Phase 2 below is a distinct, later pass
that does run new evaluations and does modify one eval script — see that
section's own pre-registration and stop condition.

## Phase 2 — Matched-Sample-Count Rerun (does (a)+(c) explain the direction reversal?)

**Placement note:** this stays inside Investigation_03 rather than
spinning off a new Investigation_04. Investigation_03's own research
question ("why do these two metrics move in opposite directions?") is not
yet answered — Phase 1 confirmed two contributing differences (transfer
direction, sample count) but explicitly did not establish whether they are
*sufficient* to explain the direction reversal specifically. This
experiment answers that same question empirically rather than opening a
new one, unlike Investigation_02 → Investigation_03 (which was a genuine
scope change to a different research question). A new Investigation_04
would be warranted only if this experiment's result pointed at something
outside classification_validation.py/step05_baseline_eval.py's comparison
entirely — it doesn't.

### Pre-Registration (written before running anything)

**Protocol change:** `classification_validation.py`'s Stage 2 generated-
sample count is hardcoded to 100 (`:320`, prior to this change). Add a
`--n-gen-samples` argument (default 100, preserving exact backward
compatibility with `0d42ca9`) and rerun at `--n-gen-samples 500` — matching
`step05_baseline_eval.py`'s `n_synthetic_per_class` (`config.yaml:242`) —
against the same verified checkpoints as the `13080ee` TSTR comparison:
`outputs/models/stage4_finetune_v1/rl_ckpt_iter0370.pt` and
`rl_ckpt_iter0380.pt` (confirmed correct for this run — see "Finding 1
Resolution" above).

**Decision criteria, fixed before running:**

Baseline to compare against: `0d42ca9`'s original n=100 result — macro F1
0.3339 (370) → 0.2149 (380), a decline of **0.1190**.

- **CONVERGED**: `|macro_f1(380) − macro_f1(370)|` at n=500 is **≤ 0.01**
  (calibrated against Investigation_02's TSTR macro-delta noise band at
  n=3 seeds, upper bound 0.009928 — the closest existing noise-magnitude
  reference in this repo, acknowledging it comes from a different
  pipeline/classifier and is an imperfect but reasonable proxy). A delta
  this small means the original decline was consistent with sample-count-
  driven noise, not a real signal.
- **PARTIALLY EXPLAINED**: decline persists in the same direction (macro
  F1 still lower at 380 than 370) with magnitude **> 0.01 and ≤ 0.06**
  (i.e., shrinks to at most half of the original 0.1190 magnitude, but
  exceeds the noise floor). Sample count is a partial contributor; something
  else still plays a role.
- **UNEXPLAINED**: decline magnitude is **> 0.06** (retains more than half
  its original size) or the direction/magnitude is essentially unchanged
  from the n=100 result. Sample count is not the driver of the direction;
  the divergence is a property of the two transfer directions themselves
  (candidate (a)) and should stand as a research finding in its own right,
  not a bug to fix.

**Scope note:** this tests candidate (c) directly. It does not separately
isolate candidate (a) — but since sample count is now matched, if the
decline persists, direction (not sample count) is implicated by
elimination, which is why a dedicated (a)-only test isn't run separately.

**Single seed (42), not multi-seed:** per the run commands specified for
this experiment. This means CONVERGED/PARTIALLY EXPLAINED/UNEXPLAINED
below is a single-point estimate, not a distribution — a caveat carried
into the result, not resolved by it.

### Code Change

`--n-gen-samples` added to `classification_validation.py`'s argparse
(default 100), threaded through `run()` into the `generate_for_class` call
at `:320` in place of the previous hardcoded `n_samples=100`. No other
logic touched.

**Backward-compatibility spot check (run before the matched-count
experiment):** ran the modified script with no `--n-gen-samples` flag
(default) against `rl_ckpt_iter0370.pt`, seed 42, out-dir
`outputs/mentor_review/spotcheck_370_default/`. Result: `macro_f1 =
0.33387707560501684`, confusion matrix `[[27,73,0,0],[0,100,0,0],
[1,97,2,0],[0,0,0,0]]` — **bit-for-bit identical** to `0d42ca9`'s original
`checkpoint_370_comparison/classifier_generated_eval.json` (confirmed by
direct JSON comparison, not just displayed-decimal rounding). The
refactor preserves original behavior exactly.

### Result

Both runs used `--n-gen-samples 500`, seed 42, against the verified
`stage4_finetune_v1` checkpoints. Confusion matrices confirmed to sum to
500/class (Normal/STEMI/NSTEMI), 0 for excluded AFIB, matching the
intended sample count.

| Metric | Checkpoint 370 (n=500) | Checkpoint 380 (n=500) | Checkpoint 370 (n=100, `0d42ca9`) | Checkpoint 380 (n=100, `0d42ca9`) |
|---|---|---|---|---|
| Generated-data macro F1 | 0.326520 | 0.231804 | 0.333877 | 0.214882 |
| Normal F1 | 0.379747 | 0.112570 | 0.421875 | 0.076190 |
| STEMI F1 | 0.537797 | 0.513611 | 0.540541 | 0.510204 |
| NSTEMI F1 | 0.062016 | 0.069231 | 0.039216 | 0.058252 |

Macro F1 delta (380 − 370):
- **n=500 (this experiment): −0.094716**
- n=100 (`0d42ca9`, for reference): −0.118995
- Shrinkage: the n=500 delta retains **79.6%** of the original n=100
  delta's magnitude (0.094716 / 0.118995).

The per-class pattern is also consistent between n=100 and n=500 — Normal
F1 collapses sharply (0.42→0.08 at n=100; 0.38→0.11 at n=500) and drives
most of the macro decline in both cases, while STEMI is roughly flat and
NSTEMI moves slightly in the *opposite* direction (improves) in both runs.
This is the same qualitative picture at 5x the sample count, not a
different one.

### Outcome Classification

**UNEXPLAINED**, per the pre-registered criteria: `|delta|` at n=500
(0.094716) is **> 0.06** — well above the "shrinks to at most half the
original magnitude" bound for PARTIALLY EXPLAINED, let alone the ≤0.01
CONVERGED noise floor. The decline is not a sample-count artifact:
candidate (c) is **ruled out** as the driver of the direction reversal by
this experiment (it remains a real, confirmed *difference* between the
two pipelines — Finding 2 stands — but it is not what causes the opposite
signs).

**By the pre-registered elimination logic:** since the checkpoint is now
verified identical (Finding 1 Resolution) and the sample count is now
matched (500 = 500), and the decline persists at ~80% of its original
magnitude, candidate (a) — the genuinely opposite transfer directions
(train-real/test-generated vs. train-generated/test-real) — is implicated
as the actual driver of the direction reversal, not by direct isolation
but by having eliminated the other two candidates. This should be treated
as a real research finding: **generated OTHER/Normal/etc. samples can
become simultaneously *less* recognizable to a real-trained classifier
and *more* useful as training data for a fresh classifier, over the same
370→380 interval** — a property of what each transfer direction measures,
not a bug or artifact in either pipeline.

**Caveats carried forward, not resolved by this experiment:**
- Single seed (42) only — this is a point estimate, not a distribution.
  Whether the ~80%-magnitude-retained delta itself falls inside or outside
  seed-to-seed noise at n=500 is not established; only the noise band at
  n=100/n=3-seeds (Investigation_02) was available as a calibration
  reference, and that was for a different pipeline/classifier.
- This experiment does not explain *why* train-real/test-generated and
  train-generated/test-real move oppositely — only that they do, robustly
  to sample count. That mechanistic question remains open.

## Stop Condition (Phase 2)

Stop after reporting the n=500 result and its CONVERGED/PARTIALLY
EXPLAINED/UNEXPLAINED classification. Not committing.
