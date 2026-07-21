# Investigation 03: TSTR vs. classification_validation.py Metric Divergence

**Date:** 2026-07-21
**Status:** Step 1 (pre-registration) and Step 2 (read-only investigation)
complete. Stopping here per pre-registered stop condition — no fix proposed,
no new evaluation run, no code modified.

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

### Finding 1 — Open Provenance Limitation: which checkpoint set was used cannot be independently reconstructed

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

**Leading item — the investigation's premise itself carries an open
provenance limitation (Finding 1).** Before weighing why the two metrics
disagree, it's worth being explicit that this machine holds two genuinely
different checkpoints under the same filenames (`rl_ckpt_iter0370.pt`/
`0380.pt` in `stage4_finetune_v1/` vs. `experimentA_reward_replay_cyclic/`),
and neither `0d42ca9`'s nor `13080ee`'s result records which one it
actually loaded. A read-only cross-check against both runs' own training
logs (above) confirms both checkpoint files are genuine and correctly
attributed to their claimed runs, but cannot determine which one either
past script invocation used — that remains open on both sides. This is
**not evidence that either result used the wrong checkpoint**; it is a
limit on how much confidence either result can carry until closed, and it
means the two findings below should be read as "assuming both results
describe the same `stage4_finetune_v1` transition" rather than as fully
settled.

With that limitation stated, two of the three pre-registered candidates
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
as opposed to just differing magnitude or noise. Confirming that would
require a matched-protocol rerun (same checkpoint, verified path; matched
sample count) — out of scope for this read-only pass per the stop
condition below.

## Stop Condition

Stopping here per instruction. Not proposing a fix, not running any new
evaluation, not modifying either eval script. Findings above are read-only
source/artifact inspection only.
