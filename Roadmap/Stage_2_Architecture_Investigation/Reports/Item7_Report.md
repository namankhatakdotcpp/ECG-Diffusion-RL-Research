# Item 7 (Class-Embedding Evolution Across Training) -- Report

## Executive summary

**BLOCKED (permanent).** No verdict -- VERIFIED, FALSIFIED, or
otherwise -- is possible for this training run. The artifact this item
requires (multiple per-epoch checkpoints spanning the training
trajectory) does not exist anywhere, confirmed by direct filesystem and
archive search, not assumed. This is reported as a genuine, permanent
gap in the evidence base, not a soft "inconclusive."

## Architectural question (as specified)

Does `class_emb.weight`'s pairwise inter-class distance
(`||class_emb.weight[c] - class_emb.weight[c']||`) grow, shrink, or stay
flat over the course of training -- direct evidence for or against the
embedding space itself learning to differentiate, independent of
everything measured downstream of it?

## What was attempted

Per the pre-registration (`Item7_PreRegistration.md`), the plan was to
compute pairwise `class_emb.weight` distances at multiple saved
per-epoch checkpoints from the training run that produced the current
`outputs/models/diffusion_best.pt`, and observe the trend across
training time.

## Verification -- why this could not proceed (Phase D discipline: confirm, don't assume)

Confirmed by direct filesystem and archive search:
```
find <repo> -iname "diffusion_ckpt_ep*.pt"          -> no results
tar -tzf stage1_results.tar.gz | grep "ckpt_ep"      -> no results
```
Only the single final/best checkpoint (`diffusion_best.pt`, epoch 200)
exists. **Root cause, confirmed, not inferred:** periodic checkpoints
ARE saved by design every `save_every=25` epochs
(`step04_transformer_diffusion.py:950`), but a retention policy
(`KEEP_LAST_N_CHECKPOINTS=2`, `step04_transformer_diffusion.py:700,958-961`)
prunes all but the most recent 2 as training progresses. Confirmed via
`git merge-base --is-ancestor f78c6c2 01b55091` (true) that the
retention-policy commit is an ancestor of this training run's own base
commit -- the pruning was active throughout the run that produced the
checkpoint currently on disk. This is the same root cause independently
confirmed for Item 4's epoch-25 comparison gap.

**Item 7, as specified, requires the intermediate weight VALUES at
multiple past epochs -- data that was overwritten, not merely unlabeled
or hard to locate.** No reconstruction from the single surviving
checkpoint is possible.

## Results

None -- no experiment ran. There is no partial or degraded result to
report; the blocker is total.

## Interpretation

Per this project's standing discipline (never assume, never substitute
memory for source), forcing a VERIFIED or FALSIFIED verdict here would
require evidence that does not exist. **Status: BLOCKED (permanent
limitation for this training run)** -- a third, explicit outcome, not a
forced binary choice.

## Cross-validation

Not applicable -- no data exists to cross-validate against other items.
Item 4's training-time gradient analysis measures the *current, final*
checkpoint only, not the embedding's trajectory across training, and
does not substitute for this item.

## A related-but-different analysis considered, and rejected as a substitute

`stage1_results.tar.gz` contains separate final/best checkpoints from
Item 1's own Experiment 2 (dataset-scaling: sizes 380, 1000, 2500,
5000, 10000, full) -- independently-trained runs at different DATA
AMOUNTS, not different TRAINING TIMES of one run. Comparing
`class_emb.weight` pairwise distances across these would answer "does
more training data change embedding differentiation," a genuinely
different question than "does embedding differentiation change over
the course of ONE training run." Not substituted here without explicit
sign-off -- silently answering a different question under Item 7's
name would misrepresent what was actually tested.

## Limitations

Total: no measurement of any kind was possible. This is not a precision
or sample-size limitation like other items' caveats -- it is a complete
absence of the required artifact.

## Decision

**BLOCKED (permanent).** No verdict possible for the current training
run. The dataset-scaling-checkpoint alternative above is flagged as a
possible FUTURE item (answering a different question), not pursued
here.

## Next steps

If a future Stage 3 retrain occurs for any reason, preserving periodic
checkpoints (a configuration change, not a new experiment) would make
this item answerable retroactively. Not a pending follow-up task for
Stage 2 itself.

## Artifacts

None produced. See `Item7_PreRegistration.md` for the full evidentiary
chain (filesystem search commands, git-ancestry confirmation) behind
this BLOCKED status.
