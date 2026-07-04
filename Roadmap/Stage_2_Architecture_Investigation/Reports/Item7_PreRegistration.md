# Item 7 -- Class-Embedding Evolution Across Training -- Pre-Registration (BLOCKED)

## Hypothesis, quoted verbatim (`Stage2_Master_Prompt.md:224-228`)

> "7. Class-embedding evolution across training. Pairwise
> `||class_emb.weight[c] - class_emb.weight[c']||` across saved per-epoch
> checkpoints. Shrinking or flat pairwise distances over training is
> direct evidence the embedding space itself isn't differentiating,
> independent of everything downstream."

## Blocker, confirmed by direct evidence, not assumed

**This item requires multiple per-epoch checkpoints from the SAME
training run, spanning the training trajectory. None exist.**

Confirmed by direct filesystem and archive search (this session):
```
find <repo> -iname "diffusion_ckpt_ep*.pt"          -> no results
tar -tzf stage1_results.tar.gz | grep "ckpt_ep"      -> no results
```
Only `outputs/models/diffusion_best.pt` (the single final/best
checkpoint, epoch 200) exists. This is the **same root cause already
confirmed for Item 4**: periodic checkpoints ARE saved by design every
`save_every=25` epochs (`step04_transformer_diffusion.py:950`,
`config.yaml:99` region), but a retention policy prunes all but the
most recent 2 (`KEEP_LAST_N_CHECKPOINTS=2`,
`step04_transformer_diffusion.py:700,958-961`) as training progresses.
Confirmed via `git merge-base --is-ancestor f78c6c2 01b55091` (true) --
the retention-policy commit is an ancestor of the training run's own
base commit (per `Reproducibility_Manifests/exp1_baseline_reproduction.json`'s
ledger-claimed commit), so this pruning was active throughout the run
that produced the checkpoint currently on disk.

**Item 7, as literally specified, cannot be executed with any
artifact that exists or can be reconstructed from the current
checkpoint alone** -- `class_emb.weight` evolution across training
requires the intermediate weight VALUES at multiple past epochs, which
were overwritten/deleted, not merely unlabeled or hard to find.

## Why this is not forced into VERIFIED/FALSIFIED

Per this project's own discipline (never assume, never substitute
memory for source): reporting a VERIFIED or FALSIFIED verdict here
would require evidence that does not exist. Neither outcome is honest.
**Status: BLOCKED (permanent limitation for this training run)** --
a third, explicit outcome, not a forced binary choice, matching Item
4's own "permanent limitation" language and reasoning for the
epoch-25 comparison.

## What WOULD unblock this (stated for completeness, not pursued now)

Only a NEW training run with periodic checkpoints preserved (e.g.
disabling or relaxing `KEEP_LAST_N_CHECKPOINTS`, or copying periodic
checkpoints off-server before pruning) would produce the artifact this
item needs. That is a retraining decision, out of scope for a Tier 0
measurement-only investigation, and not undertaken here.

## A related-but-different analysis considered, and rejected as a substitute

`stage1_results.tar.gz` contains SEPARATE final/best checkpoints from
Item 1's own Experiment 2 (dataset-scaling: sizes 380, 1000, 2500,
5000, 10000, full) -- these are independently-trained runs at
different DATA AMOUNTS, not different TRAINING TIMES of one run.
Comparing `class_emb.weight` pairwise distances across these would
answer "does more training data change embedding differentiation,"
a genuinely different question than "does embedding differentiation
change over the course of ONE training run." **Not substituted here**
without explicit sign-off -- silently answering a different question
under the same item name would misrepresent what was actually tested,
exactly the failure mode this project's discipline exists to prevent.

## Decision

**BLOCKED.** Permanent limitation for the current training run, per
the same evidence already established for Item 4. No VERIFIED/
FALSIFIED verdict is possible. Flagging the dataset-scaling-checkpoint
alternative above as a possible FUTURE item (not Item 7 itself, since
it answers a different question) for consideration in the Stage 2
Decision Report or a later Stage 3 item, not pursued autonomously here
without sign-off.

## Next Steps

Move to Item 8. If a future GPU session captures periodic checkpoints
during any retrain, Item 7 becomes executable retroactively -- this is
recorded as a standing possibility, not a pending follow-up task.
