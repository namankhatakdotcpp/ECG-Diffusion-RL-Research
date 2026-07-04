# Item 4 -- Gradient Competitiveness Probe -- Pre-Registration

Locked before implementation, per this project's standing "no code until
it's in writing" discipline. Item 4's own design questions (loss
definition, batch-averaging, comparison basis, statistical treatment)
are resolved explicitly below, per the master prompt for this item --
none are left for the implementation script to decide silently.

## Hypothesis, quoted verbatim (`Stage2_Master_Prompt.md:205-209`)

> "4. Activation and gradient norms. Gradient norm at class_emb.weight vs.
> every other parameter group, at the checkpoint nearest epoch 200 vs.
> nearest epoch 25 if available. Tests whether the class embedding's
> gradient signal was ever competitive, independent of the forward-pass
> sensitivity probe already run in Stage 1."

## Dependency check

**None.** Item 4 measures backward-pass gradient dynamics at a single
checkpoint -- a completely different signal from Items 1-3's forward-pass
activation/residual measurements. Its result would not change if any of
Items 1-3's numbers were different, and it does not inherit Item 2B's
confound or anything else.

## Permanent limitation (checkpoint availability, resolved before this document)

**Only `outputs/models/diffusion_best.pt` (epoch 200) exists.** No
epoch-25 checkpoint is available: confirmed by direct source read that
one was created (`step04_transformer_diffusion.py:950`, `save_every=25`
per `config.yaml:99` region) and then deleted by the retention policy
(`step04_transformer_diffusion.py:958-961`, `KEEP_LAST_N_CHECKPOINTS=2`),
with that retention logic confirmed active in this run's own commit
lineage (`git merge-base --is-ancestor f78c6c2 01b55091` -- true; the
ledger's claimed base commit for this run is a descendant of the
retention-policy commit).

**If an epoch-25 checkpoint is unexpectedly discovered in archived GPU
storage, the temporal comparison specified in the master roadmap can be
executed as an optional extension. Otherwise, this limitation is
permanent for the current training run because early checkpoints were
overwritten by the retention policy (`save_every=25`,
`KEEP_LAST_N_CHECKPOINTS=2`, confirmed active via git ancestry
`f78c6c2` -> `01b55091`).** This is stated as closed-and-permanent, not
open-and-pending -- it will not appear on Item 3's revisitable
follow-up list.

## Design question 1: loss and forward-pass definition

**Loss:** the exact training loss, confirmed by direct source read, not
assumed -- `loss = F.mse_loss(eps_pred, noise)`
(`step04_transformer_diffusion.py:882`), where `eps_pred = model(x_t, t_diff, batch_cls)`
(line 881) and `x_t, _ = diffusion.q_sample(batch_x, t_diff, noise)`
(line 878), i.e. the standard DDPM noise-prediction MSE.

**Input data:** real training batches (`X_train.npy`, via the same
`ECGDataset`/weighted `DataLoader` construction already used in
`train()`, `step04_transformer_diffusion.py:788-796`) -- NOT the
synthetic `torch.randn` pseudo-`x_t` convention Items 1-3 used. Items
1-3 were probing the model's forward-pass RESPONSE to a given
noise/timestep/class combination, where treating `x_t` as approximately
Gaussian was a reasonable proxy. Item 4 asks about GRADIENT dynamics
*during actual training*, which requires reproducing the real training
step (real `X_train` sample -> `q_sample` -> loss -> backward) as
faithfully as possible, since a synthetic-noise proxy would answer "what
gradient would occur under an unrealistic input," not "was the gradient
ever competitive during training."

**CFG dropout:** applied exactly as training did (per-sample,
`p_uncond=0.10`, `step04_transformer_diffusion.py:868-874`) -- this is
part of the real gradient distribution `class_emb.weight` actually
experienced, not an artifact to eliminate.

**Timesteps -- two complementary designs, stated explicitly:**
1. **Primary (faithful to real training):** per-sample random timesteps,
   `t_diff = torch.randint(0, T, (B,))` exactly as training did
   (`step04_transformer_diffusion.py:876`, `T=1000` per `config.yaml:88`)
   -- this is what real training gradients actually saw, mixed across
   timesteps within every batch. This design answers the headline
   question ("was gradient ever competitive during real training").
2. **Secondary/diagnostic (mirrors Item 1's convention, for the
   per-timestep breakdown the master prompt's own reporting style
   requires):** fixed-timestep variants at `t in {100, 500, 900}`
   (forcing `t_diff = torch.full((B,), t_val)` for the whole batch),
   to test whether competitiveness itself depends on timestep. Reported
   separately from the primary design, not conflated with it.

**Model mode:** `model.train()`, matching real training exactly --
`eval()` would silently answer a different, cleaner-but-counterfactual
question (dropout inactive), when the actual question is about gradient
competitiveness *as training actually experienced it*. Dropout
randomness is why multiple draws are averaged (Design Question 2).

**Hard invariant:** no `optimizer.step()` is ever called -- gradient
computation only, weights never update. Verified by a state-dict
checksum (SHA-256 of the concatenated, flattened parameter tensors)
before and after the full sweep; must match exactly, or the sweep is
aborted (STOP CONDITION, not a soft check).

## Design question 2: gradient-norm stability (batch averaging)

**N=10 independent draws** for the primary (real-random-timestep)
design, and **N=10 draws per fixed timestep** for the secondary design
(3 timesteps x 10 draws = 30 additional forward+backward passes).
**N=10, not Item 1-3's K_DRAWS=20**, stated explicitly rather than
silently reused: `K_DRAWS=20` was calibrated for Item 1's single-sample,
single-noise-vector paired forward pass (a cheap, low-variance
operation); Item 4's draws are full mini-batch (`batch_size=32`)
forward+backward passes against real data with active dropout -- a
substantially heavier and differently-distributed operation. N=10 is
chosen as a reasonable stability/cost balance for this different
quantity, not a reuse of `common/utils.py`'s constant.

Each draw: `zero_grad()`, forward, backward, record `||grad||` per
named-parameter tensor, discard (never step). Pooled: mean and **std**
across the 10 draws per tensor (not just the mean -- a single mean
would hide whether `class_emb`'s gradient is consistently small or just
unlucky on one draw).

**Reproducibility check:** two draws using the identical seed and batch
must produce bit-identical per-tensor gradient norms -- this is the
zero-grad/no-accumulation check (STOP CONDITION if it fails).

## Design question 3: comparison basis

**Primary: percentile rank of `class_emb.weight`'s pooled mean `||grad||`
within the full distribution of all 95 other individual named-parameter
tensors' pooled mean `||grad||`** (per-tensor, NOT bucketed by type).
(Corrected count: the model has 96 named-parameter tensors total -- 8
non-block tensors [`patch_embed.*`, `time_mlp.*`, `class_emb.weight`] +
84 block tensors [14 per block x 6 blocks] + 4 tail tensors
[`final_norm.*`, `unproj.*`] -- so 95 others, not 83. The original "83"
was this document's own arithmetic error, mistaking the 84-tensor
per-block subtotal for the grand total; the GPU run's actual computed
value, `n_other_tensors=95`, is correct and was not affected by this
documentation error, since the code counts `len(other_means)` at
runtime rather than using a hardcoded number.)
Chosen over bucketing (e.g. "all attention params" as one number)
because averaging over the 6 blocks' worth of a given type would
obscure block-specific heterogeneity -- exactly the kind of information
loss this project's own discipline (Item 3's block-level vs. sub-block
scope statement) has already flagged as worth avoiding. A single-mean
comparator is explicitly rejected for the same reason: it could hide
`class_emb` being competitive with some groups (e.g. LayerNorm params)
but not others (e.g. attention QKV projections), which would itself be
an interesting, legible finding.

**Secondary (interpretability only, not the primary claim): a
type-bucketed summary** -- embeddings (`patch_embed.lead_emb.weight`,
`class_emb.weight`), norms (`norm1`/`norm2`/`final_norm`), attention
(`attn.*`), FFN (`ff.*`), adaLN (`adaLN.*`), projection
(`patch_embed.proj.*`, `time_mlp.*`, `unproj.*`) -- reported alongside
the per-tensor percentile rank for readability, never substituted for it.

## Design question 4: statistical treatment

**Purely descriptive** (percentile rank + distribution summary, mean +
std per tensor). No formal hypothesis test invented: unlike Items 1-3,
there is no natural paired/null-hypothesis structure here (a single
checkpoint, not a multi-condition sweep) -- inventing a test for its own
sake would manufacture false precision. If a future item needs to
compare gradient competitiveness across multiple checkpoints (were one
ever available), a paired test would become appropriate then, not now.

## `common/` reuse (restated from the investigate-only pass; one correction)

- `common/io.py` -- **REUSABLE AS-IS** (`load_model_checkpoint`,
  `load_config`, `get_logger`, `REPO_ROOT`).
- `common/utils.py` -- **PARTIALLY REUSABLE, one correction from the
  investigate-only pass:** `K_DRAWS`/`TIMESTEPS`/`class_pairs`/
  `GAIN_GRID` are NOT reused for Item 4's primary design (see Design
  Question 2 -- N=10 is a new, separately-justified constant). The
  secondary fixed-timestep design DOES reuse the `{100, 500, 900}`
  convention (informally, as a value set, not by importing
  `common/utils.TIMESTEPS` into a design it wasn't calibrated for).
  This corrects the investigate-only pass's implicit assumption that
  `utils.py`'s constants would carry over wholesale.
- `common/hooks.py` -- **NOT APPLICABLE** (no forward-activation hooks;
  gradients come from `named_parameters()` after `.backward()`).
- `common/metrics.py` -- **NOT APPLICABLE** (gradient norm per parameter
  tensor is a different quantity than any existing function).
- `common/statistics.py` -- **NOT APPLICABLE** (Item 2's constants are
  gain-sweep-specific; Item 4 has no decision table, per Design
  Question 4).
- `common/plotting.py` -- **REUSABLE WITH EXTENSION**: one new function,
  a bar/distribution plot of `class_emb.weight`'s gradient norm against
  the distribution of all other parameter tensors.

## Required outputs

- `gradient_probe_raw.json` -- per-tensor, per-draw gradient norms (both
  primary and secondary designs), plus the state-dict checksum
  before/after and the zero-grad reproducibility check result.
- `gradient_probe_summary.csv` -- per-tensor pooled mean/std `||grad||`,
  percentile rank of `class_emb.weight`, type-bucket summary.
- One plot: `class_emb.weight` vs. distribution of all other tensors.
- `Item4_Report.md` -- architectural question stated first, findings,
  permanent-limitation section (verbatim from above), verdict.

## Runtime / compute (corrected -- original estimate below was wrong)

**Original estimate ("seconds to low minutes on CPU") was wrong,
corrected here rather than left stale in this document.** Unlike Items
1-3 (synthetic single-sample forward-only probes), Item 4 loads the real
836MB `X_train.npy` and does real per-record CSV lookups against
`ptbxl_database.csv`, then runs 40 full mini-batch (`batch_size=32`)
forward+backward passes with active dropout on an 8.4M-parameter model
-- a substantially heavier operation. **Measured directly on this Mac
(CPU-forced, two timed runs, not estimated):** one-time data load ~1-5.5
minutes (variance from OS file-cache state -- the second run was faster
because the 836MB file was already in the page cache from the first
run), plus ~20-25 seconds per forward+backward draw. Total observed:
~25-30 minutes CPU-only on this machine, with per-draw timing also
affected by other processes competing for CPU on this laptop during the
timed runs.

**Moved to the GPU server for the full sweep** given this cost --
expected substantially faster per-draw compute on a real CUDA GPU
(the 8.4M-parameter model's forward+backward should drop from ~20-25s
to low single digits of seconds per draw); the ~1-5.5 minute data-load
component is disk I/O / CPU-side tensor conversion and is not expected
to speed up from GPU compute alone. No optimizer step, no weight
updates, either way.

## Amendment (approved before the final rerun -- N_DRAWS 10 -> 30)

Runtime on CUDA was substantially lower than estimated during planning
(~0.5s/draw, confirmed by the first GPU run's log timestamps -- 10
primary draws completed in ~5 seconds). **Before final analysis, the
number of stochastic draws was increased from 10 to 30 (applied to both
the primary and secondary/fixed-timestep designs) to reduce estimator
variance in the percentile-rank statistic** -- the percentile rank is
inherently a sampled quantity (each draw uses a fresh random real-data
batch), and a 10-draw estimate carries more sampling noise than a
30-draw one, at a marginal GPU cost of ~10-15 seconds. **No hypotheses,
metrics, decision criteria, loss definition, hooks, timestep choices,
checkpoint, or filtering logic were changed** -- only the draw count,
justified purely by the cost/precision tradeoff revealed once real CUDA
runtime was measured (this is the same class of amendment as Item 6's
post-hoc CI computation: improving the precision of an already-locked
measurement, not changing what is being measured).

## Report structure

Architectural question first ("was the class embedding's gradient
signal ever competitive, or structurally starved from the start,
independent of what the forward pass showed in Items 1-3"), then
findings (percentile rank, per-timestep breakdown, type-bucket summary),
the permanent-limitation section verbatim, then a descriptive verdict
(no pass/fail decision table, per Design Question 4).

## Lock

This pre-registration is frozen once implementation begins. Any change
discovered necessary mid-implementation becomes an explicit "Item 4
Revision" note, not a silent edit, matching Items 2/3's standing rule.
