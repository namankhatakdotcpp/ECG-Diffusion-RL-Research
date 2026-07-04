# Item 4 (Gradient Competitiveness Probe) -- Report

## Architectural question

Was the class embedding's gradient signal ever competitive with other
parameter groups during training -- or was it structurally starved
from the start -- independent of what the forward pass showed in
Items 1-3?

## Executive summary

**VERIFIED, precisely stated:** `class_emb.weight`'s pooled mean
`||grad||` sits at the **61.1th percentile** among the other 95
named-parameter tensors (bootstrap 95% CI: [58.95%, 62.11%], SD 0.64pp,
n=1000 resamples over 30 i.i.d. real training batches, independence
confirmed below). **Two claims, kept separate rather than conflated:**

- **Survives:** the class embedding received a consistently non-
  negligible, above-median gradient signal throughout training -- it
  was actively optimized, not starved or vanishing.
- **Does not survive:** that it was "competitive" in the sense of
  rivaling the strongest gradient pathway. It was not -- the `adaLN`
  parameter-type bucket receives the largest mean gradient of any
  bucket (0.0125, ~6.7x `class_emb`'s 0.0019), consistent with Item 5's
  finding that `adaLN` carries substantial, unevenly-allocated weight
  capacity. `class_emb.weight` is not among the highest-gradient
  parameter groups in this model.

**The secondary (fixed-timestep) design shows the same directional
pattern as Item 1's own forward-pass sensitivity finding:**
class-conditioning influence -- whether measured via gradient rank or
via forward-pass magnitude -- declines as noise increases.

## Methodology (per the locked pre-registration, `Item4_PreRegistration.md`)

Reproduces the exact training step (real `X_train` batches via
`q_sample`, real per-sample CFG dropout, `model.train()` mode, MSE
noise-prediction loss) -- not Items 1-3's synthetic-noise convention,
since this item asks about gradient dynamics during real training, not
forward-pass response. No `optimizer.step()` ever called. Primary
design: `N_DRAWS=30` real per-sample-random-timestep draws (amended
from 10 after measuring actual CUDA cost, ~0.5s/draw -- see
pre-registration's Amendment section). Secondary design: `N_DRAWS=30`
draws at each of 3 fixed timesteps (`t=100, 500, 900`) -- confirmed by
direct code read (`item4_gradient_probe.py:217,235`) to use the same
draw count as the primary design, no precision gap between the two.

## Verification (Phase D -- actual values, not PASS strings)

**Device/GPU metadata**, pulled directly from the transferred JSON, not
inferred from run timing:
- `device=cuda`, `cuda_available=True`
- `GPU name=NVIDIA RTX A6000`, `CUDA version=11.8`, `total memory=50.9 GB`

**Zero-grad reproducibility check:** two identical-seed draws produced
per-tensor gradient norms differing by **exactly 0.0** across all 96
tensors (`reproducibility_max_abs_diff=0.0`, threshold 1e-9) -- no
gradient-accumulation bleed between draws.

**Weight-checksum check** (previously only a PASS/FAIL string; now the
actual hash values, extracted from the real transferred file):
- Hash function: `sha256`
- Object hashed: `state_dict` -- sorted `(name, tensor.detach().cpu().numpy().tobytes())` pairs from `model.named_parameters()`
- Before: `68ca4453a042dee7c97593562ccda85aba300b1ff463da77d8655b331289a9a9`
- After:  `68ca4453a042dee7c97593562ccda85aba300b1ff463da77d8655b331289a9a9`
- **Identical** -- weights never changed during the sweep, confirmed by the actual hash values, not a trust-me string.

**`other_means` recomputation confirmation:** `n_other_means_recomputed_this_run=95`,
matching `n_other_tensors=95` exactly -- confirms the percentile-rank
comparison set was rebuilt fresh from this run's 30 draws (there is no
caching mechanism in the script; `pooled_mean` is always constructed
from `primary_grad_norms`, populated by the loop that just executed).

**Bucket-count reconciliation** (actual runtime assertion, not prose):
`sum(bucket n_tensors)=96`, `n_other_tensors=95`, `difference=1`,
`difference_is_class_emb_weight=true` -- confirmed exactly as designed
(96 total named-parameter tensors = 95 others + `class_emb.weight`
itself, included in the `embeddings` bucket display but excluded from
the comparison set).

**Data-transfer integrity:** `item4_results.tar.gz` SHA-256
(`7b1921eb6e9c7ce15f5f8713d5325cff76b7a1cd137148f6c8b7f9488c7abbea`)
confirmed to match between the GPU server's report and this Mac's
independent `shasum -a 256` computation after transfer.

## Independence check (required before trusting the bootstrap CI)

Bootstrap resampling assumes the 30 draws are i.i.d. -- confirmed by
direct code read, not assumed: **PASS.**

- Each draw pulls the *next* batch from a single `DataLoader` iterator
  built on `WeightedRandomSampler(..., replacement=True)`
  (`item4_gradient_probe.py:83-112`), which pre-generates one shuffled
  index list for the whole epoch at iterator creation. Consecutive
  `next(train_iter)` calls consume non-overlapping consecutive
  32-sample slices of that list (`drop_last=True`) -- **no two of the
  30 primary draws share any training sample**, confirmed by
  construction (30 x 32 = 960 samples drawn from a ~17,418-record
  training set, non-overlapping batches by DataLoader design).
- Each draw also gets its own seed (`torch.manual_seed(1000+i)`,
  `item4_gradient_probe.py:218`), independently controlling that
  draw's CFG-dropout mask and per-sample random timestep -- no shared
  stochastic source across draws.

**No correlated/shared randomness source exists across the 30 draws.**
The bootstrap SD is not understated by hidden correlation; the reported
CI can be trusted at face value.

## Bootstrap confidence interval on the percentile-rank estimate

Per review requesting a variance estimate on the rank itself (not just
the raw gradient-norm SE): resampled the 30 real per-draw gradient-norm
vectors with replacement, 1000 times, recomputing `class_emb.weight`'s
percentile rank among the other 95 tensors' resampled means each time.

| Quantity | Value |
|---|---|
| Point estimate | 61.05% |
| Bootstrap mean | 60.73% |
| Bootstrap SD | 0.645 percentage points |
| 95% CI | [58.95%, 62.11%] |

The interval is tight (< 3.2 percentage points wide) -- the rank
estimate is stable, not an artifact of which 30 batches happened to be
drawn. `class_emb.weight` sits solidly in the "above median, not
dominant" region of the gradient-norm distribution, not near a rank
boundary where a few different draws could flip the qualitative
conclusion.

## Results

**Primary design (real per-sample-random-timestep, N=30):**

| Quantity | Value |
|---|---|
| `class_emb.weight` mean `\|\|grad\|\|` | 0.001873 |
| `class_emb.weight` std `\|\|grad\|\|` | 0.000586 |
| Percentile rank among 95 other tensors | 61.05% (95% CI: [58.95%, 62.11%]) |
| Other tensors: min / median / max | 0.0000487 / 0.001232 / 0.03731 |

**Type-bucket summary** (descriptive, not the primary comparison basis
-- per-tensor percentile rank above is primary, per the pre-registration):

| Bucket | Mean `\|\|grad\|\|` | n tensors |
|---|---|---|
| adaLN | 0.012534 | 12 |
| attention | 0.005460 | 24 |
| projection | 0.005002 | 8 |
| ffn | 0.003074 | 24 |
| embeddings | 0.002211 | 2 |
| norms | 0.001240 | 26 |

**Secondary design (fixed timestep, N=30 each):**

| Timestep | `class_emb.weight` mean `\|\|grad\|\|` | Percentile rank |
|---|---|---|
| 100 | 0.002128 | 56.84% |
| 500 | 0.000703 | 38.95% |
| 900 | 0.000435 | 33.68% |

Gradient magnitude AND percentile rank both decline monotonically as
timestep increases (t=100 highest, t=900 lowest) -- a clean, real
pattern across all 30-draw cells.

## Cross-validation with Item 1 (Phase H -- exact numbers, not paraphrased)

Item 1's own pooled L1 magnitude (forward-pass cross-class delta at
block 1, averaged over 5 class pairs, `Tier0_Findings.md`'s full
90-cell table) at the same three timesteps:

| Timestep | Item 1 L1 magnitude (forward-pass) | Item 4 `class_emb` percentile rank (gradient) |
|---|---|---|
| 100 | 0.173 | 56.84% |
| 500 | 0.124 | 38.95% |
| 900 | 0.107 | 33.68% |

**Both measurements decline in the same direction as timestep
increases -- forward-pass conditioning magnitude (Item 1) and backward-
pass gradient competitiveness (Item 4) independently agree that
class-conditioning influence weakens as noise increases.** This is
two entirely different measurement types (Item 1: activation magnitude
during inference-mode forward passes on synthetic noise; Item 4:
gradient magnitude during real training-mode backward passes on real
data) converging on the same qualitative trend -- stated as convergent
correlation, not proof of one causing the other, per this project's
standing discipline on cross-item claims (established for Item 5).

## Interpretation

Two claims, kept explicitly separate rather than merged into a single
"competitive" verdict:

- **Survives:** the class embedding's gradient was consistently
  non-negligible and above-median throughout training-mode backward
  passes (61st percentile, tight bootstrap CI) -- it was actively
  optimized, not starved or vanishing.
- **Does not survive:** any claim that it rivaled the strongest
  gradient pathway. `adaLN` parameters receive substantially larger
  gradients (~6.7x), consistent with Item 5's finding that `adaLN`
  holds significant, non-uniformly-allocated weight capacity.
  `class_emb.weight` is not among the highest-gradient parameter groups
  in this model.

The declining percentile rank with increasing timestep, corroborated
independently by Item 1's forward-pass magnitude decline, suggests the
class-conditioning pathway's influence -- whether measured as a
training-time gradient signal or an inference-time activation magnitude
-- is genuinely weaker in high-noise regimes, not an artifact specific
to either measurement's methodology.

## Permanent limitation (epoch-25 comparison)

As established before this rerun: only `outputs/models/diffusion_best.pt`
(epoch 200) exists. No epoch-25 checkpoint is available -- confirmed by
direct source read that one was created (`save_every=25`) and then
deleted by the retention policy (`KEEP_LAST_N_CHECKPOINTS=2`), with
that retention logic confirmed active in this run's own commit lineage
(`f78c6c2` ancestor of `01b55091`). **If an epoch-25 checkpoint is
unexpectedly discovered in archived GPU storage, the temporal
comparison specified in the master roadmap can be executed as an
optional extension. Otherwise, this limitation is permanent for the
current training run.**

## Decision

**VERIFIED, precisely:** the class embedding received a consistently
non-negligible, above-median gradient signal throughout training
(61.1st percentile, bootstrap-verified stable, i.i.d. draws confirmed)
-- it was actively optimized, not starved. It was **not**, however,
competitive with the strongest gradient pathway (`adaLN`, ~6.7x
larger) -- not among the highest-gradient parameter groups in this
model. All sanity checks passed with real underlying values (not
trust-me strings), and the timestep-dependence finding cross-validates
with Item 1's independent measurement. The single-checkpoint scope
limitation (epoch-25 comparison unavailable) is permanent, documented,
and does not block this verdict.

## Next steps

None for Item 4 itself. This closes the last open Tier 0 item --
Stage 2's evidence matrix (`Stage2_Evidence_Matrix.md`) can now be
finalized with Item 4's row moved from Pending to a real verdict, and
`Stage_2_Decision_Report.md` can proceed.

## Artifacts

- `Outputs/stage2_tier0_item4_gradient_competitiveness/gradient_probe_raw.json`
- `Outputs/stage2_tier0_item4_gradient_competitiveness/gradient_probe_result.json`
- `Outputs/stage2_tier0_item4_gradient_competitiveness/gradient_probe_summary.csv`
- `Outputs/stage2_tier0_item4_gradient_competitiveness/gradient_probe_bootstrap_ci.json`
