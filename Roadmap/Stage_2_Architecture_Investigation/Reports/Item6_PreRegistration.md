# Item 6 -- Attention Entropy and Attention-Map Inspection -- Pre-Registration

## Hypothesis, quoted verbatim (`Stage2_Master_Prompt.md:216-222`)

> "6. Attention entropy and attention-map inspection.
>        H = -sum(p * log(p))
>    averaged over heads and query positions, for NORM-labeled vs.
>    STEMI-labeled generation with identical noise/seed/timestep. Near-
>    identical entropy/maps regardless of class label argues AGAINST
>    cross-attention being sufficient by itself -- adding cross-attention
>    to an already class-blind attention mechanism would not obviously help."

## Correction, confirmed by direct source read, not assumed

**There is no "STEMI" class in this project's real taxonomy.** Confirmed
against the loaded checkpoint's `class_names` (also matches Item 1-5's
own consistent usage): `['NORM', 'MI', 'STTC', 'CD', 'HYP', 'OTHER']`.
STEMI is a clinical subtype of MI, not one of this model's 6 output
classes -- the master prompt's wording here predates or is inconsistent
with the finalized taxonomy (same class of correction already applied
once in this project's history, per `Stage2_Master_Prompt.md`'s own
"OTHER (n=2 project-wide)" correction note). **Substituting MI (class
index 1)** for "STEMI," matching every other Tier 0 item's own 0-vs-1
first-pair convention (Items 1-3's `class_pairs` always start with
`(0, 1)` = NORM vs. MI).

## Architectural question

Does the model's existing self-attention mechanism (there is no
cross-attention in this architecture -- confirmed by source read,
`step04_transformer_diffusion.py:157-160`, `nn.MultiheadAttention`
called as `self.attn(h, h, h)`, i.e. self-attention on the token
sequence, with class conditioning injected only via `adaLN`'s
modulation, never as an attention query/key/value) already vary its
attention pattern with class label, or is attention itself "class-
blind" regardless of the (adaLN-mediated) conditioning signal? This
bears on whether a future architecture change adding actual cross-
attention (Stage 3, Tier 1, item 9-11 territory) would plausibly help,
or whether attention is already receiving the conditioning signal (via
adaLN's modulation of its input) and simply isn't using it distinctively.

## Design

- **No cross-attention exists in this model** -- confirmed directly.
  "Attention-map inspection" here means the existing self-attention
  matrices, computed on the same token sequence for two different
  class labels (0=NORM, 1=MI), same seed/noise/timestep.
- Per block, per class label: capture `attn_output_weights` with
  `need_weights=True, average_attn_weights=False` (per-head, NOT
  pre-averaged across heads -- entropy is a non-linear function of the
  distribution, so averaging attention weights across heads BEFORE
  computing entropy would give a different number than the master
  prompt's own formula, which computes `H` per (head, query) row THEN
  averages -- these are not interchangeable, and the pre-averaged
  default (`average_attn_weights=True`) would silently answer a
  different question). Confirmed the model's own `TransformerBlock.forward`
  (`step04_transformer_diffusion.py:174`, `h, _ = self.attn(h, h, h)`)
  discards attention weights entirely -- there is no way to get them
  from the model's own forward call as written; must call `block.attn`
  directly (same module, same weights, called a second time with
  `need_weights=True`) using the exact input tensor the block actually
  received (captured via a forward pre-hook, same mechanism as Item 3's
  `register_block0_input_hook`, reused pattern not reinvented).
- `H(head, query) = -sum_key(p * log(p + eps))` per attention row;
  average over all heads and all 600 query positions to get one scalar
  entropy per block per class label per draw.
- Same-seed, same-timestep, class-label-only-differs pairs -- same
  design principle as Item 1/3, reusing `class_pairs`/`TIMESTEPS`/
  `K_DRAWS` from `common/utils.py` (this item's forward-pass cost
  profile matches Items 1-3's synthetic-noise convention exactly, not
  Item 4's real-data convention -- no training data needed here, this
  is inference-mode inspection).
- `model.eval()` mode (no dropout) -- this is inspection of the frozen
  model's attention behavior, not training dynamics (unlike Item 4).

## Metric

`entropy_diff_k(pair, t, draw) = |H_k(class_B) - H_k(class_A)|` per
block, plus the raw `H_k(class_A)` and `H_k(class_B)` values themselves
(not just the difference -- a near-zero difference could mean "both
classes produce near-maximum entropy" (uniform, uninformative attention)
or "both produce near-zero entropy" (both classes collapse attention to
the same few tokens) -- these are different findings with the same
diff, so both raw values are reported, not just the delta).

## Decision criteria

Per the master prompt's own framing: **near-identical entropy across
class labels argues the attention mechanism is class-blind** (evidence
against cross-attention being a likely fix if added later). Locked
threshold: `mean(entropy_diff_k)` pooled across all cells/draws/blocks
`< 0.05` (on a `log(600) ~= 6.4`-scale entropy, this is a conservative
~0.8% relative threshold) -> **class-blind** (VERIFIED per the master
prompt's stated interpretation); `>= 0.05` -> **NOT class-blind**
(FALSIFIED -- attention already varies measurably with class label).
This threshold is stated now, before running, same discipline as every
prior item.

## `common/` reuse

- `common/io.py`, `common/utils.py` (`class_pairs`, `TIMESTEPS`,
  `K_DRAWS`) -- REUSABLE AS-IS (same design as Item 1/3).
- `common/hooks.py`'s `register_layer_hooks` pattern -- REUSABLE AS
  A PATTERN for the pre-hook capturing each block's attention input
  (new function needed: attention weights are not an activation
  `register_layer_hooks` already captures).
- `common/metrics.py` -- NOT APPLICABLE (entropy is a new metric,
  distinct from magnitude/consistency/residual-ratio).
- `common/statistics.py` -- NOT APPLICABLE (own threshold above).
- `common/plotting.py` -- REUSABLE WITH EXTENSION: one new plot
  (entropy per block, class A vs. class B overlaid).

## Required outputs

- `attention_entropy_raw.json`, `attention_entropy.csv`.
- One plot: entropy vs. block, class A/B overlaid.
- `Item6_Report.md`.

## Runtime / compute

CPU, same order as Item 1/3 (5 pairs x 3 timesteps x 20 draws, single
forward pass per class per draw, no training data, no backward pass) --
expected seconds to ~1-2 minutes.

## Lock

Frozen once implementation begins.
