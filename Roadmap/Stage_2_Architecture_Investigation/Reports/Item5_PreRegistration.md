# Item 5 -- AdaLN/FiLM Parameter Statistics -- Pre-Registration

## Hypothesis, quoted verbatim (`Stage2_Master_Prompt.md:211-214`)

> "5. AdaLN / FiLM parameter statistics. Per-block adaLN weight-matrix
> Frobenius norm; fraction devoted to scale1/scale2 vs shift1/shift2.
> A second, independent way to test the channel-capacity question
> Stage 1's CFG sweep addressed indirectly."

## Architectural question

Does the adaLN modulation mechanism allocate its parameter capacity
evenly between the two modulation types (additive shift vs. multiplicative
scale), or is one type structurally dominant -- and does this allocation
pattern change across the 6 blocks, correlating with where Item 1/3
found conditioning signal concentrates or attenuates?

## Confirmed by direct source read (not assumed)

`step04_transformer_diffusion.py:169, 171-172`:
```python
self.adaLN = nn.Linear(2 * model_dim, 4 * model_dim)
...
shift1, scale1, shift2, scale2 = self.adaLN(cond).chunk(4, dim=-1)
```
`adaLN.weight` has shape `(4*model_dim, 2*model_dim)` = `(1024, 512)`
(confirmed directly against the loaded checkpoint's `named_parameters()`
in Item 4's investigate-before-code phase). Since `adaLN` is a plain
`nn.Linear` (`y = W @ cond + b`), `torch.chunk(4, dim=-1)` on the OUTPUT
splits it into 4 contiguous length-`model_dim` (256) segments in a fixed
order: `shift1, scale1, shift2, scale2`. Each output segment is produced
by a fixed, disjoint ROW-SLICE of `W` (and the matching slice of `bias`):
`shift1 <- W[0:256, :]`, `scale1 <- W[256:512, :]`,
`shift2 <- W[512:768, :]`, `scale2 <- W[768:1024, :]`. This is an exact
decomposition confirmed from the chunk semantics and linear-layer math,
not an approximation.

## Design

- **No forward pass needed at all** -- this is pure weight inspection.
  CPU-trivial (milliseconds), no checkpoint sampling, no draws, no
  timesteps. This is a genuinely different cost profile from Items 1-4.
- Per block `k` (1-6): compute `||W_k||_F` (full adaLN weight Frobenius
  norm), and `||W_k[shift1]||_F`, `||W_k[scale1]||_F`,
  `||W_k[shift2]||_F`, `||W_k[scale2]||_F` (each of the 4 row-slices).
- **Fraction devoted to scale vs. shift**, per the master prompt's own
  phrasing: `scale_fraction_k = (||scale1||_F^2 + ||scale2||_F^2) /
  ||W_k||_F^2` (sum-of-squares, since Frobenius norms combine in
  quadrature for disjoint row-blocks: `||W||_F^2 = sum of each
  row-block's ||.||_F^2` exactly, confirmed algebraically -- not an
  approximation). `shift_fraction_k = 1 - scale_fraction_k`.
- Bias vectors treated identically (`bias.norm()` per chunk), reported
  alongside but not conflated with the weight-matrix fractions (the
  master prompt's phrasing is about the weight matrix specifically).
- Repeat for all 6 blocks; report per-block trend, not just a pooled
  average -- consistent with Items 1/3's "per-block AND pooled" standard.

## Statistical treatment

Purely descriptive (no draws, no randomness, no sampling -- every
number is an exact, deterministic property of the frozen checkpoint's
weights). No hypothesis test applicable; reporting exact values with
no confidence interval needed, since there is no sampling variance to
characterize.

## Cross-validation targets (Phase H, planned before running)

- **Item 4** (just completed): found `adaLN` parameters receive the
  largest mean gradient norm of any bucket (~0.013, ~6x `class_emb`'s
  mean). If Item 5 finds `scale_fraction` is heavily skewed at the SAME
  blocks where Item 1/3 found conditioning effects concentrate (block
  1 and/or block 6), that would be a direct mechanistic explanation
  connecting "where gradient flows" (Item 4) to "what capacity exists to
  receive it" (Item 5) to "where the forward-pass effect appears" (Item
  1/3) -- three independent measurement types converging on the same
  loci, which would be a strong finding for the eventual Decision Report.
- **Item 1's two-drop shape** (dominant block1->2, smaller block5->6):
  check whether `scale_fraction`/`||W||_F` at blocks 1 and 6 differ
  from blocks 2-5 in a way that could explain why those two transitions
  are where the effect concentrates.

## Decision criteria

Descriptive verdict (VERIFIED/FALSIFIED per the master prompt's binary
framing, applied here as): **VERIFIED** if a clear, non-uniform
scale/shift allocation pattern exists across blocks (i.e., not all 6
blocks have `scale_fraction` within a few percentage points of each
other) AND it is legible enough to state a directional relationship to
Item 1/3/4's findings; **FALSIFIED** (more precisely: no informative
pattern) if allocation is essentially uniform across blocks with no
discernible relationship to prior findings. This is stated before
running, per the master prompt's "never report a metric without
sample size/limitations" discipline -- though sample size here is
trivially n=1 (one frozen checkpoint, no draws), which is itself a
limitation to state explicitly, not hide.

## `common/` reuse

- `common/io.py` -- REUSABLE AS-IS (`load_model_checkpoint`, `load_config`, `get_logger`).
- `common/hooks.py`, `common/metrics.py` -- NOT APPLICABLE (no forward pass, no activations).
- `common/statistics.py` -- NOT APPLICABLE (no decision-table constants apply; this item has its own binary verdict rule above).
- `common/plotting.py` -- REUSABLE WITH EXTENSION: one new bar/grouped-bar plot (scale vs. shift fraction per block).

## Required outputs

- `adaln_statistics.json` -- per-block Frobenius norms (full + 4 chunks), scale/shift fractions, bias norms.
- `adaln_statistics.csv` -- per-block summary table.
- One plot: scale-fraction vs. block index (with shift-fraction as the complement).
- `Item5_Report.md`.

## Runtime / compute

**CPU, sub-second.** Pure tensor-norm computation on already-loaded
checkpoint weights, no forward pass, no data loading, no draws.

## Limitations, stated up front

n=1 (one frozen checkpoint) -- no variance/confidence interval is
meaningful here; this measures a static property of the trained
weights, not a sampled quantity. Cannot by itself establish causation
between allocation pattern and the forward-pass/gradient findings from
Items 1/3/4 -- only correlation/consistency, stated as such in the
report's Interpretation section.

## Lock

Frozen once implementation begins; any change becomes an explicit
revision note, matching Items 2-4's standing rule.
