# Item 3 -- Residual-Path Attenuation -- Pre-Registration

Locked before implementation, per this project's standing "no code until
it's in writing" discipline (established for Item 2, applied here without
another multi-cycle review marathon, per explicit instruction). Item 3's
own falsification criteria are drafted here in full, since the master
prompt (`Stage2_Master_Prompt.md:199-203`) gives only the measurement
definition, not a decision procedure -- same gap Item 2 had before its
own pre-registration.

## Preliminary check: does the zero-new-inference claim actually hold?

**PASS, confirmed by direct source read** (`step04_transformer_diffusion.py:257-282`):
`cond_film` is computed once, before the block loop (line 268:
`cond_film = torch.cat([t_emb, c_emb], dim=-1)`), and passed **unmodified**
to every block in `for block in self.blocks: tokens = block(tokens,
cond_film)` (lines 274-275) -- only `tokens` is reassigned per iteration,
`cond_film` never changes. This fully justifies the dependency audit's
claim that block *k*'s output tensor **is** block *k+1*'s input tensor,
bit-identical, with no intervening transform -- not an assumption from a
one-line snippet, a confirmed fact from the full `forward()` method.

## Architectural question

Item 1 showed that class-conditioning magnitude decreases across the
network (the two-drop shape: dominant block1->2, smaller real
block5->6). **Item 3 asks whether that decrease is already present
inside each residual update, or whether it only appears after the
residual update has been merged into the running representation:**

**Does conditioning attenuation occur primarily inside the residual
branch itself (visible as a declining residual-update ratio), or does it
emerge only after residual addition (update ratio flat/stable, while
Item 1's raw post-addition output magnitude still attenuates)?**

This is the question Item 3's data can actually discriminate between --
not a restatement of "compute this ratio," but a question about *where*,
architecturally, the effect Item 1 already confirmed originates.

**Item 3 measures an intra-pass quantity (input vs. output within the
same forward pass), whereas Item 1 measures an inter-condition quantity
(same seed/timestep, different class labels). Therefore the two
measurements are complementary rather than redundant.**

### What Item 3 adds beyond Item 1

Item 3 is not a duplicate of Item 1 -- it measures a different physical
quantity, at a different point in the computation graph:

| | Item 1 | Item 3 |
|---|---|---|
| Measures | Post-addition output magnitude | Residual update magnitude (pre-addition) |
| Quantity type | Cross-class delta (`feat_B - feat_A`), inter-condition | Within-pass update (`H_k^out - H_k^in`), intra-pass, per class |
| Signal location | The residual stream itself, after the block's contribution is merged in | The residual branch's own contribution, before it is merged in |

Per the model source (`step04_transformer_diffusion.py:171-177`,
`TransformerBlock.forward`), each block actually contains **two**
residual adds, not one -- `x = x + h` (attention output) then
`x = x + self.ff(...)` (FFN output). Item 3's `R_k`, per the master
prompt's own block-level definition (`||block_output - block_input||`,
not a sub-block quantity), measures the **combined** effect of both
adds together, matching Item 1/2's own block-level hook granularity
(neither hooks inside a block). The diagram below shows both sub-block
residual adds explicitly, so this combination is visible rather than
implied:

```
Input H_k^in
   |
LayerNorm (norm1) -> Attention -> (+) -------\
   |                                          |
   +<-----------------------------------------/   [1st residual add]
   |
LayerNorm (norm2) -> FFN -> (+) -------\
   |                                    |
   +<-----------------------------------/          [2nd residual add]
   |
Output H_k^out = H_k^in + (attn contribution) + (ffn contribution)
   |
   |   Item 3 measures the COMBINED update here:
   |   DeltaH_k = H_k^out - H_k^in  (spans both adds above, block-level)
   v
Output Tensor H_k^out  <---- Item 1 measures here (post-addition, cross-class)
```

If the update ratio itself declines block-to-block, the attenuation
Item 1 found is (at least partly) generated inside the residual branch's
own computation. If the update ratio stays flat while Item 1's own
output-magnitude curve still declines, the attenuation is a property of
how the update interacts with the growing residual stream after
addition, not of the update's own magnitude -- two different
architectural loci for the same observed downstream symptom.

## Formal definition

For a fixed (class-pair, timestep, draw `i`), let `H_k^in(i)` and
`H_k^out(i)` denote the full per-token hidden-state tensor TransformerBlock
`k` receives and returns respectively (shape `(1, 600, model_dim)`, same
tensor family Item 1/2 already hook, mean-pooled the same way for the
metrics below). Define the residual update:

```
DeltaH_k(i) = H_k^out(i) - H_k^in(i)

R_k(i) = ||DeltaH_k(i)|| / ||H_k^in(i)||     (mean-pooled norms, Item 1's own convention)
```

**Item 3 analyzes `R_k`** -- its mean across draws per block, per class,
per pair, and its trend across blocks 1-6. This replaces the earlier
prose description ("residual update ratio") with a fixed, unambiguous
notation for every equation and result table that follows.

**Scope statement:** `R_k` measures the combined effect of both
residual adds within block `k` (attention-sublayer and FFN-sublayer,
per `step04_transformer_diffusion.py:171-177`); sub-block decomposition
is out of scope for Item 3, consistent with Item 1/2's block-level
hooking granularity (neither hooks inside a block). A future
sub-block-level probe, if ever needed, is a separate item, not a silent
extension of this one.

## Scientific hypothesis (directional-neutral)

**`R_k`, the residual update magnitude, varies systematically across
Transformer blocks 1-6** (not assumed to be attenuation -- Item 1's own
finding was a front-loaded two-drop shape, not smooth geometric decay,
which is itself evidence against assuming a shape before measuring).
Competing alternatives, none privileged as the expected answer:

- **Attenuation:** `R_k` declines across blocks.
- **Amplification:** `R_k` grows across blocks.
- **Non-monotonic:** `R_k` rises and falls without a consistent trend.

## Null hypothesis

`R_k` is flat or noise-like across blocks 1-6 -- no systematic
block-to-block trend, consistent with the null that the residual
branch's own contribution is architecture-driven noise rather than a
locus of conditioning-signal change.

## All possible outcomes and interpretations (stated before data exists)

| Observation | Interpretation |
|---|---|
| `R_k` decreases | Consistent with the residual branch itself attenuating conditioning signal |
| `R_k` flat | Consistent with attenuation arising only after residual addition, not inside the branch |
| `R_k` increases | Would not match Item 1's simple attenuation narrative at the update level -- requires reconciling why output magnitude still declines despite growing per-block updates |
| `R_k` non-monotonic | Suggests block-specific dynamics rather than a single uniform mechanism |

Stating all four outcomes now, before any sweep is run, is what prevents
picking an interpretation to fit whatever number comes back later.

## Candidate statistical test (not locked)

**Leading candidate: Wilcoxon signed-rank test**, matching Item 1's own
paired design (`a089496`) and its pooled n=15 (5 pairs x 3 timesteps)
evidentiary bar. Not locked yet -- final choice confirmed once Item 3's
real data distribution (normality, ties, effect size) is inspected
during implementation, same discipline as leaving the statistical test
open until data exists, not assumed in advance.

## Decision criteria

**Item 3 defines a different physical quantity than Item 2 (`R_k`, a
within-pass residual-update ratio, vs. Item 2's cross-class output-
magnitude drop). Therefore no thresholds, pooled constants, or decision
tables are inherited from Item 2** -- specifically, `R_k`'s criteria are
**TBD from Item 3's own fresh sweep**, explicitly not borrowed from
Item 2's `POOLED_BLOCK1_TO_2_DROP = 0.0635`. Locking Item 3's own pooled
baseline and thresholds happens after the fresh sweep produces real
numbers, following the same procedure Item 1 used to establish its own
pooled statistics before Item 2's criteria were written -- not before.
This line is stated explicitly and in bold specifically to prevent a
future silent reuse of `0.0635` "because it worked before" -- it would be
measuring the wrong quantity's threshold against the right quantity's
number.

## Interpretation framework

(Not "failure modes" -- none of these outcomes are a failure of the
experiment; each is a legitimate scientific result with its own
follow-up question, phrased as a question to investigate further, not a
locked conclusion.)

| Observation | Follow-up question |
|---|---|
| `R_k` flat | Why does Item 1's output magnitude attenuate while the residual update itself does not? Requires direct comparison against Item 1's own raw output-magnitude curve (same blocks, same checkpoint) -- not assumed consistent with a "post-addition" story until that comparison is actually made. |
| `R_k` declining | Which residual sub-component (attention vs. FFN) drives the decline, if Item 3's granularity (block-level, per the master prompt's definition) can distinguish that at all -- may require a scope note that this question is out of Item 3's current granularity, not silently assumed answerable. |
| `R_k` increasing | Why would residual amplification be hidden in Item 1's own output-magnitude measurement? Requires reconciling growing per-block updates with a still-declining output signal, not asserted as a contradiction before that reconciliation is attempted. |
| `R_k` non-monotonic | Does this reflect a real architecture effect (block-specific dynamics) or a methodological difference between Item 3's within-pass measurement and Item 1's cross-class measurement? Investigate before drawing any conclusion about consistency with Item 1's two-drop shape -- not pre-classified as "contradicts Item 1" until that investigation happens. |

## Reused modules / new extensions (from the dependency audit, restated)

- `common/io.py` -- reused as-is (`load_model_checkpoint`, `load_config`,
  `get_logger`, `REPO_ROOT`).
- `common/utils.py` -- `class_pairs`, `K_DRAWS`, `TIMESTEPS` reused
  as-is (same paired-pass design as Item 1/2). `GAIN_GRID`,
  `uniform_per_block_gain`, `N_UNIFORM_BLOCKS` not applicable (no
  intervention in Item 3).
- `common/hooks.py` -- `register_layer_hooks` reused as-is for every
  block's output (which doubles, for free, as blocks 2-6's input, per
  the confirmed `cond_film`-constancy check above). **New addition:**
  one `register_forward_pre_hook` helper on `model.blocks[0]`, to
  capture block 1's true input directly (the one tensor with no
  existing hookable source, per the dependency audit).
- `common/metrics.py` -- `cosine_sim`/`magnitude_and_consistency`
  pattern reused as a template; **new addition:** a
  `residual_update_ratio(block_output, block_input)` sibling function
  computing `||out - in|| / ||in||` per draw, aggregated the same way
  Item 1's `magnitude_and_consistency` aggregates (mean over draws,
  normalized).
- `common/statistics.py` -- pattern reused (decision-table structure),
  **contents not reused** (Item 2's constants are specific to a
  different measured quantity, per Decision Criteria above). Item 3
  gets its own constants once real data exists.
- `common/plotting.py` -- Agg-backend/per-function pattern reused;
  **new addition:** a residual-update-ratio-vs-block plot, direct
  analogue of Item 1's own magnitude-vs-layer plot (no gain axis,
  unlike Item 2's three plots).
- **No identity-regression test** -- correctly not invented. Item 3 has
  no intervention/substitution hook, only measurement on unmodified
  forward passes (like Item 1), so there is nothing to regression-test
  against a `g=1` no-op.

## Required outputs

- `residual_probe_raw.json` -- per-pair, per-layer detail (draws,
  per-draw ratios), mirroring Item 1's own raw-JSON naming/structure.
- `residual_probe.csv` -- per-layer pooled summary (mean ratio per
  block, averaged across the 5 class pairs), same shape as Item 1's
  `layerwise_probe_baseline.csv`.
- One plot: residual-update-ratio vs. block index.
- `Item3_Report.md` -- architectural question, hypothesis/null restated,
  results table, statistical test result, verdict against Item 3's own
  (data-derived) criteria, interpretation framework actually applied
  to the observed shape (which row of the outcomes table matched, and
  which follow-up question it raises).

## Runtime / compute

**CPU-only, no GPU, ~1-2 minutes for the full sweep.** To maintain
comparability with Item 1 and Item 2, Item 3 reuses the identical
experimental design **inherited from Item 1** (5 class pairs x 3
timesteps x 20 draws) -- not independently chosen, so a future reader
should not assume Item 3 picked these numbers on its own grounds. Only
ONE forward pass per class label per draw is needed (no
baseline-vs-corrected doubling, since there's no intervention) -- if
anything cheaper than Item 1's own original run.

## Report structure

Mirrors `Item2_Report.md`/`Item2B_Report.md`'s shape: architectural
question and hypothesis restated up top, results table, statistical
test outcome, verdict against Item 3's own criteria (locked only after
the sweep produces real pooled numbers), explicit application of the
Interpretation Framework to whatever shape is actually observed -- not a
template answer decided in advance.

## Lock

This pre-registration is frozen once reviewed and cleared. No further
methodological changes once implementation begins -- a change discovered
necessary mid-implementation becomes an explicit "Item 3 Revision"
section, not a silent edit to the criteria above, matching Item 2's own
standing rule.
