# Stage 3 / Phase 0 -- Pre-Registration (Tasks 0.1 and 0.2)

Locked before implementation, per this project's standing "no code until
it's in writing" discipline (established Stage 2 Item 2, continued
here). **Decision thresholds below are locked by explicit user
directive for this pass** -- not independently derived from a fresh
sweep first, unlike Stage 2's items. This deviation from Stage 2's own
"criteria only after seeing the distribution" norm is intentional and
explicit: it trades a small risk of a miscalibrated threshold for
avoiding another multi-cycle pre-registration review, given Phase 0's
role as a cheap gate rather than a headline finding. If the real data
makes a threshold obviously meaningless (e.g. all ratios near zero),
that is reported as a limitation on the threshold, not silently
patched.

## Task 0.1 -- Dilution-ratio test

### Architectural question

Does conditioning's *proportional* influence on the residual stream
shrink across blocks -- i.e. does `conditioning_delta(block_k) /
total_output_norm(block_k)` decline from block 1 to block 6? This is
the "dilution mechanism" flagged in `Stage2_Decision_Report.md`
Conclusion 5b as a plausible synthesis of Items 1 (cross-class delta
grows/shrinks) and 3 (total output-norm growth), never measured as one
unified ratio by any Stage 2 item.

### Formal definition

For a fixed (class-pair, timestep) cell, pooled across its 20 draws --
**reusing `common/metrics.py`'s `magnitude_and_consistency` exactly, not
reimplementing a per-draw ratio.** That function's existing return
value (`magnitude`) already IS this ratio, computed with Item 1's own
pooling convention (pool numerator and denominator separately across
draws, then divide -- not a per-draw ratio averaged afterward, which is
a different, Jensen's-inequality-shifted quantity):

```
dilution_ratio(block_k, cell) = magnitude_and_consistency(feat_A_draws(block_k, cell), feat_B_draws(block_k, cell))[0]
                               = mean_i(||feat_B(i) - feat_A(i)||) / mean_i(||feat_A(i)||)
```

i.e. `conditioning_delta = mean_i(||feat_B(i) - feat_A(i)||)` (Item 1's
own cross-class delta, numerator) and `total_output_norm =
mean_i(||feat_A(i)||)` (class-A's own pooled output norm,
denominator) -- both already computed by the existing function, over
the same 20 draws each (class-pair, timestep) cell already uses.

This yields one `dilution_ratio` value per block per cell (5 pairs x 3
timesteps = 15 cells), matching Item 1's own per-cell magnitude
granularity exactly -- the quantity the Wilcoxon test below operates
on paired across cells, not across raw draws.

### Reused modules

- `common/hooks.py` -- `register_layer_hooks`, reused as-is (identical
  hook points to Item 1).
- `common/metrics.py` -- `magnitude_and_consistency`, reused as-is,
  **unmodified** -- its existing `magnitude` return value is
  `dilution_ratio` directly. No new metric function needed.
- `common/utils.py` -- `class_pairs`, `K_DRAWS`, `TIMESTEPS`, reused
  as-is (same design as Items 1/3).
- `common/io.py` -- checkpoint/config loading, reused as-is.

### Decision rule (locked)

**Statistical test:** Wilcoxon signed-rank, paired, `dilution_ratio`
at block 1 vs. block 6 across the pooled n=15 (5 pairs x 3 timesteps)
cells -- same paired-design convention as Items 1 and 3.

**SUPPORTED** if both hold:
- `p < 0.05` (null: no systematic block1-vs-block6 difference), AND
- net relative decline from block 1 to block 6 >= 30%,
  i.e. `(dilution_ratio(block_1) - dilution_ratio(block_6)) / dilution_ratio(block_1) >= 0.30`
  (pooled means, block-1-block, not per-draw ratio-of-ratios -- same
  pool-first-then-ratio convention as Item 3 Finding 3).

**NOT SUPPORTED** if any of:
- net decline < 10%, OR
- `p >= 0.05`, OR
- the ratio net-increases from block 1 to block 6.

**INCONCLUSIVE** for anything in between (e.g. 15-25% decline with
`p < 0.05`) -- reported as inconclusive rather than forced into a
binary call, consistent with this project's standing rule never to
manufacture false precision.

**Tolerance for non-monotonicity:** a single non-monotonic intermediate
block (e.g. a mid-sweep bump, as Item 3 itself found at block 6) does
not by itself invalidate a SUPPORTED verdict -- the decision rule is
evaluated strictly on the block-1-vs-block-6 endpoints, matching Item
3's own precedent that this architecture does not always produce clean
monotonic curves.

## Task 0.2 -- `final_norm`/`unproj` causal check

### Architectural question

Does the `final_norm` -> `unproj` stage (the two layers immediately
downstream of the last Transformer block, `step04_transformer_diffusion.py:226,229`)
disproportionately suppress the conditioning-specific component of the
signal relative to the whole-tensor signal, as it passes through? This
tests whether a fix targeting `final_norm`/`unproj` specifically (as
opposed to the gain-focused candidates motivated by Task 0.1) is
architecturally justified.

### Method

Directly reuses Item 3 Finding 3's **causal ablation** methodology
(override-hook substitution, not a pre-transform ratio) -- applied one
stage further downstream than Item 3's own block-6-ablation check.

1. Run a forward pass to just before `final_norm` (post-block-6
   tokens), for both class A and class B, at a fixed (pair, timestep,
   draw) -- this reuses Item 1's own paired cross-class forward-pass
   setup.
2. **Conditioning-specific delta, before vs. after:**
   - `delta_before = ||tokens_B(pre-final_norm) - tokens_A(pre-final_norm)||`
   - `delta_after  = ||unproj(final_norm(tokens_B)) - unproj(final_norm(tokens_A))||`
   - `retention_ratio_conditioning = delta_after / delta_before`
3. **Whole-tensor signal, before vs. after (the ablation control):**
   using the causal-ablation pattern from Item 3 Finding 3 (override
   block 6's residual update with its own input, i.e. skip block 6's
   contribution entirely, then measure the resulting change through
   `final_norm` -> `unproj` vs. the unablated path) -- but here applied
   to class A alone (single-class, not cross-class), to isolate how
   much of the **overall** (non-conditioning-specific) signal survives
   the same two layers:
   - `delta_before_whole = ||tokens_A(pre-final_norm) - tokens_A_ablated_block6(pre-final_norm)||`
   - `delta_after_whole  = ||unproj(final_norm(tokens_A)) - unproj(final_norm(tokens_A_ablated_block6))||`
   - `retention_ratio_whole = delta_after_whole / delta_before_whole`
4. Pool both ratios (mean across the 5 pairs x 3 timesteps x 20 draws
   design, pool-first-then-ratio, same convention as Item 3 Finding 3)
   before comparing.

**Interpretation note (stated now, before data exists, per this
project's standing discipline):** this operational definition of
"whole-tensor signal" reuses Item 3's own block-6 ablation as the
whole-tensor perturbation source, so that both ratios are measured via
the same causal-ablation mechanism rather than mixing a correlational
whole-tensor measure with a causal conditioning-specific one. This is
an explicit interpretation choice, not the only possible operational
definition of "overall signal retention" -- flagged here rather than
silently assumed to be the unique correct reading of the user's
directive.

### Decision rule (locked)

**Implicates `final_norm`/`unproj` as a fix target** if:

```
retention_ratio_conditioning < 0.5 * retention_ratio_whole
```

i.e. the conditioning-specific component is retained through these two
layers at less than half the rate the whole-tensor signal is.

**Rules out `final_norm`/`unproj`** otherwise (`retention_ratio_conditioning
>= 0.5 * retention_ratio_whole`) -- the two layers are not
disproportionately suppressing conditioning relative to the overall
signal, even if both decline in absolute terms.

## Reporting (both tasks)

Findings are appended as a **dated addendum to the existing
`Stage2_Decision_Report.md`** (Sec. 4, after Conclusion 5b) -- not a
new standalone report file, per `Stage3_Roadmap.md` Sec. 3.

## Runtime / compute

**CPU-only, expected 0 GPU hours**, per `Stage3_Roadmap.md` Sec. 3's
stated expectation (based on Item 3's own CPU-only ablation precedent
for the same causal-ablation mechanism, now applied one layer further
downstream). If this expectation is wrong for either task, that is a
stop condition, reported explicitly, not silently absorbed by
switching to GPU.

## Lock

This pre-registration is frozen once written. Both tasks' decision
thresholds are locked per explicit user directive for this pass, as
stated above -- any threshold that data reveals to be miscalibrated is
reported as a limitation in the eventual addendum, not silently
adjusted post hoc.
