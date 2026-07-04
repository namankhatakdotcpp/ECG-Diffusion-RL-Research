# Stage 2 / Tier 0 -- Evidence Matrix

Maps each cross-cutting claim tested across Stage 2's Tier 0 to every
item that produced evidence bearing on it. Built to prevent narrative
bias in the eventual `Stage_2_Decision_Report.md` -- makes explicit how
(and whether) each conclusion is actually supported across independent
experiments, rather than letting a single item's framing carry into the
synthesis unchallenged.

**Legend:** Supports / Weakly supports / Neutral / Contradicts /
Confounded / Not testable / Pending (verification incomplete).

**All 8 Tier 0 items are now closed** (7 VERIFIED/SUPPORTED, 1 BLOCKED
permanent -- Item 7). Item 4 closed as VERIFIED (`Reports/Item4_Report.md`,
bootstrap-confirmed, real checksum values). This matrix is ready to
feed `Stage_2_Decision_Report.md`.

## Evidence matrix

**Confidence calibration** (kept coarse deliberately -- a finer scale
would imply false precision given how heterogeneous these claims'
evidence bases are): **High** = statistically verified and/or
permutation/ablation-confirmed, low ambiguity; **Moderate** = real
directional signal but a stated statistical or scope caveat (wide CI,
small n, correlation-not-causation); **Pending** = verification
incomplete (Item 4 only); **Low** = single weak/indirect signal, no
independent corroboration; **N/A** = claim not tested by this row.

| Claim | Item 1 | Item 2A | Item 2B | Item 3 | Item 4 | Item 5 | Item 6 | Item 7 | Item 8 | Confidence |
|---|---|---|---|---|---|---|---|---|---|---|
| Conditioning magnitude/influence attenuates through the network | **Supports** (two-drop shape, dominant block1->2) | Supports (recovers magnitude when injected) | Supports (own criteria) | **Supports, refines** (non-monotonic U-shape, not simple decay) | -- | Weakly supports (scale/shift reallocation correlates with attenuation points) | Weakly supports (entropy declines block1->6, separate from class-blindness) | Not testable | **Contradicts a magnitude=information equivalence reading** (Fisher declines but linear separability never collapses) -- refines, does not contradict Item 1's own claim | **High** (Item 1's own claim is Wilcoxon-verified; Items 2A/2B/3 causally confirm magnitude is recoverable/real) |
| Class-conditioning information is preserved/decodable throughout the network | Neutral (measures magnitude/direction, not decodability) | Neutral | Neutral | Neutral | -- | Neutral | Neutral | Not testable | **Supports** (100% linear-probe accuracy at every block, permutation-verified) | **High** (single item, but permutation-verified -- memorization directly ruled out) |
| Blocks 5-6 are architecturally distinctive vs. blocks 1-4 | **Supports** (secondary drop at block5->6) | Neutral | Neutral | **Supports** (residual-ratio spike at block 6, causally confirmed) | **Not testable at block granularity** (Item 4 buckets gradients by parameter TYPE -- adaLN/attention/ffn/norms/embeddings/projection -- not by block index; it cannot speak to this claim one way or the other, not because verification is incomplete but because the design never captured per-block resolution) | **Supports** (only scale_fraction decrease at block5->6) | **Weakly supports** (point estimates higher at 5/6; CIs cross the 0.05 threshold -- directionally consistent, not independently confirmed) | Not testable | **Does not corroborate** (linear-probe accuracy flat at 100% everywhere, no block-5/6 signal) | **Moderate** (strong on 1/3/5, statistically softer on 6, explicitly absent on 8, Item 4 not designed to test this -- real but not unanimous) |
| LayerScale-style gain correction can restore block-6 conditioning magnitude | Motivates the hypothesis | **Supports** (own criteria, g=3.0) | **Supports** (own criteria, nominal_gain=1.5) | Neutral | Neutral | Neutral | Neutral | Not testable | Neutral | **High** (both variants independently clear their own pre-registered decision tables) |
| Localized vs. uniform gain correction -- which is more effective | -- | Compared against | **Confounded** (budget-matching formula breaks down under nonlinear compounding; "uniform wins" not a clean finding without this qualifier) | -- | -- | -- | -- | -- | -- | **Low** (confounded by design; the raw comparison should not be cited as an architectural preference) |
| Class-embedding gradient was ever competitive during training | Motivates the hypothesis (independent signal) | -- | -- | -- | **Supports** (percentile rank 61.05% among 95 other tensors, bootstrap 95% CI [58.95%, 62.11%] -- above median, not dominant, not starved) | Weakly supports (adaLN receives largest gradient bucket, ~6.7x class_emb's mean) | -- | -- | -- | **High** (bootstrap-verified tight CI, real checksum/reproducibility values, cross-validated timestep trend against Item 1) |
| AdaLN allocates scale/shift capacity evenly across blocks | -- | -- | -- | -- | -- | **Contradicts (falsified as stated)** -- allocation is non-uniform, range 0.208 | -- | -- | -- | **High** (n=1 but exact, deterministic weight property -- no sampling variance to caveat) |
| Attention already varies by class label (cross-attention would not obviously help) | -- | -- | -- | -- | -- | -- | **Supports, network-wide** (pooled diff 0.035 < 0.05); **weakly contradicts at blocks 5-6** (point estimates exceed threshold, CI-uncertain) | -- | -- | **Moderate** (network-wide verdict solid; blocks-5/6 refinement has real but CI-crossing uncertainty) |
| Class embedding differentiates over the course of training | -- | -- | -- | -- | -- | -- | -- | **Not testable** (BLOCKED, permanent -- no per-epoch checkpoint history exists) | -- | **N/A** (no evidence exists or can exist for this training run) |

## Notes on specific cells

- **Item 2B row:** its own verdict (SUPPORTED) is listed separately from
  the localized-vs-uniform comparison row, which is explicitly marked
  **Confounded** -- per the standing project rule not to merge a clean
  per-variant verdict with a confounded cross-variant comparison into
  one hedged mark.
- **Item 7:** marked **Not testable**, not Neutral or Contradicts --
  the claim was never evaluated at all, for a documented, permanent
  reason (no data exists), not because the evidence was ambiguous.
- **Item 8's "does not corroborate" mark** on the blocks-5/6 row is
  reported as-is, post-validity-check -- the permutation control
  confirmed Item 8's underlying finding (100% accuracy is real, not a
  dimensionality artifact), so this is a confirmed non-convergence, not
  a provisional one awaiting further checking.
- **Item 4 closed VERIFIED** -- the 61.05% percentile rank is bootstrap-
  confirmed (95% CI [58.95%, 62.11%], n=1000 resamples over the real
  30-draw GPU run) and all sanity checks (zero-grad, weight-checksum,
  bucket-reconciliation) show real hash/numeric values, not PASS
  strings. Safe to cite in the Decision Report.
- **Item 4's "blocks 5-6" cell is marked Not testable at block
  granularity, not Pending or Neutral** -- its bucketing is by
  parameter TYPE, never by block index, so it structurally cannot
  speak to per-block claims. This is a scope limitation of the design,
  not an unresolved verification.

## Cross-item finding table: the blocks-5/6 pattern specifically

| Item | Evidence re: blocks 5-6 | Confidence |
|---|---|---|
| 1 | Conditioning magnitude shows a secondary, smaller, real drop at block5->6 (statistically confirmed, Wilcoxon) | High |
| 3 | Residual-update ratio spikes at block 6 (~8x the block-3 valley); causal ablation confirms outsized post-normalization influence (3.24x vs. control block) | High |
| 4 | Not applicable -- Item 4's gradient analysis buckets by parameter TYPE (adaLN/attention/ffn/etc.), never by block index, so it cannot speak to per-block claims by design (not a verification gap) | N/A |
| 5 | The ONLY scale_fraction decrease across all 6 blocks occurs at block5->6; also the largest single jump at block1->2 | High (n=1, exact deterministic weight property) |
| 6 | Point estimates for entropy class-dependence are highest at blocks 5-6, but 95% CIs (n=15 cells) cross the 0.05 threshold -- directionally consistent, not independently significant | Medium |
| 8 | Linear-probe accuracy is flat 100% at every block -- does NOT single out blocks 5-6 | High (permutation-verified) |

**Four of five block-testable items (1, 3, 5, 6) show directional
convergence on blocks 5-6; one (8) explicitly does not; Item 4 is not
designed to test block-level claims at all (N/A, not a gap).** This
should be reported in the eventual Decision Report as a real but
qualified convergence -- strong on Items 1/3/5, present but statistically
softer on Item 6, absent on Item 8 -- not smoothed into a uniform "all
evidence points to blocks 5-6" claim.
