# Stage 2 / Tier 0 -- Progress Summary

Companion to `STAGE2_STATUS.md` (which tracks provenance/commits). This
file tracks the *scientific* thread across items -- hypothesis, verdict,
confidence, confounds, and dependencies -- so writing the eventual
`Stage_2_Decision_Report.md` doesn't require re-extracting conclusions
and caveats from nine separate reports by hand.

| Item | Hypothesis | Status | Verdict | Confidence | Confounds | Depends on |
|---|---|---|---|---|---|---|
| 1 | Layer-wise conditioning magnitude decays block1->6 while direction consistency stays high | Complete | SUPPORTED (two-drop shape: dominant block1->2, smaller real block5->6) | High | None | -- |
| 2A | Localized gain correction (block1->2 only) recovers block-6 magnitude while preserving direction | Complete | SUPPORTED (driven by g=3.0) | High | None (own verdict is clean) | Item 1 |
| 2B | Uniform gain correction (blocks 1-5, budget-matched) recovers block-6 magnitude while preserving direction | Complete | SUPPORTED (driven by nominal_gain=1.5, own criteria) | High (own verdict); Medium (comparison vs. 2A) | **Comparison with 2A is CONFOUNDED** -- budget-matching formula does not hold actual injected magnitude equal under nonlinear compounding (ratio grows ~0.6-1.4x at g=1.25 to ~2-3.6x at g=5.0). "Uniform beats localized" is NOT a clean architectural finding without this qualifier -- see `Reports/Item2B_Report.md` Sec. 3. | Item 1 |
| 3 | `R_k` (within-pass residual-update ratio) varies systematically across blocks 1-6 (direction-neutral -- attenuation/amplification/non-monotonic all legitimate outcomes) | Complete | SUPPORTED (Wilcoxon block1-vs-block6, p=1.67e-06) -- shape is **non-monotonic**: U-shape, valley at block 3 (0.0947), spike at block 6 (0.7616, ~8x the valley). Class-independence finding (R_k barely differs A vs. B) corroborates Item 1's "signal lives in direction not magnitude" from an independent angle. Block 6's outsized post-normalization influence CONFIRMED by causal ablation (3.24x larger final_norm-output change than ablating the valley block) -- an earlier pre-normalization-ratio-only version of this claim was caught and corrected before locking. | High | None -- the non-monotonic shape does not contradict Item 1's two-drop shape; the two measure different quantities (within-pass update vs. cross-class output delta), stated explicitly as complementary, not required to match in shape. | None (complementary to Item 1/2, not derived from them -- confirmed in pre-registration: Item 3's result would not change if Item 1/2's numbers changed) |
| 4 | Class-embedding gradient competitiveness (grad norm at class_emb.weight vs. other param groups; epoch-25 comparison permanently unavailable, see pre-registration) | In progress | Preliminary (unverified): percentile rank ~61% among 95 other tensors, secondary fixed-timestep design shows declining rank as timestep increases (t=100: 54.7% -> t=900: 33.7%) -- matches Item 1's own sensitivity-probe direction (weaker conditioning effect at high noise) but NOT YET LOCKED pending a clean GPU rerun (script had a stale JSON key and missing device/repro-diff logging, fixed in commit 36a5f0c) | Low (pending verification) | None known yet -- verdict not locked | Item 1 (motivation only, not dependency) |
| 5 | AdaLN/FiLM parameter statistics (per-block weight-matrix Frobenius norm, scale vs. shift allocation) | Complete | VERIFIED -- non-uniform allocation (range 0.2079 across blocks). Largest scale_fraction jump (block1->2, +12.4pp) and only decrease (block5->6, -5.5pp) correlate with Item 1's two-drop transitions -- stated as correlation, not causation. | High (n=1, exact deterministic weight property, no sampling variance) | None | Item 1 (cross-validation target, not a dependency -- Item 5's result would not change if Item 1's numbers changed) |
| 6 | Attention entropy/map inspection (class-blind attention test; master prompt's "STEMI" corrected to MI, real taxonomy has no STEMI class) | Complete | VERIFIED network-wide (pooled entropy diff 0.0352 < 0.05) -- attention substantially class-blind. Refinement: blocks 5-6 individually exceed threshold (0.064, 0.055) -- 4th independent measurement (after 1/3/5) converging on those blocks as distinctive. | High (network-wide verdict); the blocks-5/6 refinement is a small-magnitude effect (~0.8-0.9% of max entropy) -- real but modest | None | Item 1 (cross-validation target, not a dependency) |
| 7 | Class-embedding evolution across training checkpoints | Not started | -- | -- | -- | -- |
| 8 | Representation collapse analysis (Fisher ratio + linear probe, per block) | Not started | -- | -- | -- | -- |

**How to apply:** before starting any new item, check this table's
"Depends on" column against the item's stated hypothesis in
`Stage2_Master_Prompt.md` -- if a dependency is on a confounded verdict
(like Item 2B's comparison), that must be resolved (restated hypothesis
or explicit carried-forward qualifier) before pre-registration, not
discovered afterward.
