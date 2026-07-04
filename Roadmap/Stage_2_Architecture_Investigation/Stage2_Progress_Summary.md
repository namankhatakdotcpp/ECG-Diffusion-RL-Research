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
| 4 | Class-embedding gradient competitiveness (grad norm at class_emb.weight vs. other param groups, early vs. late checkpoint) | Not started | -- | -- | -- | -- |
| 5 | AdaLN/FiLM parameter statistics (per-block weight-matrix Frobenius norm, scale vs. shift allocation) | Not started | -- | -- | -- | -- |
| 6 | Attention entropy/map inspection (class-blind attention test) | Not started | -- | -- | -- | -- |
| 7 | Class-embedding evolution across training checkpoints | Not started | -- | -- | -- | -- |
| 8 | Representation collapse analysis (Fisher ratio + linear probe, per block) | Not started | -- | -- | -- | -- |

**How to apply:** before starting any new item, check this table's
"Depends on" column against the item's stated hypothesis in
`Stage2_Master_Prompt.md` -- if a dependency is on a confounded verdict
(like Item 2B's comparison), that must be resolved (restated hypothesis
or explicit carried-forward qualifier) before pre-registration, not
discovered afterward.
