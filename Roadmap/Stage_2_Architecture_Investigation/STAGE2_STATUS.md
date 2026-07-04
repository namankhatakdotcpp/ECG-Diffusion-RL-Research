# Stage 2 / Tier 0 -- Status

Tracks each Tier 0 item's status, verdict, and provenance. **Pre-reg
commit** = when the item's falsification criteria/design were locked in
writing (before code). **Results commit** = when the item's actual run
output + report were committed. These are deliberately kept as two
separate columns -- conflating "when we decided what to test" with "when
we ran it" is exactly the ambiguity Stage 1's dirty-tree finding already
flagged as a real provenance risk in this project.

| Item | Status | Verdict | Pre-reg commit | Results commit | Report | Archive |
|---|---|---|---|---|---|---|
| Item 1 -- Layer-wise magnitude/direction probe | Complete | Confirmed two-drop shape (dominant block1->2, smaller real block5->6) | -- | `07460c0` | `Reports/Tier0_Findings.md` | N/A -- ran locally (CPU forward passes against the already-extracted Stage 1 checkpoint), no GPU round-trip, nothing archived to tar.gz |
| Item 2A -- Localized gain (block1->2 only) | Complete | SUPPORTED (driven by g=3.0) | `e84c54c` | `1bb3062` | `Reports/Item2_Report.md` | N/A -- same reason as Item 1 (CPU-only, no GPU round-trip) |
| Item 2B -- Uniform gain (blocks 1-5) | Complete | SUPPORTED on its own criteria (driven by nominal_gain=1.5, the smallest grid point clearing the threshold). Qualifier: "Recovery advantage of uniform over localized is confounded by budget-matching breakdown under nonlinear compounding -- see Item2B_Report.md Sec. 3 before citing this as an architectural preference." | `e84c54c` | `6778f03` | `Reports/Item2B_Report.md` | N/A -- same reason as Item 1/2A |
| Item 3 -- Residual-path attenuation | Complete | SUPPORTED (Wilcoxon block1-vs-block6, p=1.67e-06) -- non-monotonic U-shape (valley block 3, spike block 6). Class-independence corroborates Item 1 from an independent angle. Block 6's outsized post-normalization influence confirmed by causal ablation (3.24x); an earlier pre-normalization-ratio-only version of that claim was caught and corrected before locking -- see `Reports/Item3_Report.md` Finding 3. | `3101f28` | `3847f96` | `Reports/Item3_Report.md` | N/A -- same reason as Item 1/2A/2B |
| Item 4 -- Gradient competitiveness | Complete | VERIFIED -- `class_emb.weight` gradient sits at the 61.1th percentile among 95 other tensors during real training-mode backward passes (bootstrap 95% CI [58.95%, 62.11%], n=1000 resamples over 30 real draws) -- competitive, not dominant. Weight-checksum, zero-grad, and bucket-reconciliation checks all confirmed with real hash/numeric values (not PASS strings). Secondary fixed-timestep design cross-validates with Item 1's own forward-pass magnitude decline (both weaken as timestep increases: 56.8%/38.9%/33.7% rank vs. 0.173/0.124/0.107 magnitude at t=100/500/900) -- see `Reports/Item4_Report.md`. | `546ec26` | `2e50b98` (script fixes) / *(this commit)* (report) | `Reports/Item4_Report.md` | N/A -- CUDA run confirmed (NVIDIA RTX A6000, CUDA 11.8), no persistent archive kept locally beyond the transferred `item4_results.tar.gz` |
| Item 5 -- AdaLN/FiLM parameter statistics | Complete | VERIFIED -- non-uniform scale/shift allocation across blocks (range 0.2079). Largest scale_fraction jump (block1->2, +12.4pp) and only decrease (block5->6, -5.5pp) correlate with Item 1's two-drop transitions (stated as correlation, not causation) -- see `Reports/Item5_Report.md`. | *(this commit)* | *(this commit)* | `Reports/Item5_Report.md` | N/A -- CPU-only, weight inspection, no forward pass, no GPU |
| Item 6 -- Attention entropy/map inspection | Complete | VERIFIED network-wide (pooled mean entropy diff 0.0352 < 0.05 threshold) -- attention is substantially class-blind. Blocks 5-6 show directionally higher class-dependence than blocks 1-4 in point estimates (0.064, 0.055 vs. 0.016-0.031), the 4th measurement type converging in DIRECTION with Items 1/3/5. **Post-review correction: 95% CIs on blocks 5/6 cross the 0.05 threshold (computed across n=15 pooled cells) -- the specific "exceeds threshold" claim is directionally suggestive, not statistically confirmed; blocks 1-4's CIs stay entirely below 0.05.** Correction applied before running: master prompt's "STEMI" class doesn't exist in the real taxonomy, substituted MI -- see `Reports/Item6_Report.md`. | *(this commit)* | *(this commit)* | `Reports/Item6_Report.md` | N/A -- CPU-only, synthetic-noise probe (same cost profile as Item 1/3), no GPU |
| Item 7 -- Class-embedding evolution across training | **BLOCKED (permanent)** | No verdict possible -- requires multiple per-epoch checkpoints from the same training run; none exist (same root cause as Item 4's epoch-25 limitation: `save_every=25` created them, `KEEP_LAST_N_CHECKPOINTS=2` pruned them, confirmed active in this run's commit lineage). Confirmed by filesystem + archive search, not assumed. See `Reports/Item7_PreRegistration.md`. | *(this commit)* | N/A -- no experiment ran | `Reports/Item7_PreRegistration.md` | N/A -- blocked before execution |
| Item 8 -- Representation collapse (Fisher ratio + linear probe) | Complete | VERIFIED at every block/timestep -- linear-probe accuracy is 100% everywhere (zero train/test gap, not the interpolation-regime artifact), even where Fisher ratio declines substantially (up to 11.6x, block1 vs block6 at t=900). **Post-review statistical validity check (n_train<=d at every cell, label-permutation control, 3 independent splits): finding SURVIVES -- permutation-test accuracy stays near chance (0.083-0.278) with permutation train accuracy only ~0.40-0.57 (nowhere near 1.0), directly ruling out the memorization/dimensionality-artifact concern.** Refines rather than contradicts Item 1. This is the first measurement type that does NOT single out blocks 5-6 (probe accuracy is flat everywhere) -- see `Reports/Item8_Report.md`. | *(this commit)* | *(this commit)* | `Reports/Item8_Report.md` | N/A -- CPU-only, synthetic-noise probe, no GPU |
**Tier 0 stops at Item 8** (`Stage2_Master_Prompt.md:18`, "Tier 0 (items
1-8)"). Items 9-11 are Tier 1 -- real architecture changes, Stage 3's
job, logged under `Stage_3_Architecture_Improvements`, not tracked in
this file. An earlier revision of this table carried a stale "Item 9 |
Not started" row, incorrectly implying Item 9 belongs to Stage 2's own
Tier 0 -- removed here (Phase E internal-consistency check) rather than
left standing.

Item 1 has no separate pre-registration commit: it was Stage 1's own
Experiment 3.5, reused here (per its own docstring, "DERIVED COPY, not an
independent implementation") rather than freshly pre-registered under
Stage 2's later "no code until it's in writing" discipline, which began
with Item 2.

**Infrastructure:** `Code/common/` (hooks.py, metrics.py, statistics.py,
plotting.py, io.py, utils.py) holds the shared hook mechanisms,
magnitude/direction-consistency formulas, decision-table logic, and
plotting code, extracted from Item 1 + Item 2A after both were already
complete and committed. Item 1 and Item 2A's own scripts and outputs are
left untouched as the historical record; only Item 2B onward import from
`common/` by default. The extraction was verified bit-identical against
Item 2A's committed `sweep_summary.json` before this file existed (max
diff = 0.0 across every reported metric).
