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
| Item 4 -- Gradient competitiveness | In progress -- GPU run completed once, two data-quality issues found and fixed in the script (stale JSON key, missing device/repro-diff logging); awaiting a clean rerun before verdict is locked. Preliminary (unverified) percentile rank: ~61%. | `36a5f0c` | -- (pending clean rerun) | `Reports/Item4_PreRegistration.md` | Pending -- `item4_results.tar.gz` (GPU-side, not yet finalized) |
| Item 5 -- AdaLN/FiLM parameter statistics | Complete | VERIFIED -- non-uniform scale/shift allocation across blocks (range 0.2079). Largest scale_fraction jump (block1->2, +12.4pp) and only decrease (block5->6, -5.5pp) correlate with Item 1's two-drop transitions (stated as correlation, not causation) -- see `Reports/Item5_Report.md`. | *(this commit)* | *(this commit)* | `Reports/Item5_Report.md` | N/A -- CPU-only, weight inspection, no forward pass, no GPU |
| Item 6 -- Attention entropy/map inspection | Complete | VERIFIED network-wide (pooled mean entropy diff 0.0352 < 0.05 threshold) -- attention is substantially class-blind. Refinement: blocks 5-6 individually exceed the threshold (0.064, 0.055), the 4th independent measurement type (after Items 1/3/5) to flag those blocks as distinctive. Correction applied: master prompt's "STEMI" class doesn't exist in the real taxonomy, substituted MI -- see `Reports/Item6_Report.md`. | *(this commit)* | *(this commit)* | `Reports/Item6_Report.md` | N/A -- CPU-only, synthetic-noise probe (same cost profile as Item 1/3), no GPU |
| Item 7 -- Class-embedding evolution across training | **BLOCKED (permanent)** | No verdict possible -- requires multiple per-epoch checkpoints from the same training run; none exist (same root cause as Item 4's epoch-25 limitation: `save_every=25` created them, `KEEP_LAST_N_CHECKPOINTS=2` pruned them, confirmed active in this run's commit lineage). Confirmed by filesystem + archive search, not assumed. See `Reports/Item7_PreRegistration.md`. | *(this commit)* | N/A -- no experiment ran | `Reports/Item7_PreRegistration.md` | N/A -- blocked before execution |
| Item 8 | Not started | -- | -- | -- | -- | -- |
| Item 9 | Not started | -- | -- | -- | -- | -- |

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
