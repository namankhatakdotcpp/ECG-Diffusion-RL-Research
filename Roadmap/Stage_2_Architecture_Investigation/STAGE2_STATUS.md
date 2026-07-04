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
| Item 3 | Not started | -- | -- | -- | -- | -- |
| Item 4 | Not started | -- | -- | -- | -- | -- |
| Item 5 | Not started | -- | -- | -- | -- | -- |
| Item 6 | Not started | -- | -- | -- | -- | -- |
| Item 7 | Not started | -- | -- | -- | -- | -- |
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
