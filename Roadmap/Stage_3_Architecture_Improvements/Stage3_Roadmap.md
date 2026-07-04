# Stage 3 Roadmap -- Architecture Improvements

**Status: structure only, drafted from scratch this session.** No prior
"Stage 3 roadmap" existed anywhere in this repository or its git
history before this document -- confirmed by direct search, not
assumed, after a planning artifact referencing one turned out not to
exist. This file is the real, first, tracked version.

**Numeric thresholds for Phase 0's decision rules are NOT included
here** -- they are a separate, subsequent pre-registration step
(`Stage3_Phase0_PreRegistration.md`, not yet written), reviewed against
this roadmap's actual structure rather than decided in the abstract
before the structure existed.

## 1. Purpose

Stage 2 investigated *why* class conditioning weakens across this
diffusion model's Transformer blocks. Stage 3's job is different in
kind, not just in scope: **produce a measurably improved model**, not
additional diagnosis. Stage 2 optimized for certainty (multi-cycle
review, statistical validation at every step); Stage 3 should optimize
for learning-rate-per-GPU-hour -- fewer intermediate-analysis cycles,
more implement/train/evaluate iterations, with investigation reserved
for (a) the one cheap validation pass before committing to a design
direction (Phase 0) and (b) explaining a result that will actually ship
(Phase 5), not results in between.

## 2. Current evidence (imported from Stage 2, not re-derived)

Only claims that survived Stage 2's verification are listed. Full
detail, per-item citations, and the complete confidence calibration
live in `../Stage_2_Architecture_Investigation/Reports/Stage2_Evidence_Matrix.md`
and `../Stage_2_Architecture_Investigation/Reports/Stage2_Decision_Report.md`
-- this table does not replace either, it is a pointer.

| Finding | Confidence | Source |
|---|---|---|
| Conditioning magnitude/influence attenuates across the network, and is recoverable via gain correction | High | Items 1, 2A, 2B |
| Class-conditioning information is preserved/decodable at every block (never information-theoretically lost) | High | Item 8 |
| Blocks 5-6 are architecturally distinctive vs. blocks 1-4 | Moderate (strong on 1/3/5, softer on 6, explicitly absent on 8, Item 4 not designed to test this) | Items 1, 3, 5 (+6 suggestive; 8 dissents) |
| LayerScale-style gain correction restores block-6 conditioning magnitude on its own criteria | High | Items 2A, 2B |
| Class embedding received a non-negligible, above-median (but not dominant) gradient during training | High | Item 4 |
| AdaLN allocates weight capacity non-uniformly across blocks, and receives the largest gradient of any parameter-type bucket | High | Items 4, 5 |
| Attention is substantially class-blind network-wide (cross-attention not strongly indicated as a fix) | High (network-wide); Moderate (blocks-5/6 refinement) | Item 6 |
| Dilution mechanism (conditioning-delta / total-output-norm shrinking across blocks) | **Hypothesis only, Low-to-Moderate** -- synthesized from Items 1+3, never measured as a unified ratio by any item | Decision Report Conclusion 5b |
| Class-embedding evolution across training | Not testable -- BLOCKED, permanent (no per-epoch checkpoint history exists) | Item 7 |
| Localized-vs-uniform gain comparison | **Confounded** -- do not cite as an architectural preference | Item 2B |

**Two flagged limitations carried forward, not silently dropped:**
Item 2B's localized-vs-uniform comparison is confounded and must not
drive a distribution-vs-concentration design choice without a corrected
budget-matching methodology. The dilution mechanism is a hypothesis,
not an established finding -- Phase 0 Task 0.1 exists specifically to
test it before any gain-mechanism design assumes it's true.

## 3. Phase 0 -- Cheap validation (no retraining, no GPU)

Structure only in this document; numeric decision rules are
pre-registered separately, before Task 0.1/0.2 run.

### Task 0.1 -- Dilution-ratio test

- **Metric:** `conditioning_delta(block_k) / total_output_norm(block_k)`
  per block. Reuses Item 1's cross-class delta computation and Item
  3's output-norm computation from `common/` -- not reimplemented from
  scratch.
- **Design:** reuses the existing 5 class-pairs x 3 timesteps x 20
  draws design (same as Items 1/3), for comparability and cost.
- **Pre-registration required before running** (separate step): a
  numeric definition of "steadily drops" (e.g. net decline across
  blocks 1->6 exceeding some threshold, explicitly tolerant of Item
  3-style intermediate non-monotonicity -- a single non-monotonic point
  must not auto-falsify, since this architecture has already shown it
  doesn't always produce clean monotonic curves), the exact sample
  size/cells this is computed over, and what pattern would count as
  genuine disconfirmation vs. noise.

### Task 0.2 -- `final_norm`/`unproj` causal check

- **Method:** reuses Item 3's **ablation-based** methodology (not a
  new correlational approach) -- the same principle as the block-6
  causal ablation, applied one layer further downstream: does
  `final_norm`/`unproj` disproportionately affect the conditioning-
  specific component relative to the whole-tensor signal, tested by
  intervention, not just correlation.
- **Pre-registration required before running** (separate step): what
  result implicates this region as a fix target vs. rules it out.

### Reporting

Both tasks' findings are appended as **dated addenda to the existing
`Stage2_Decision_Report.md`** (Sec. 4, after Conclusion 5b) -- not left
to live only in Phase 0's own standalone output files. This preserves
the "every claim cites its source, centrally" discipline Stage 2
established.

### Decision Gate A

Two outcomes only, evaluated against the pre-registered rules:

- **Dilution supported** -> proceed toward gain-focused candidates in
  Phase 1 (LayerScale, late-block gain).
- **Dilution not supported** -> do not build architecture assuming
  dilution; if Task 0.2 instead implicates `final_norm`/`unproj`, that
  becomes a new candidate (not on the list below) and the gain-focused
  candidates are deprioritized.

## 4. Phase 1 -- Architecture implementation

**Candidate list is NOT fixed in advance.** It is explicitly
conditional on Phase 0's outcome -- this is a hard rule, not a
preference, precisely to avoid spending Phase 2's GPU budget on
candidates Phase 0's evidence didn't actually support.

| Candidate | Motivated by (if Phase 0 confirms dilution) |
|---|---|
| LayerScale (learnable per-block gain) | Items 1, 2A, 2B |
| Late-block gain (blocks 5-6 specifically) | Items 1, 3, 5 |
| Residual scaling | Item 3 |
| Hybrid (combination of the above) | Multiple findings |
| Baseline (no change, re-verify current numbers) | Control |
| *(6th candidate, name TBD)* -- `final_norm`/`unproj` modification | Only added if Task 0.2 implicates this region instead |

If Phase 0 refutes dilution and implicates `final_norm`/`unproj`
instead, none of the first five candidates touch that region -- the
6th candidate becomes the priority and the gain-focused candidates are
deprioritized or dropped, not built anyway "for completeness."

## 5. Phase 2 -- GPU training

Automated implement -> push -> pull -> train -> checkpoint cycle per
candidate. No analysis during this phase -- that's Phase 3/5's job.

## 6. Phase 3 -- Evaluation

Standardized per-candidate output bundle: `metrics.json`, `plots/`,
`summary.md`. Same structure for every candidate, so Phase 4's
comparison is apples-to-apples without per-candidate custom analysis.

## 7. Decision Gate B -- when to stop and analyze mid-cycle

Five pre-committed triggers (not a judgment call each time):

1. A candidate improves >=2 primary metrics over baseline.
2. Unexpected degradation vs. Stage 2's own hypotheses.
3. **Two metrics move in opposite directions** -- this is exactly the
   failure signature Item 2B's confound produced (uniform gain "won" on
   recovery% while efficiency quietly signaled a problem); codified
   here as a hard stop-and-look rule, not something to notice only in
   hindsight.
4. Training instability (divergence, NaN, collapse).
5. A result directly contradicts a Stage 2 verified finding (e.g., a
   candidate that somehow makes conditioning *less* decodable, which
   Item 8 found never happens at baseline).

**Kill policy** (applies within Phase 2/3, independent of Gate B):

| Condition | Action |
|---|---|
| Training diverges | Stop immediately |
| No improvement after a pre-specified epoch budget | Terminate early |
| Inferior on >=3 metrics vs. the current best | Discard candidate |
| Clearly better than current best | Continue full training |

Otherwise: implement -> train -> evaluate -> next candidate, no pause.

## 8. Phase 4 -- Architecture selection

One comparison table, one ranking, across all evaluated candidates.
Pick the winner(s); document discarded candidates as negative results
(same discipline as Stage 2's "what NOT to pursue" section), not
silently dropped.

## 9. Phase 5 -- Deep investigation (gated on a clear winner)

Only once Phase 4 has a clear winner: investigate *why* it works, at
the same rigor level Stage 2 used. Explaining a result that will ship
is worth the cost; investigating results before a downstream decision
depends on them is exactly the overhead Stage 2 spent extra cycles on
that Stage 3 should not repeat.

## 10. Time allocation (directional, not exact)

Stage 2 was roughly Investigation 70% / Engineering 20% / GPU 5% /
Documentation 5%. Stage 3 shifts to roughly **Investigation 15% /
Engineering 45% / GPU Training 30% / Documentation 10%** -- the shift
itself matters, not the precise numbers.

## 11. Operating mode

Optimize for engineering throughput while preserving scientific
traceability. Every completed Phase 1-4 cycle should produce one of:
a trained checkpoint, a rejected architecture (documented, not
discarded silently), or a promoted architecture. Prefer implementation,
GPU training, and standardized evaluation over lengthy intermediate
analysis. Stop only when a predefined decision gate (A or B) triggers,
or a result contradicts a Stage 2 verified finding.

## Next steps

1. Review this structure.
2. Pre-register Task 0.1/0.2's numeric decision rules
   (`Stage3_Phase0_PreRegistration.md`) as a separate step.
3. Only then does Phase 0 execute.
