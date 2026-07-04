# Stage 3 Roadmap -- Architecture Improvements

**Status: structure only.** No prior "Stage 3 roadmap" existed
anywhere in this repository or its git history before this document --
confirmed by direct search, not assumed, after a planning artifact
referencing one turned out not to exist. This is the real, tracked
version, revised once after review (time/scope budgets on Phase 0 and
on infrastructure-building, parallel implementation track, experiment
IDs, quantitative kill-policy form, baseline-comparison protocol).

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

**Time/scope budget (added on review):** Phase 0 is a validation check,
not a new investigation cycle. Stage 2's own history (Item 2 alone went
through three pre-registration revisions before any code ran) shows
investigation expands unless bounded. Cap: **target 2-3 days elapsed,
new code additions kept small** (extending `common/` and two analysis
scripts, not new infrastructure) for both tasks combined. If either
task is trending past this, that itself is a signal to stop and
re-scope, not push through.

**GPU:** *Expected 0 hours*, based on Item 3's CPU-only ablation
precedent (Task 0.2 reuses that exact methodology). This is stated as
an expectation to confirm before running, not an absolute ceiling --
if `final_norm`/`unproj` ablation turns out to need GPU-scale batching
that Item 3's original didn't, that must be surfaced explicitly rather
than silently violated or used to block a legitimate check.

**Parallel implementation track (Track A/B):** Phase 1's candidate
*implementation code* (LayerScale, residual scaling, etc.) has no
actual dependency on Phase 0's result -- only *which candidates get
trained* in Phase 2 depends on it. Writing this code in parallel with
Phase 0's validation is free parallelism: worst case some of it goes
unused, best case it saves real days once Decision Gate A resolves.
Track A = Phase 0 validation; Track B = candidate implementation,
running concurrently, not gated on each other.

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

**Experiment IDs and queue:** each candidate run gets a stable ID
(`S3-001`, `S3-002`, ...) assigned at Phase 1, used consistently
through Phase 2 training, Phase 3 evaluation, and Phase 4's comparison
table -- so a result can always be traced back to its exact candidate
definition and commit, the same provenance discipline Stage 2 applied
per-item. Runs are worked in a simple queue (pending -> running ->
done/killed), not a scheduler or dashboard.

**Infrastructure budget:** the queue and any shared evaluation
automation get their own cap -- **target 1-2 days, reuse existing
`mentor_eval/run_all.py` rather than building a new harness** (see
Sec. 6). This is the same principle as Phase 0's budget: pipeline-
building is exactly the kind of overhead Stage 3 exists to avoid, so
it does not get an open-ended allowance just because it's
"infrastructure" rather than "investigation."

**Directory layout:** `Code/stage3_candidates/<S3-XXX>_<name>/` per
candidate (implementation + training script), `Results/<S3-XXX>/`
per run (checkpoint pointer, `metrics.json`, `plots/`, `summary.md`) --
mirrors Stage 2's per-item `Code/stage2_tier0_itemN_.../` convention.

## 5. Phase 2 -- GPU training

Automated implement -> push -> pull -> train -> checkpoint cycle per
candidate. No analysis during this phase -- that's Phase 3/5's job.

## 6. Phase 3 -- Evaluation

Standardized per-candidate output bundle: `metrics.json`, `plots/`,
`summary.md`. Same structure for every candidate, so Phase 4's
comparison is apples-to-apples without per-candidate custom analysis.
**This bundle is produced by running the existing `mentor_eval/run_all.py`
pipeline** (already produces the 4-class accuracy/AUC/similarity
numbers cited throughout this project) against each candidate's
checkpoint -- not a new evaluation harness built for Stage 3.

### Baseline-comparison protocol

Every candidate is judged against one fixed reference point, not a
number that drifts per comparison:

- Baseline metrics come from **one frozen, checksum-verified run** of
  `mentor_eval/run_all.py` against the current `diffusion_best.pt` --
  recorded once, reused for every candidate, not silently
  re-measured differently each time.
- Every candidate is evaluated with the **identical evaluation code
  path** as the baseline -- no per-candidate custom evaluation logic,
  same reasoning as Item 2's "compare against the immediately
  preceding, verified baseline" discipline.
- This matters because a checkpoint that "looks fine" can hide bugs
  (Stage 1/2's own history includes an EMA bug and a dirty-tree
  finding found only by re-verification) -- Decision Gate B's
  "candidate improves on baseline" trigger is only as trustworthy as
  that baseline's own provenance.

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

**Kill policy** (applies within Phase 2/3, independent of Gate B --
quantitative form; exact N is set in pre-registration, not here):

| Condition | Action |
|---|---|
| Training diverges (loss NaN/Inf, or explodes past a pre-set bound) | Stop immediately |
| No improvement for N consecutive evaluations (plateau) | Terminate early |
| Statistically worse than baseline on the primary metric | Discard candidate |
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

## 10. Time allocation and milestones

Stage 2 was investigation-heavy (~70% investigation); Stage 3 shifts
decisively toward engineering and GPU training instead, since the job
is now building and testing candidates rather than diagnosing the
existing model -- this is a one-time calibration signal for the pivot,
not a tracked KPI, so no percentage table is maintained here.

| Milestone | Target |
|---|---|
| Phase 0 (Tasks 0.1/0.2) + Decision Gate A | within budget in Sec. 3 |
| Track B candidate implementations ready | in parallel with Phase 0 |
| Phase 2 GPU training, all candidates | after Gate A resolves |
| Phase 3 evaluation bundles, all candidates | rolling, as each finishes training |
| Phase 4 selection | after last candidate's Gate B resolves |
| Phase 5 (if triggered) | after Phase 4 |

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
