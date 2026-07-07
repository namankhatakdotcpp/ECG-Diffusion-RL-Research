# Hausdorff Distance -- Implementation Report

Pre-registered 2026-07-07, per the mentor sync-up action item to add Hausdorff
distance (mentioned in the meeting transcript as "Hashtag Distance," confirmed
as a transcription artifact by the cleaner 07 July 2026 meeting writeup) to
the evaluation framework.

## Mathematical Definition

Each lead's signal is treated as an **unordered 1D point set of amplitude
values** (order-blind), not a `(time, amplitude)` 2D trajectory:

```
h(A, B) = sup_{a in A} inf_{b in B} |a - b|
H(A, B) = max(h(A, B), h(B, A))
```

computed independently per lead, then mean-aggregated across the 12 leads
for a single-number summary (`mentor_eval/hausdorff_distance.py`).

### Why amplitude-only, not `(time, amplitude)`

A `(time, amplitude)` 2D point-set framing requires a Euclidean distance
between points whose two coordinates are in incompatible units: time index
(range ~1000) and amplitude (range 8, given this project's z-score-then-
clip-to-`[-4, 4]` preprocessing, `config.yaml`'s `clip_range`). Making that
framing meaningful requires an explicit time/amplitude relative-scaling
constant, which has no precedent anywhere else in this pipeline --
Mahalanobis/Bhattacharyya operate on 48-dim summary statistics with no time
axis at all, and cosine similarity is a fixed-index dot product with no
explicit temporal-tolerance mechanism either. Amplitude-only avoids
introducing an unjustified hyperparameter and is unit-consistent by
construction.

### What it captures, and what it explicitly does not

Hausdorff distance is included as a worst-case amplitude deviation metric.
It complements cosine similarity, Mahalanobis distance, and Bhattacharyya
distance by quantifying the maximum amplitude mismatch between matched real
and generated ECGs. It is not intended to evaluate temporal alignment or
waveform morphology.

**Does not capture morphology or timing.** This is a real, stated
limitation, not an oversight -- see Verification Case 2 below, where it is
made concrete rather than left abstract.

### Nearest-neighbour pairing (methodology)

Generated ECGs are compared against their nearest-neighbour real ECG within
the same disease class (via Euclidean distance in the 12000-dim raw
waveform space) rather than random or index-aligned pairing
(`mentor_eval/hausdorff_distance.py`'s `matched_hausdorff`). This reduces
pairing bias -- comparing a generated sample against an arbitrary or
poorly-matched real sample would inflate distance values for reasons
unrelated to generation quality -- and maintains consistency with the
existing cosine similarity evaluation protocol in `similarity_metrics.py`,
which uses the same matching convention.

### Expected value range

Bounded in `[0, 8]` given the `[-4, 4]` clip range (worst case: one signal
pinned at each extreme). For a well-matched real/generated pair of the same
class, most per-lead values should be well under 1.0; values approaching
2-4 indicate a real amplitude-range mismatch worth flagging, not just
reporting as a bare number.

## Verification

Four pre-registered synthetic cases (`mentor_eval/test_hausdorff_distance.py`):

| Case | Expected | Actual | Result |
|---|---|---|---|
| 1. Identical signals | `0.0` | `0.0` | Match |
| 2. Constant circular time-shift (50 samples) | generically expected "nonzero" | `0.0` | **Does not match generic expectation -- see below** |
| 3. 1.5x amplitude scale | nonzero | `1.0165` | Match (order of magnitude sane, within `[0,8]`) |
| 4. Real vs. matched-std noise | large relative to case 1 | `0.5682` | Match |

**Case 2 does not match the naive expectation, and this was not silently
adjusted to force a match.** A circular time-shift does not change the SET
of amplitude values a signal takes, only when they occur; since the metric
is order-blind by design, `H(A,B) = 0` exactly for any pure time-shift. This
is the concrete, now-verified expression of the "does not capture
morphology or timing" limitation stated above -- **amplitude-only Hausdorff
will score a badly time-shifted generation as a perfect match.** If a
temporal-misalignment-sensitive metric is what's actually wanted, this
metric structurally cannot provide it.

Additional tests: bounded by `[0, 8]` (verified exactly at the pinned-extreme
case), raises on shape mismatch / wrong lead count / non-finite input
(matching this pipeline's existing convention rather than silently
skipping/imputing), symmetric, and the batch `matched_hausdorff()` (nearest-
neighbour matched, same matching convention as `matched_cosine_similarity`)
correctly returns `0.0` for real-vs-itself and a nonzero, discriminating
value for real-vs-noise.

13 tests total, all passing.

## Limitations

Hausdorff distance is computed over amplitude values only and therefore
measures worst-case amplitude deviation. It is intentionally not used as a
temporal or morphological similarity metric; those aspects are evaluated
separately through classifier performance, cosine similarity, Mahalanobis
distance, and waveform visualizations.

The reported Hausdorff value is a mean across 12 leads. Given this
project's Stage 3 finding that disease-discriminative failure concentrates
in specific frequency subbands and is most visually apparent in Lead V1, a
per-lead breakdown (rather than a single averaged value) may reveal
amplitude-range anomalies this aggregate metric obscures. Recommend
per-lead reporting as a follow-up if Hausdorff is retained beyond this
initial deliverable -- `compute_hausdorff_per_lead()` already exists in
`hausdorff_distance.py` for this purpose, it just isn't wired into the
disease-wise table's output yet.

## Recommendation for Future Work

Given Stage 2's finding that disease-signal loss concentrates in
conditioning dilution across transformer blocks (Phase 0 Task 0.1) and the
`final_norm`/`unproj` boundary (Task 0.2) -- mechanisms that plausibly
affect signal timing/morphology, not just amplitude range -- Dynamic Time
Warping (DTW) or Fréchet distance would more directly measure
temporal/morphological similarity than amplitude-only Hausdorff can.
Flagged here as a recommendation for Dr. Balaji's consideration in a future
evaluation cycle, not a substitution for the requested Hausdorff metric --
the mentor explicitly named Hausdorff distance (confirmed twice: the ASR
transcript and the cleaner meeting writeup), so this report delivers what
was asked, with the limitation stated plainly, rather than silently
substituting a different metric because it may be scientifically preferable.

## 4th Metric -- Not Confirmed

The 2026-07-07 sync-up document's table template lists a 4th metric column
as "to be finalized as per project requirements, e.g., Peak-to-Peak or Drop
Distance" -- a suggestion in the source document, not a confirmation from
Dr. Balaji. `mentor_eval/disease_similarity_table.py` computes Bhattacharyya
distance as a **provisional placeholder** (already in the pipeline, zero
marginal cost) and labels the column accordingly in both the CSV/MD output
and this report. This requires Dr. Balaji's confirmation before being
treated as final for publication.

## Healthy Sinus -- Open Question, Not Silently Resolved

The mentor's table template includes a "Healthy Sinus" row distinct from
"Normal." This project's class taxonomy (`mentor_eval.class_mapping.MENTOR_CLASSES`)
has no such class. Checked `data/ptbxl/scp_statements.csv` directly: PTB-XL
has a separate `SR` ("sinus rhythm") code under a different statement
category (`rhythm`) from `NORM` (`diagnostic`) -- so "Healthy Sinus"
plausibly means "diagnostically Normal AND confirmed sinus rhythm," a
stricter subset of Normal, not a synonym. `disease_similarity_table.py`
reports this as its own row, explicitly marked pending clarification,
rather than merging it into Normal or omitting it silently.

## Files

- `mentor_eval/hausdorff_distance.py` -- `compute_hausdorff`,
  `compute_hausdorff_per_lead`, `matched_hausdorff`
- `mentor_eval/test_hausdorff_distance.py` -- 13 tests
- `mentor_eval/disease_similarity_table.py` -- assembles the mentor-requested
  table (Cosine, Mahalanobis, Hausdorff, provisional 4th metric), reusing
  `similarity_metrics.py`'s existing functions, not reimplementing them

## Status

Requires a trained checkpoint to produce real numbers (same "write-only
deliverable until run on the GPU server" status as `similarity_metrics.py`
itself). Verified end-to-end locally against `outputs/models/diffusion_best.pt`
with `--n-generated 2` (control-flow smoke test only, not a real result --
too few samples for a meaningful number, output deleted after verification).
Run with the full `--n-generated 200` (default) on the GPU server, against
whichever checkpoint is the actual target of the mentor's request, for the
real deliverable.
