# Experiment 4 — MentorClassifier Verification (AFIB reject-class test)

**Status: COMPLETE.** Run locally on 2026-07-02 (no GPU/diffusion checkpoint
needed — see `Code/Experiment_4_Classifier_Verification/README.md`).
Raw data: `Outputs/Experiment_4_Classifier_Verification/*.csv`. Figures:
`Figures/Experiment_4_Classifier_Verification/*.png`.

## Setup actually used

MentorClassifier trained fresh (best val macro F1 = 0.7243 over 30 epochs,
13,698 train / 1,682 val records). Evaluated on the official PTB-XL test
fold: 1,708 records — Normal 963, STEMI 516, NSTEMI 185, **AFIB only 44**.
That class-size imbalance (AFIB is 2.6% of the test set, vs. Normal's 56%)
is itself relevant context for everything below.

## Finding 1 — AFIB is already the weakest class on clean data

At `sigma=0` (no added noise), per-class accuracy: Normal 0.974, STEMI
0.643, NSTEMI 0.751, **AFIB 0.455**. AFIB is worst by a wide margin, and its
mean prediction confidence when correct is also lowest (0.653, vs. Normal's
0.942). The classifier is both least accurate and least confident on real
AFIB before any corruption is introduced.

## Finding 2 — AFIB absorbs a large, disproportionate share of noise-driven misclassifications at moderate noise

| sigma | frac. of all flips landing on AFIB | AFIB attraction ratio (vs. 1/4 chance) |
|---|---|---|
| 0.10 | 42.2% | 1.69x |
| 0.25 | **79.9%** | **3.19x** |
| 0.50 | **89.5%** | **3.58x** |
| 1.00 | 58.5% | 2.34x |
| 2.00 | 23.2% | 0.93x |

At moderate corruption (sigma=0.25-0.5), roughly 4 out of 5 misclassified
samples — regardless of their true class — get relabeled AFIB. That is far
beyond chance (would be 25% if flips were evenly distributed across the 4
classes) and is exactly the signature the master research question asked
about: **AFIB behaves like an out-of-distribution / reject bucket at
moderate noise levels.**

## Finding 3 — the reject-bucket role shifts to NSTEMI at extreme noise

At `sigma=1.0` and `sigma=2.0`, AFIB's share of flips drops back toward (and
below) chance, while NSTEMI's share rises to 76.8% at `sigma=2.0`. Combined
with `noise_robustness.csv` showing Normal and STEMI accuracy collapse to
**exactly 0.0** by `sigma=0.5`-`1.0` while NSTEMI accuracy partially
recovers (0.46 → 0.52 → 0.80 as sigma rises from 0.5 to 2.0), the picture is:
under moderate corruption the model's default "I don't recognize this"
answer is AFIB; under extreme corruption (input essentially pure noise) its
default shifts to NSTEMI instead. There are **two different reject
attractors depending on corruption severity**, not one consistent OOD
bucket — a more nuanced finding than a single "AFIB is the reject class"
statement would suggest.

## Finding 4 — confidence calibration is poor but not in the "confidently wrong" direction for AFIB specifically

AFIB's mean confidence when predicted stays in a narrow 0.48-0.65 band
across all noise levels — it does not stay artificially high the way a
classic miscalibrated reject-bucket would (compare Normal's confidence,
which starts at 0.94 and craters to 0.44 as its own accuracy collapses).
So while AFIB clearly absorbs a disproportionate share of ambiguous input
at moderate noise (Finding 2), the model isn't confidently certain when it
does so — it's moderately-confidently wrong, which is a different (milder)
failure mode than "the model is sure everything unfamiliar is AFIB."

## What this means for Stage 1's other experiments

- **Experiments 1-3 must never use AFIB predictions on generated samples**
  as an "is conditioning correct" signal even indirectly — there is no
  generated AFIB in this pipeline anyway (see `Architecture.md` §4), but
  this experiment shows real AFIB predictions themselves are unreliable
  enough (55% error rate at zero noise, heavy OOD absorption under
  moderate corruption) that AFIB should be treated as the least trustworthy
  of the 4 mentor classes for any conditioning-quality inference, not just
  an "excluded, no trained class" footnote.
- Because generated ECGs are unlikely to be literally random noise (a
  reasonably-trained diffusion model produces plausible-looking, if
  possibly wrong-class, waveforms), the **moderate-noise regime
  (sigma≈0.25-0.5) is the more relevant analogy** for "what happens to
  MentorClassifier when it sees an ECG that isn't quite in-distribution
  for its predicted class" — and that is exactly the regime where AFIB
  absorbs 80-90% of the errors. Any Experiment 1/2/3 result showing
  generated samples classified as AFIB-adjacent, or generated-STEMI/NSTEMI
  distinction breaking down, should be cross-checked against this finding
  before concluding it's a conditioning failure specific to those classes.

## Answer to the specific question posed

**Does AFIB behave like an out-of-distribution reject class? Yes, at
moderate corruption levels (sigma≈0.25-0.5) — with 3.2-3.6x the expected
share of misclassifications landing there — but this is not AFIB's
exclusive property under all conditions: at extreme corruption, NSTEMI
takes over the same role.** AFIB is not a reliable arbiter of "correct
conditioning" and should be excluded from any binary pass/fail judgment
about conditioning quality (it already is, structurally, from generation —
this finding extends that exclusion to real-data-based evaluations too).
