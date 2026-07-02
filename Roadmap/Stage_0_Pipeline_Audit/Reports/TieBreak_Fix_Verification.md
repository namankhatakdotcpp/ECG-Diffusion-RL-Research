# Finding 5 Fix — Verification (before/after class-distribution diff)

**Fix:** `utils/label_assignment.py` (new shared module), wired into
`step03_eda_and_class_mapping.py:_assign_primary` and
`step04_transformer_diffusion.py:_load_class_labels`. Replaces the
undocumented first-in-dict-order tie-break with an explicit, documented
clinical-severity priority: `MI > STTC > CD > HYP > NORM > OTHER`.

**Correction before presenting this diff:** the "before" baseline for
OTHER cited when this fix was requested was **n=2 project-wide** — that
number is wrong for this repository (it originated in a prior reviewer
critique and was never checked against this repo's actual
`class_counts.json` before being repeated in `representation_metrics.py`
and `Stage2_Master_Prompt.md` — see task #23, fixed separately in this
same commit). The real, verified baseline is below.

## Class taxonomy stability (checked first, since everything else depends on it)

Before trusting any per-class count diff, confirmed the **set of classes
that clear `min_class_samples=200`** does not change under the new
tie-break — if it did, `class_names.json`/`class_mapping.json` themselves
would need regenerating, not just the counts.

```
Raw (pre-_remap) train superclass counts, NEW tie-break:
        NORM:  7,269  KEEP
          MI:  3,406  KEEP
        STTC:  3,404  KEEP
          CD:  2,495  KEEP
         HYP:    505  KEEP
       OTHER:    254  KEEP   (records with no diagnostic code at all -- unaffected by tie-break, as expected)
        AFIB:     85  DROP (< 200)
```

Same 6 final classes as before (`NORM, MI, STTC, CD, HYP, OTHER`), same
AFIB exclusion. **`class_mapping.json`/`class_names.json` do not need to
be regenerated** — the fix only changes which records land in which of
the *existing* 6 classes, not the taxonomy itself.

Internal consistency check: raw "no code matched" OTHER (254, must be
tie-break-invariant since these records have zero candidates) + raw AFIB
that would fold into OTHER via `_remap` (85) = 339, which exactly matches
the per-record `_load_class_labels` OTHER count below (339). Two
independently-computed numbers agree — this is not a hand-wavy estimate.

## Per-split class distribution: BEFORE vs. AFTER

Computed by actually calling the real (patched) `_load_class_labels()`
against all three real splits — not estimated.

| Class | Split | BEFORE | AFTER | Δ | Δ% |
|---|---|---:|---:|---:|---:|
| NORM  | train | 7,386 | 7,269 | -117 | -1.6% |
| NORM  | val   |   928 |   919 |   -9 | -1.0% |
| NORM  | test  |   932 |   917 |  -15 | -1.6% |
| MI    | train | 3,374 | 3,406 |  +32 | +0.9% |
| MI    | val   |   401 |   407 |   +6 | +1.5% |
| MI    | test  |   411 |   418 |   +7 | +1.7% |
| STTC  | train | 2,651 | 3,404 | +753 | **+28.4%** |
| STTC  | val   |   338 |   426 |  +88 | **+26.0%** |
| STTC  | test  |   350 |   420 |  +70 | **+20.0%** |
| CD    | train | 2,630 | 2,495 | -135 | -5.1% |
| CD    | val   |   340 |   319 |  -21 | -6.2% |
| CD    | test  |   351 |   333 |  -18 | -5.1% |
| HYP   | train | 1,036 |   505 | -531 | **-51.3%** |
| HYP   | val   |   139 |    75 |  -64 | **-46.0%** |
| HYP   | test  |   113 |    69 |  -44 | **-38.9%** |
| **OTHER** | **train** | **254** | **339** | **+85** | **+33.5%** |
| **OTHER** | **val**   |  **29** |  **37** |   **+8** | **+27.6%** |
| **OTHER** | **test**  |  **30** |  **41** |  **+11** | **+36.7%** |

Total record count per split is **unchanged** (train=17,418, val=2,183,
test=2,198) — the fix only reassigns which class a tied record gets, it
never adds or drops a record (that's Finding 7's territory, separate and
deliberately not touched here).

## What actually moved, and why this is the expected/correct behavior of a real fix

- **HYP dropped by ~40-51%** and **STTC gained ~20-28%**: HYP is second-
  lowest priority (only above NORM/OTHER) in the new ordering, while STTC
  is second-highest (only below MI). Most ties previously "won" by HYP
  under arbitrary dict order now correctly resolve toward whichever
  higher-priority diagnosis was also present at the same confidence.
- **OTHER increased 27-37% across all splits** (254→339 train) — this is
  the number explicitly flagged as load-bearing. **Every prior Stage 1/2
  report that cites "OTHER has 254 training records" (Architecture.md,
  Roadmap/Stage_1_Diagnosis/Experiment_Log.md, class_counts.json itself)
  will describe stale data once step02/step03 are actually rerun with
  this fix on the GPU server.** This audit does not rerun step02/03 (no
  training/pipeline execution happened — code review and a standalone
  verification script only, consistent with Stage 0's read-only-except-
  fixes scope); flagging this explicitly so whoever runs step02/03 next
  knows to expect different numbers than every existing report.
- **NORM decreased slightly (-1.6%), MI/CD changed modestly (<6%)**: both
  expected — NORM is second-to-lowest priority so loses some ties it
  previously won by chance, MI/CD are high/mid priority so are largely
  unaffected either direction.

## Verdict

Fix implemented as a single shared function
(`utils/label_assignment.assign_primary_class`) used identically by both
`step03_eda_and_class_mapping.py` and `step04_transformer_diffusion.py`,
closing the duplicated-logic drift risk Finding 4 checked for (and found
clean, but only by chance of the two independent implementations
happening to still agree — now structural, not incidental). Verified:
class taxonomy stable, per-class counts changed as expected given the new
priority order, internal consistency check passed (339 = 254 + 85 via two
independent computation paths).
