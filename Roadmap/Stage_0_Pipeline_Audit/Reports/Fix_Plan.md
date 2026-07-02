# Stage 0 Pipeline Audit — Fix Plan

**Updated 2026-07-02 (same day, twice): Finding 14 was raised as Critical,
then resolved.** The three fixes below (Findings 5, 6, 8) were implemented
and committed before Finding 14 was raised — they are correct and remain
in place, and their verification diffs (run on the full 17,418-record
corpus) are now confirmed to describe the real population, not a
different one.

## RESOLVED — Finding 14: the ~380-record curated training population question

**What was found:** no code in this repository's current tree, or full
git history across all branches, produces a 380-record training subset.
`X_train.npy` is confirmed `(17418, 1000, 12)`.

**How it was resolved:** an independent, unrelated earlier session in
this project's history — sizing local Mac compute budget, with no
awareness the 380-record claim existed — measured real wall-clock
throughput and reported "544 steps/epoch." `17,418 // 32 = 544` exactly
(verified: `step04_transformer_diffusion.py`'s `DataLoader` uses
`drop_last=True`, so 544 is the literal per-epoch step count, not a
rounding artifact). Two independent methods — this session's code+history
search, and that earlier session's runtime benchmark — landed on the same
number for unrelated reasons, and both actively contradict the "380
curated sequences, sub-ten-minute run" framing (380/32 floor = 11
steps/epoch, 2,200 total — the math that framing would need; 17,418
records gives 108,800 total steps instead).

**Disposition:** the real, long-standing training population is
~17,418 records. The entire prior "Investigation Timeline" (embedding-
scale experiment, AdaLN-Zero, decoupled signals, CFG sweep, every
conditioning-collapse percentage, AFIB-attractor findings) is
**historical narrative only** — not evidence — until independently
re-derived from a real run in this repository. No number from that
report should motivate Stage 2 priority ordering (including the
LayerScale hypothesis's promotion to first-in-line) until then. The old
dataset-scaling experiment's `n_train_records_actual=380` ledger entries
are the same: historical narrative, not a bug to fix, since no code path
was ever capable of producing that number in the first place. A genuine
dataset-scaling experiment, if wanted later, needs a real, documented,
version-controlled subsetting mechanism built from scratch — not a
retrofit of the old script.

**`--sanity-check` is authorized, against the real, confirmed
17,418-record corpus.** Do not subsample to 380 to match the old
narrative.

---

## RESOLVED — implemented, committed, and re-verified (before Finding 14 was found)

### Finding 6 — Stale class-taxonomy fallback (HIGH, borders Critical) — FIXED (commit `f8dba53`)

**Why blocking:** this is not latent — `config.yaml`'s fallback values
are wrong *today*, and the Experiment 1 GPU-server README's copy-over
instructions create a concrete, plausible path to triggering it on the
very next real run (the one Stage 2's Verification Gate is waiting on).

**Proposed fix (two parts, either sufficient alone, both recommended):**
1. Correct `config.yaml:42-49`'s `ptbxl.classes` list to the real 6-class
   taxonomy (`NORM, MI, STTC, CD, HYP, OTHER`), and either fix or remove
   the unused `n_classes: 7` field.
2. Update `Roadmap/Stage_1_Diagnosis/Code/Experiment_1_Baseline/README.md:25`
   to explicitly list `class_names.json`, `class_mapping.json`, and
   `class_counts.json` alongside the `.npy` files in the copy-over
   instructions — or better, make `step04_transformer_diffusion.py`'s
   fallback path (`step04_transformer_diffusion.py:711-717`) raise an
   error instead of silently proceeding with a guessed class list, since
   "guess and continue" is the actual root cause risk, independent of
   whether the guessed values happen to be correct this time.

**Status: DONE.** `config.yaml` fixed to 6 classes; the "raise loudly"
version of the fix was implemented (an `assert` in `train()`, not just a
doc update) — verified both directions: passes with the corrected config,
and would have raised immediately on the old 7-vs-6 mismatch.

### Finding 5 — Undocumented tie-break order-dependence (HIGH) — FIXED (commit `a9e6047`)

**Why blocking-adjacent, not strictly blocking:** it doesn't invalidate
existing results (Finding 4 proved step03/step04 currently agree with
each other on every record), and it doesn't change if left alone before
the next run — the same deterministic labels get produced either way.
Flagged HIGH rather than a stronger classification because of this
distinction: it affects reproducibility across code changes/dict-order
assumptions, not correctness of the current run in isolation. Recommend
the human explicitly decide whether to treat this as blocking or
deferrable, since "deterministic but not principled" is a judgment call
about acceptable risk, not a pure correctness question.

**Proposed fix:** define one explicit, documented tie-break rule (e.g.
alphabetical-by-code, or a clinical severity ranking) and apply it via a
single shared function imported by both `step03_eda_and_class_mapping.py`
and `step04_transformer_diffusion.py`, rather than the current two
independently-duplicated implementations of the same rule (which is
exactly the drift-risk pattern Finding 4 checked for and — this time —
found clean, but duplicated logic remains one bad edit away from
silently diverging).

**Status: DONE.** Implemented as `utils/label_assignment.py`, a single
shared function used identically by both files. Fix verified with a
before/after class-distribution diff on all 21,799 real records — but
per Finding 14, that diff was run on the full corpus, which may not be
the population that matters. **The fix itself is correct regardless**
(it closes a real duplicated-logic drift risk either way); only its
*quantified impact* is pending re-verification against the real
training population once Finding 14 is resolved.

---

## DEFERRABLE — real, but does not block the next run

### Finding 8 — Checkpoint accumulation (MEDIUM) — FIXED (commit `f78c6c2`)

Quantified: ~1.22GB per full Experiment-1-style run, ~540MB per
`--sanity-check` run, compounding indefinitely via `snapshot_before_write`'s
rename-not-delete behavior. Does not block the *next* run (the
disk-headroom gate already added to the Step 2 SSH runbook catches an
immediate out-of-space failure), but should be addressed before this
project accumulates many more runs on `himtenduh`.

**Proposed fix:** (a) retention policy in `step04_transformer_diffusion.py`'s
checkpoint-save block — keep last N periodic checkpoints + best, delete
older; (b) a retention policy for `snapshot_before_write`-created backup
directories specifically, since `--sanity-check` runs have no reason to
have their backups preserved indefinitely.

### Finding 7 — Empty-`scp_codes` handling divergence (LOW)

0 records currently affected. Fix when convenient — align step04's
empty-dict case with its own existing "no code matched" fallback two
lines later.

### Finding 11 — Comment-only validation invariant (LOW)

Add the suggested assertion (`Pipeline_Code_Audit.md` Finding 11) next
time `step04_transformer_diffusion.py`'s validation loop is touched for
any other reason — low value in a dedicated PR by itself.

### Finding 12 — Hardcoded `n_leads`/signal-shape literals (LOW)

Style/robustness only. Defer indefinitely unless `signal_length` or lead
count ever actually needs to change.

---

## Explicitly NOT requiring any fix

Findings 1, 2, 3, 4, 9, 10, 13 — all refuted by direct evidence (real-data
checks, full diffs, or clean scans). No action needed; documented in
`Pipeline_Code_Audit.md` so the checks are on record and don't need to be
re-litigated later.

---

## What actually happened, in order

1. Findings 6, 5, 8 reviewed and fixed (commits `f8dba53`, `a9e6047`, `f78c6c2`).
2. Finding 5's fix verified with a before/after diff on all 21,799 records
   — correct for the full corpus.
3. While sanity-checking that diff against this project's own numeric
   provenance discipline, discovered the diff's population (the full
   corpus) does not match the ~380-record population described in every
   prior conditioning-collapse report — and that no code anywhere in this
   repository, current or historical, produces that 380-record
   population. **This became Finding 14, Critical.**
4. Resolved same day: an independent, unrelated earlier benchmark session
   (544 steps/epoch, exactly matching 17,418/32) corroborated the full
   corpus as the real population. Findings 5/6/8's verification diffs are
   confirmed accurate as-is.

## Current state

**`--sanity-check` is authorized**, against the real, confirmed
17,418-record corpus. Do not subsample to 380 to match the old narrative
— see the Finding 14 resolution above for why that would be backwards.
Run it, report the ledger entry back, and do not proceed to the full
200-epoch run until that's reviewed.
