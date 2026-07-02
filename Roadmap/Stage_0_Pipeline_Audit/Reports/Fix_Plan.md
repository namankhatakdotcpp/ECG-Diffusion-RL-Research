# Stage 0 Pipeline Audit — Fix Plan

**Updated 2026-07-02: 1 Critical finding now exists (Finding 14).** The
three fixes below (Findings 5, 6, 8) were implemented and committed
*before* Finding 14 was discovered — they are correct fixes for what they
claim to fix, and remain in place. But Finding 14 means their
before/after verification diffs (run on the full 17,418-record corpus)
may not describe the actual ~380-record population every prior
conditioning-collapse finding in this project was built on. **Finding 14
is the current blocker, superseding the original "no Critical findings"
framing this document opened with.**

## CRITICAL — blocks `--sanity-check` and all further GPU execution

### Finding 14 — The ~380-record curated training population has no corresponding code path anywhere in this repository

**Why this blocks everything else:** every fix in this plan, and every
verification diff run against those fixes, assumed the training
population was whatever `step02`/`step04` currently produce from the full
PTB-XL corpus. Prior investigation reports (LaTeX Table 3) describe a
*different*, much smaller (~380-record) curated population with a
specific per-class breakdown (NORM=231, STTC=50, CD=45, MI=36, HYP=16,
OTHER=2) that no code in this repository's current tree, or anywhere in
its full git history across all branches, produces. `X_train.npy` on this
machine is confirmed `(17418, 1000, 12)` — not 380.

**This is not resolvable by more code review.** Two possibilities, not
distinguishable from this repository alone: (a) the 380-record population
was built out-of-band on a different machine/session and never committed
(`outputs/` is gitignored by design), and has since been silently
overwritten by a full-corpus run; or (b) the "380 curated sequences"
narrative was never actually implemented as described. **A human needs to
either locate the actual artifact/script (if it exists somewhere outside
this repo) or decide to treat every prior conditioning-collapse finding
as provisionally retracted pending reproduction from a population that
demonstrably exists in this codebase.**

**Do not run `--sanity-check`, and do not proceed with any further GPU
execution, until this is resolved one way or the other.** Running
`--sanity-check` now would implicitly treat this open question as
settled — it isn't.

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
   population. **This is Finding 14, Critical, and is now the blocker.**

## Current state

**Do not begin Steps 1-4 execution or the `--sanity-check` flow.**
Findings 5/6/8 being fixed does not change this — Finding 14 means the
population those fixes were verified against may not be the population
that matters, and until a human resolves where the real ~380-record
subset comes from (or decides to proceed on the full corpus instead,
explicitly), authorizing any further GPU execution would be building on
the same kind of unverified premise this whole audit exists to catch.
