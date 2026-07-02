# Stage 0 Pipeline Audit — Fix Plan

**No Critical findings.** Per the master prompt's rule, this audit does
not block on any Critical, since none exist — but the master prompt also
says not to start Steps 1-4 execution or the `--sanity-check` flow until
this document exists and every finding has been explicitly reviewed by a
human. That review has not happened yet as of this writing; this plan is
input to that review, not a substitute for it.

## BLOCKING — should be resolved (or explicitly waived by the human) before the next real GPU run

### Finding 6 — Stale class-taxonomy fallback (HIGH, borders Critical)

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

**Estimated effort:** trivial (a config edit + a doc edit), or small (an
exception instead of a `log.warning` + fallback) for the more robust fix.

### Finding 5 — Undocumented tie-break order-dependence (HIGH)

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

**Estimated effort:** small — extract `_assign_primary`/`_load_class_labels`'s
shared selection logic into `utils/` or similar, update both call sites.

---

## DEFERRABLE — real, but does not block the next run

### Finding 8 — Checkpoint accumulation (MEDIUM)

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

## Recommended order if the human approves proceeding with fixes

1. Finding 6 (trivial, highest operational relevance to the imminent GPU run)
2. Finding 5 (small, closes the duplicated-logic drift risk before it can
   ever silently diverge)
3. Re-run this audit's Finding 4 diff (the step03/step04 comparison
   script used for this audit) after any change to either file's
   selection logic, to confirm the fix didn't introduce a *new*
   divergence between the two files
4. Everything else, at the human's discretion, on no particular schedule

**Do not begin Steps 1-4 execution or the `--sanity-check` flow until a
human has explicitly reviewed this plan** (per the master prompt) —
this document is the input to that review, not the review itself.
