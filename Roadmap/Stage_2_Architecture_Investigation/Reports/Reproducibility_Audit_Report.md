# Stage 2.0.1 — Reproducibility Audit Report

**Status: COMPLETE.** Different question than the Verification Gate (do
reported numbers reproduce) or the Repository Audit (are artifacts
stale/duplicated). This asks: is each ledger entry's claimed artifact
identity — checkpoint, code state, config — actually what's on disk,
checked with a manifest that records its own comparison provenance so a
later reader can tell whether a verdict here is still fresh.

## Method and provenance (read this before trusting anything below)

`Roadmap/_infra/audit_reproducibility.py` builds one checksummed manifest
per ledger experiment. Every manifest carries an `audit_metadata` block
recording the **local machine's** git HEAD/branch/dirty-state and a
timestamp **at the moment the audit ran** — not the experiment's original
run time. This matters specifically because a bare "commit X matches"
claim is meaningless without saying what it was checked against; this
project has already had git state drift between the GPU server and this
Mac more than once this session, and a manifest that doesn't record its
own comparison basis would silently repeat that mistake.

This run's audit metadata (identical across all 7 manifests, run in one
batch): `audit_performed_at=2026-07-03T21:42:3x`,
`local_git_head_at_audit_time=5ce76c0bb16aa7738c97fbae5d4478feda45573a`,
branch `main`, local working tree **dirty** (expected — these report
files themselves were uncommitted at audit time). **Do not treat the
PASS verdicts below as valid past this commit/timestamp** — re-run the
audit if local git state moves on, per the manifest's own note.

Full per-experiment manifests: `Roadmap/Stage_2_Architecture_Investigation/
Outputs/Reproducibility_Manifests/<experiment_id>.json` (untracked —
regenerable, per repo policy; this report is the tracked summary).

## Results — all 7 Stage 1 ledger experiments

| Experiment | Commit reachable locally? | GPU working tree clean when it ran? | Checkpoint sha256 (first 16 hex) | Checkpoint size |
|---|---|---|---|---|
| `exp1_baseline_reproduction` | **PASS** | **WARNING — dirty** | `16ac1715ac90ecb3...` | 135,109,555 bytes |
| `exp2_dataset_scaling_380` | **PASS** | **WARNING — dirty** | `2b39752623f11e70...` | 67,571,219 bytes |
| `exp2_dataset_scaling_1000` | **PASS** | **WARNING — dirty** | `549d1b0845fd2498...` | 67,571,219 bytes |
| `exp2_dataset_scaling_2500` | **PASS** | **WARNING — dirty** | `64a8f6cd11831132...` | 67,571,219 bytes |
| `exp2_dataset_scaling_5000` | **PASS** | **WARNING — dirty** | `f99c956c874dbe59...` | 67,571,219 bytes |
| `exp2_dataset_scaling_10000` | **PASS** | **WARNING — dirty** | `269c2b42dd0fce08...` | 67,571,219 bytes |
| `exp2_dataset_scaling_full` | **PASS** | **WARNING — dirty** | `b2c7a83dc3e783b2...` | 67,571,219 bytes |

**Totals: 7/7 commits reachable (PASS), 7/7 ran with a dirty GPU working
tree (WARNING), 0 FAIL, 0 checkpoint checksum collisions** (all 7
distinct — consistent with the Repository Audit's separate duplicate
check on the same 6 Experiment 2 checkpoints).

`config.yaml`, `outputs/processed/class_names.json`, and
`outputs/processed/class_mapping.json` checksums recorded in every
manifest for future cross-run comparison (e.g. if a Tier 1 retrain
happens later and someone wants to confirm the class taxonomy didn't
silently change between runs).

## The one real, consistent finding: every run executed against an uncommitted, dirty working tree

All 7 experiments' `environment.dirty` field is `true`. This means: for
every one of them, the exact code that actually executed on the GPU is
**not fully captured by the logged commit hash alone** — there were
uncommitted local modifications on top of that commit at run time, and
nothing in the ledger records what those modifications were (a `git
diff` at run time was never captured; `ExperimentLogger` records
dirty/clean as a boolean, not the diff itself).

**What this does and doesn't mean:** it does not mean the results are
wrong — the commit hash still identifies a real, known base state, and
nothing in Stage 2.0's Verification Gate found any internal
inconsistency in the numbers themselves. It does mean that if a future
session ever needs to reproduce one of these runs bit-for-bit, "checkout
commit X" is necessary but not sufficient — there's an unrecorded delta
on top of it for every single run so far. Worth fixing going forward
(capture `git diff` output, not just a dirty boolean, in
`ExperimentLogger`), but that's a Stage 2 infra improvement to consider,
not a blocker for Tier 0 analysis, since the checkpoints/arrays
themselves are checksummed and stable regardless of what uncommitted
code produced them.

## Verdict

**PASSED**, with the dirty-working-tree caveat above noted for every
run, not just flagged and forgotten. Proceeding to Tier 0 item 1 (layer-
wise conditioning magnitude/direction) is authorized.
