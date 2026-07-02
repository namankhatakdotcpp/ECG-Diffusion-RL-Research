# Stage 2.0 — Verification Gate Report

**Status: BLOCKED — cannot proceed as specified.**

## What was checked

Per the Stage 2 master prompt, Stage 2.0 requires:
`Roadmap/Stage_1_Diagnosis/Reports/results_ledger.jsonl` and
`Roadmap/Stage_1_Diagnosis/Reports/MASTER_LOG.md` to exist on disk.

```
$ ls Roadmap/Stage_1_Diagnosis/Reports/
Architecture_Review_Findings.md
Stage1_Results_Digest.md
baseline_report.md
classifier_validation_report.md
dataset_scaling_report.md
directional_conditioning_report.md

$ find Roadmap/Stage_1_Diagnosis -iname "Logs" -o -iname "*ledger*"
(no output)
```

**Neither `results_ledger.jsonl` nor `MASTER_LOG.md` exists anywhere under
`Roadmap/Stage_1_Diagnosis`.** There is no `Logs/` directory either.

## Why, not just that

This is not a mystery to investigate — it's a straightforward sequencing
fact from this project's own history. `ExperimentLogger` and
`build_master_log.py` (the tools that produce `results_ledger.jsonl` and
`MASTER_LOG.md`) were designed and built in *this session*, specifically
as prep for Stage 2, and committed only just now (`d374995`). Stage 1's
actual experiments — including the one real result that exists,
Experiment 4's MentorClassifier verification — were run *before* that
infrastructure existed, using an earlier, purpose-built (ECG-specific, not
generic) mechanism: `Roadmap/Stage_1_Diagnosis/collect_stage1_results.py`,
which produced `Reports/Stage1_Results_Digest.md` by scanning known output
file paths directly, not a JSON-Lines ledger.

So the Verification Gate's literal instruction ("if either is missing,
stop and report exactly what is missing") is correctly triggered — but the
underlying cause is a tooling-generation-order gap, not evidence that
Stage 1's results are unreliable or unverified in some other sense.

## What Stage 1 evidence actually exists, and its real verification status

Reading `Stage1_Results_Digest.md` (the actual artifact Stage 1 produced)
directly, rather than assuming the ledger's absence means no evidence
exists at all:

| Experiment | Status per digest | Ledger-based re-verification possible? |
|---|---|---|
| 1 — Baseline Reproduction | Not run (no GPU checkpoint on the dev machine) | No — never ran |
| 1.5 — Checkpoint Verification | Not run (depends on Exp 1) | No |
| 2 — Dataset Scaling | Not run (needs GPU server) | No |
| 2.5 — Training Curves | Not run (built into Exp 2) | No |
| 3 — Directional Conditioning | Not run (needs GPU server) | No |
| 3.5 — Layer-wise Direction Probe | Not run (needs Exp 1's checkpoint) | No |
| **4 — MentorClassifier Verification** | **Complete, ran locally** | **Yes, in principle — but not via a ledger; via re-running the script and diffing its CSV output** |
| 4.5 — Feature Drift Visualization | Real+noise half complete; generated-sample half pending Exp 1 | Partially, same caveat as 4 |

**Correction to the Stage 2 prompt's "PROJECT CONTEXT" section:** that
section states, as background, "Real-data MentorClassifier: ~83% accuracy,
~0.95 macro AUC" and "Generated-data classifier accuracy: ~2-7%,
near-chance or below" and a layer-wise magnitude-decay claim
(0.91 → 0.24) as things reported but not yet confirmed. Based on what
actually exists in this repository:

- The real-data number is close but not identical to what's on disk:
  `outputs/mentor_review/classification_validation/classifier_real_eval.json`
  (pre-existing, predates this session) reports accuracy=0.8437,
  macro_f1=0.7428, macro_auc=0.9582 — consistent with "~83%/~0.95" as a
  rough paraphrase.
- **The generated-data classifier accuracy (~2-7%) and the layer-wise
  magnitude-decay numbers (0.91 → 0.24) do not correspond to anything in
  this repository.** No generated-data classifier evaluation exists
  (Experiment 1 has never been run — no checkpoint exists on any machine
  this session has access to), and no layer-wise probe has been run
  either (Experiment 3.5 requires Experiment 1's checkpoint, which does
  not exist). These specific numbers must have come from a different run,
  a different environment, or a different conversation than the one that
  produced this repository's actual Stage 1 output. **They should not be
  treated as background fact for this repository's Stage 2 investigation
  until Experiment 1 actually runs on the GPU server and produces them
  here.**

## Verdict

- **CONFIRMED, independently re-derivable right now:** Experiment 4's
  headline finding (AFIB attraction ratio peaking at 3.58x chance at
  sigma=0.5) — the raw CSV (`noise_robustness.csv`, `prediction_drift.csv`)
  is on disk and was read directly to write
  `classifier_validation_report.md`. Re-running
  `classifier_verification.py` would reproduce it (same seed=42, same
  deterministic data).
- **NEEDS-REPRODUCTION:** Experiments 1, 1.5, 2, 2.5, 3, 3.5 — none have
  ever been run. There is nothing to verify; there's only code to run.
- **DISQUALIFIED (per this report, not per any MASTER_LOG flag, since none
  exists):** the "~2-7% generated accuracy" and "0.91→0.24 layer-wise
  decay" figures quoted in the Stage 2 prompt's context section. Do not
  cite these anywhere in Stage 2 reasoning until they are produced by an
  actual run in this repository.

## Recommendation

1. Do not proceed to Stage 2.2 (dataset-scaling root cause) or any Tier 0
   item that depends on Experiment 1's checkpoint (items 1-3, 5-8 all
   need it either directly or via the generated-sample evaluation chain;
   item 4's gradient-norm comparison needs at least two checkpoints from
   different epochs, also from Experiment 1).
2. Going forward, every experiment — including a first, overdue retrofit
   of Experiment 4 — should be wrapped in `ExperimentLogger` so this gate
   has something real to check next time. This report itself should be
   treated as the trigger for that retrofit, not just a one-time
   exception.
3. Experiment 1 (and therefore everything downstream of it) still requires
   the GPU server per Stage 1's own conclusion — this has not changed.
