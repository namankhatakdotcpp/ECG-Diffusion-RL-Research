# Research Roadmap

This directory is the single home for all diagnosis / architecture / optimization
research work on the ECG conditional diffusion model, organized as a 5-stage
scientific roadmap. Nothing related to this roadmap should be stored outside
this tree.

## Rule

**Do not skip stages.** Each stage's `Decisions.md` must justify — with
evidence, not assumption — why it is safe to proceed to the next stage.
Stage 1 is diagnosis-only: no architecture changes are made until Stage 1
concludes.

## Structure

```
Roadmap/
    README.md                  — this file
    MASTER_PROGRESS.md         — single-page status across all stages
    roadmap.json                — machine-readable stage/experiment status

    Stage_1_Diagnosis/                    — why does conditioning fail? (complete)
    Stage_2_Architecture_Investigation/   — (complete) verification + audit +
                                             Tier 0 measurement, no architecture changes
    Stage_3_Architecture_Improvements/    — (in progress) actual architecture
                                             changes, one at a time -- Phase 0 complete,
                                             candidates S3-001..006 implemented,
                                             GPU training underway
    Stage_4_Optimization/                 — (not started — gated on Stage 3)
    Stage_5_Final_Model/                  — (not started — gated on Stage 4)
```

Each stage folder contains:
- `README.md` — what this stage is and why it exists
- `Objectives.md` — concrete questions this stage must answer
- `Progress.md` — running status
- `Experiment_Log.md` — dated log of what was run, with commands/configs used
- `Decisions.md` — conclusions and the evidence backing them
- `Outputs/`, `Figures/`, `Code/`, `Reports/` — stage artifacts

Some stages' `Objectives.md`/`Progress.md`/`Decisions.md`/`Experiment_Log.md`
are deliberately left as one-line pointers rather than filled in when a
stage's own status file (e.g. `STAGE2_STATUS.md`, `Stage3_Status.md`)
already tracks that same information -- kept in one place rather than two
copies that can drift apart. Check each stage's own `README.md` first.

## Current stage: Stage 3 — Architecture Improvements

See [`Stage_3_Architecture_Improvements/Stage3_Status.md`](Stage_3_Architecture_Improvements/Stage3_Status.md)
for live status. Stage 1 ([`Stage_1_Diagnosis/README.md`](Stage_1_Diagnosis/README.md))
and Stage 2 ([`Stage_2_Architecture_Investigation/README.md`](Stage_2_Architecture_Investigation/README.md))
are both complete.
