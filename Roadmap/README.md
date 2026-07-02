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

    Stage_1_Diagnosis/                    — why does conditioning fail?
    Stage_2_Architecture_Investigation/   — (not started — gated on Stage 1 decision)
                                             verification + audit + Tier 0 measurement,
                                             no architecture changes
    Stage_3_Architecture_Improvements/    — (not started — gated on Stage 2 Tier 0 evidence)
                                             actual architecture changes, one at a time
    Stage_4_Optimization/                 — (not started)
    Stage_5_Final_Model/                  — (not started)
```

Each stage folder contains:
- `README.md` — what this stage is and why it exists
- `Objectives.md` — concrete questions this stage must answer
- `Progress.md` — running status
- `Experiment_Log.md` — dated log of what was run, with commands/configs used
- `Decisions.md` — conclusions and the evidence backing them
- `Outputs/`, `Figures/`, `Code/`, `Reports/` — stage artifacts

## Current stage: Stage 1 — Diagnosis

See [`Stage_1_Diagnosis/README.md`](Stage_1_Diagnosis/README.md).
