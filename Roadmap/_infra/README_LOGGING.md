# Experiment Logging Infrastructure

Two files, no dependencies beyond the Python standard library (and `torch`
if present — it's optional and only used to record environment metadata):

- `experiment_logger.py` — wrap every experiment's code in `ExperimentLogger`
- `build_master_log.py` — regenerates `Reports/MASTER_LOG.md` from the
  ledger; called automatically on every `ExperimentLogger` exit, or run
  standalone

## Where files land

Both files live in `Roadmap/_infra/`. Each stage directory
(`Roadmap/Stage_1_Diagnosis/`, `Roadmap/Stage_2_.../`, ...) gets its own
`Logs/` and `Reports/` subfolder the first time an experiment in that stage
runs — `ExperimentLogger` creates them if missing.

```
Roadmap/
    _infra/
        experiment_logger.py
        build_master_log.py
        README_LOGGING.md
    Stage_1_Diagnosis/
        Logs/                          # one .log file per run, full stdout+stderr transcript
            exp2_dataset_scaling_5000_20260702T091500Z.log
            ...
        Reports/
            results_ledger.jsonl       # append-only, one JSON object per run — never hand-edit
            MASTER_LOG.md              # auto-generated from the ledger — never hand-edit
    Stage_2_.../
        Logs/
        Reports/
            results_ledger.jsonl
            MASTER_LOG.md
```

## Minimal usage inside any experiment script

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_infra"))
from experiment_logger import ExperimentLogger

with ExperimentLogger(
    experiment_id="exp2_dataset_scaling_5000",   # unique per run
    stage="Stage_1_Diagnosis",                   # matches the folder name
    root_dir=Path(__file__).resolve().parents[1],  # .../Stage_1_Diagnosis
    params={"dataset_size_requested": 5000, "n_epochs": 200},
    seed=42,
) as exp:
    # ... existing experiment code, completely unchanged ...
    # every print() statement is captured automatically

    exp.log_metric("n_train_records_actual", len(X_train))
    exp.log_metric("accuracy", acc)
    exp.log_metric("collapse_fraction", collapse)
    exp.log_artifact(csv_path, "raw per-epoch metrics")
    exp.log_note("re-run after fixing the dataset loader slicing bug")
```

No other code changes are required. Nothing about how the script prints
needs to change — the logger tees stdout/stderr, it doesn't replace it.

## What you get automatically

1. A full console transcript for every run, timestamped, saved even if the
   run crashes.
2. One structured JSON record per run in `results_ledger.jsonl`, containing
   params, metrics, artifact paths, git commit hash, branch, dirty-working-
   tree flag, Python/torch versions, and CUDA device — so six months from
   now you can answer "what commit produced this number" without guessing.
3. `MASTER_LOG.md`, regenerated after every single run, containing:
   - A **Flags** section at the top listing anomalies an automated check
     found — including the exact class of bug that produced the
     `n_train_records_actual = 380` issue across every dataset-scaling run.
     This check is generic: it doesn't know anything about ECGs, it just
     notices when a `*_requested` param and a matching `*_actual` metric
     disagree, or when metrics are byte-identical across a family of runs
     that were supposed to vary a parameter. A run with a real bug like
     that will not silently pass Stage 1 review next time — it gets
     surfaced the moment the run finishes, not after someone manually
     eyeballs a CSV.
   - A single table of every run across the whole stage, sortable by
     whatever metrics you've logged.
   - A dedicated section listing every failed/crashed run with its full
     traceback inline, so a crashed overnight run doesn't require digging
     through log files to find out what happened.

## Regenerating the digest without running anything

If you ever need to rebuild `MASTER_LOG.md` from an existing ledger (e.g.
after manually editing a bad record, which should be rare):

```bash
python Roadmap/_infra/build_master_log.py Roadmap/Stage_1_Diagnosis
```

## Extending the sanity checks

`_run_sanity_checks()` in `build_master_log.py` is a plain list of small,
independent functions-in-a-loop — add a new check by adding a new loop over
`records` that appends human-readable strings to `flags`. Keep every check
generic (operating on the params/metrics dict shape, not on ECG-specific
field names) so it keeps paying off in later stages that have nothing to
do with dataset scaling.
