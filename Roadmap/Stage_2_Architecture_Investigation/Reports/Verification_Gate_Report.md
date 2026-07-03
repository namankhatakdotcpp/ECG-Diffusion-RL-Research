# Stage 2.0 — Verification Gate Report (real data)

**Status: PASSED.** This supersedes the earlier
`Verification_Gate_Report.md` written when no Stage 1 ledger existed
locally. `stage1_results.tar.gz` has since been extracted; this report
is derived entirely from those real, on-disk artifacts — every number
below was independently read, cross-checked, or recomputed, not restated
from a prior chat summary.

## 1. Ledger presence and completeness

`Roadmap/Stage_1_Diagnosis/Reports/results_ledger.jsonl` and
`MASTER_LOG.md` are present locally (extracted from the tar.gz, untracked
in git per the current repo policy — see `.gitignore`). Regenerated
`MASTER_LOG.md` locally against the real ledger:

```
## Flags (automated sanity checks)
_No anomalies detected._
```

7 entries, all `status: success`: `exp1_baseline_reproduction` and
`exp2_dataset_scaling_{380,1000,2500,5000,10000,full}`. No failures, no
crashes, zero flags. This also confirms the `build_master_log.py` fix
(committed earlier this session, `d0f43dd`) works correctly against the
real dataset it was designed for — the exact `batch_size` vs
`n_train_records_actual` and constant-`n_generated` patterns that would
have false-flagged this data under the old logic do not fire.

## 2. Cross-checked headline numbers (ledger vs. independent artifact)

| Check | Ledger value | Independent source | Match? |
|---|---|---|---|
| `exp1` `best_val_loss` | 0.0623630982... | `outputs/models/diffusion_architecture.json`'s `best_val_loss` | **Exact match** |
| `exp2` all 6 sizes' `collapse_frac`, `accuracy`, `macro_f1`, `train_time_sec`, `peak_gpu_mem_gb`, `final_train_loss`, `n_generated` | — | `Roadmap/Stage_1_Diagnosis/Outputs/Experiment_2_Dataset_Scaling/dataset_scaling_metrics.csv` | **Exact match, every field, all 6 rows** |

## 3. Off-by-one record-count mechanism — VERIFIED, not merely plausible

Real requested-vs-actual counts:

| Requested | Actual (`n_train_records_actual`) | Deviation |
|---|---|---|
| 380 | 379 | 1 |
| 1000 | 999 | 1 |
| 2500 | 2500 | 0 |
| 5000 | 5000 | 0 |
| 10000 | 9999 | 1 |
| 17418 (full) | 17418 | 0 |

This matches the mechanism already traced earlier this session (not a
new hypothesis): `run_dataset_scaling.py`'s `stratified_subset()`
independently rounds each of the 6 real classes' share via
`max(1, round(len(idxs) * frac))`, so the aggregate deviation from the
requested total is bounded by per-class rounding, empirically measured
at 0-2 records against this exact real class distribution earlier this
session, and confirmed again here against the actual production run
(0, 0, 0, 1, 1, 1) — consistent, well inside the registered tolerance
(5) in `build_master_log.py`'s `REQUESTED_ACTUAL_PAIRS`. **Verified**,
not "plausible" — the code path has been read, the mechanism traced, and
now two independent measurements (a standalone empirical test earlier,
and this real production run) agree.

## 4. Cross-check against prior "PROJECT CONTEXT" claims — mostly confirmed, one overstated

| Claim | Verified against real data | Verdict |
|---|---|---|
| Full corpus: 17,418 training records | `exp1` and `exp2_full` both show 17418 | **Confirmed** |
| Real-data MentorClassifier: 83.49% accuracy, 0.9554 macro AUC | `real_data_accuracy=0.83489...`, `real_data_macro_auc=0.95545...` | **Confirmed** |
| Generated-data (full-corpus) accuracy 55.3%, macro F1 0.380 | `generated_data_accuracy=0.55333...`, `generated_data_macro_f1=0.38041...` | **Confirmed** |
| Scaling sweep did not exhibit the "identical n_train_records" bug | `n_train_records_actual` = 379/999/2500/5000/9999/17418 — scales correctly | **Confirmed** |
| Collapse fraction non-monotonic (0.94@379, 0.98@5000, 0.91@full) | `collapse_frac` = 0.94/0.6133/0.8/**0.98**/0.9533/**0.91** | **Confirmed exactly** |
| "Accuracy and macro F1 DO improve with scale" | `accuracy`: 0.017→0.307→0.34→0.343→0.373→**0.413** (monotonic increase, endpoints 0.017→0.413). `macro_f1`: 0.023→0.212→0.182→**0.142**→0.183→**0.231** (**NOT monotonic** — dips at 2500 and again more sharply at 5000, below the 1000-record value, before recovering) | **Overstated for macro_f1.** Accuracy does trend up cleanly; macro_f1's overall direction is up (0.023→0.231) but with a real, non-trivial dip in the middle (5000-record run's macro_f1=0.142 is the *lowest* of all six sizes except the 380 baseline) that a clean "improves with scale" framing hides. Treat macro_f1-vs-scale as noisy/non-monotonic, same caveat class as collapse_frac, not a clean trend. |

## 5. Wall-clock rate caveat — restated as unverified, per the hedging correction

`exp2_dataset_scaling_full` took 39,980s for 17,418 records
(~2.30 s/record); the other five sizes' rate: 380→1.53s/rec,
1000→1.51s/rec, 2500→1.47s/rec, 5000→1.46s/rec, 10000→1.45s/rec — a
smooth, mild downward trend that breaks at the full-size run. A
plausible explanation is shared-GPU contention during the ~11-hour full
run, but this is **not independently verified** against server logs or
`nvidia-smi` history — I have no access to check that. Stated as a
caveat for any future compute-cost table, not a settled fact. Does not
gate anything below; not spending further time chasing it.

## 6. Verdict

**PASSED.** Every item is CONFIRMED except the one explicitly marked
NEEDS-REPRODUCTION-caveat (wall-clock rate anomaly, which doesn't gate
anything) and the one overstated claim corrected above (macro_f1 vs.
scale). Proceeding to Stage 2.0.1 (Reproducibility Audit) and Stage
2.0.5 (Repository Audit) is authorized.
