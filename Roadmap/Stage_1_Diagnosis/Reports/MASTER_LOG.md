# Master Experiment Log — Stage_1_Diagnosis

_Auto-generated from `results_ledger.jsonl`. Do not hand-edit — edits are overwritten on the next run._

## Flags (automated sanity checks)

**These require human review before the affected results can be trusted:**

- ⚠️ [exp1_baseline_reproduction] requested batch_size=32 but recorded n_train_records_actual=17418 — the parameter may not have reached the training/generation code
- ⚠️ [exp2_dataset_scaling_380] requested dataset_size_requested=380 but recorded n_train_records_actual=379 — the parameter may not have reached the training/generation code
- ⚠️ [exp2_dataset_scaling_1000] requested dataset_size_requested=1000 but recorded n_train_records_actual=999 — the parameter may not have reached the training/generation code
- ⚠️ [exp2_dataset_scaling_10000] requested dataset_size_requested=10000 but recorded n_train_records_actual=9999 — the parameter may not have reached the training/generation code
- ⚠️ [exp2_dataset_scaling_full] requested dataset_size_requested='full' but recorded n_train_records_actual=17418 — the parameter may not have reached the training/generation code
- ⚠️ [exp2_dataset_scaling] metric 'n_generated' is byte-identical across 6 runs with different params — check whether the varying parameter actually reached the underlying training/generation code

## All Experiment Runs

| Experiment | Stage | Status | Duration (s) | Timestamp | n_train_records_actual | best_val_loss | checkpoint_saved | real_data_accuracy | real_data_macro_f1 | real_data_macro_auc | generated_data_accuracy | generated_data_macro_f1 | generated_data_excluded_classes | train_time_sec | peak_gpu_mem_gb | final_train_loss | accuracy | macro_f1 | collapse_frac | n_generated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exp1_baseline_reproduction | Stage_1_Diagnosis | success | 24832.35 | 2026-07-02T14:09:54.286171+00:00 | 17418 | 0.062363098227027534 | True | 0.8348946135831382 | 0.7157811513232513 | 0.9554486012020065 | 0.5533333333333333 | 0.380405205560485 | ['AFIB'] |  |  |  |  |  |  |  |
| exp2_dataset_scaling_380 | Stage_1_Diagnosis | success | 619.1 | 2026-07-02T21:05:44.134740+00:00 | 379 |  |  |  |  |  |  |  |  | 579.6560924053192 | 6.12745216 | 0.17783549766648898 | 0.016666666666666666 | 0.023214285714285715 | 0.94 | 300 |
| exp2_dataset_scaling_1000 | Stage_1_Diagnosis | success | 1547.97 | 2026-07-02T21:16:03.257198+00:00 | 999 |  |  |  |  |  |  |  |  | 1508.3502023220062 | 6.192649216 | 0.13700343980904547 | 0.30666666666666664 | 0.21186819916152433 | 0.6133333333333333 | 300 |
| exp2_dataset_scaling_2500 | Stage_1_Diagnosis | success | 3723.07 | 2026-07-02T21:41:51.246135+00:00 | 2500 |  |  |  |  |  |  |  |  | 3683.457941532135 | 6.124888064 | 0.1011145750586039 | 0.34 | 0.18215615921336875 | 0.8 | 300 |
| exp2_dataset_scaling_5000 | Stage_1_Diagnosis | success | 7332.44 | 2026-07-02T22:43:54.338724+00:00 | 5000 |  |  |  |  |  |  |  |  | 7292.552843570709 | 6.192915456 | 0.0902793717642243 | 0.3433333333333333 | 0.14165796991762478 | 0.98 | 300 |
| exp2_dataset_scaling_10000 | Stage_1_Diagnosis | success | 14592.88 | 2026-07-03T00:46:06.801040+00:00 | 9999 |  |  |  |  |  |  |  |  | 14548.828769922256 | 6.26004992 | 0.08035741275988328 | 0.37333333333333335 | 0.18310510732790525 | 0.9533333333333334 | 300 |
| exp2_dataset_scaling_full | Stage_1_Diagnosis | success | 40048.02 | 2026-07-03T04:49:19.698721+00:00 | 17418 |  |  |  |  |  |  |  |  | 39980.326567173004 | 6.124486656 | 0.07237188656376127 | 0.41333333333333333 | 0.23099875242215911 | 0.91 | 300 |


## Failures and Crashes

_None._