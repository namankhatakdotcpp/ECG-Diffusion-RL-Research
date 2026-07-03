# Repository Audit Report

_Checks artifact provenance (right checkpoint, not stale, not duplicated) -- NOT numerical correctness of reported results, which is the Verification Gate's job._

## Summary: 0 error(s), 5 warning(s)

## Warnings (review before citing affected artifacts)

- ⚠️ **[stale_artifact]** result figure outputs/results/fig_all_12_leads.png is OLDER than the current diffusion_best.pt (artifact mtime predates checkpoint mtime by 1288595s) -- this artifact was likely generated from a previous checkpoint and never regenerated. Do not cite it as representing the current model.
- ⚠️ **[stale_artifact]** result figure outputs/results/fig_lead_I_single.png is OLDER than the current diffusion_best.pt (artifact mtime predates checkpoint mtime by 1288596s) -- this artifact was likely generated from a previous checkpoint and never regenerated. Do not cite it as representing the current model.
- ⚠️ **[stale_artifact]** result figure outputs/results/fig02_ecg_examples.png is OLDER than the current diffusion_best.pt (artifact mtime predates checkpoint mtime by 1288323s) -- this artifact was likely generated from a previous checkpoint and never regenerated. Do not cite it as representing the current model.
- ⚠️ **[stale_artifact]** result figure outputs/results/fig01_class_distribution.png is OLDER than the current diffusion_best.pt (artifact mtime predates checkpoint mtime by 1288324s) -- this artifact was likely generated from a previous checkpoint and never regenerated. Do not cite it as representing the current model.
- ⚠️ **[metadata_consistency]** cannot cross-check -- missing training_log.csv

## Info

- ℹ️ scanned 1 checkpoint files, 1 distinct content hashes

## Notes on the warnings above (added after manual review, not part of the script's own output)

- **The 4 stale-figure warnings are false positives for this repo, not real findings.** `fig01_class_distribution.png`, `fig02_ecg_examples.png`, `fig_all_12_leads.png`, and `fig_lead_I_single.png` are step01/step03 EDA/data-visualization figures — produced before any model training, never derived from a checkpoint at all. The staleness heuristic assumes every file under `outputs/results/` is model-derived; it isn't in this case. Not citing these as representing the current model was never a risk (they never represented any model).
- **The missing `training_log.csv` warning is real** — `logs/` is empty in the extracted archive. This means the per-epoch training curve for `exp1_baseline_reproduction` isn't available locally; only the final `diffusion_best.pt` and `diffusion_architecture.json` were included in `stage1_results.tar.gz`. If per-epoch analysis is ever needed (e.g. comparing to Experiment 1.5's checkpoint-verification approach), this file needs to be added to a future archive.
- **Checkpoint-only present, no periodic snapshots** — `outputs/models/` has only `diffusion_best.pt`, no `diffusion_ckpt_ep*.pt` files, even though the checkpoint-retention fix (commit `f78c6c2`, keeps last 2 periodic + best) should have produced 2 periodic checkpoints alongside it. Either the archive didn't include them, or something about this run's `save_every`/`n_epochs` combination meant fewer than expected were ever written. Worth confirming which, next archive.

## Extended check: Experiment 2's six per-size checkpoints (not covered by the script above -- nested under Roadmap/Stage_1_Diagnosis/Outputs/Experiment_2_Dataset_Scaling/checkpoints/size_*/, not outputs/models/)

Computed sha256 directly for all six:

```
size_380:   2b39752623f11e70...
size_1000:  549d1b0845fd2498...
size_2500:  64a8f6cd11831132...
size_5000:  f99c956c874dbe59...
size_10000: 269c2b42dd0fce08...
size_full:  b2c7a83dc3e783b2...
```

**No duplicates — all 6 have distinct content hashes.** Confirms each
dataset-scaling run actually produced different weights, not the same
checkpoint copied across sizes.
