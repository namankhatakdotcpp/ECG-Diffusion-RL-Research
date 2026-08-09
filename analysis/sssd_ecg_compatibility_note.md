# SSSD-ECG Compatibility Note (Task 2, pre-training investigation)

**Source**: `AI4HealthUOL/SSSD-ECG`, cloned read-only to `/Users/naman/Documents/HCL_Internship/SSSD-ECG-baseline` (sibling to this repo, not inside it). Investigated: `README.md`, `src/sssd/train.py`, `src/sssd/config/config_SSSD_ECG.json`, `src/ptb_xl/clinical_ts/ecg_utils.py`.

This is a read-only investigation — nothing has been trained, no data pipeline was touched on either side.

## What matches

| Aspect | Ours | SSSD-ECG | Match? |
|---|---|---|---|
| Source dataset | PTB-XL | PTB-XL | ✓ |
| Sampling rate | 100 Hz | 100 Hz (downsampled from the 500 Hz `filename_hr` records, same as us) | ✓ |
| Segment length | 1000 samples (10s) | 1000 samples (`segment_length: 1000` @ 100Hz = 10s) | ✓ |
| Normalization style | Per-lead z-score (global mean/std, `preprocessing_stats.json`) | Per-lead z-score, global mean/std averaged across records (`dataset_get_stats`, `axis=0`) | ✓ conceptually — same style, but the actual mean/std values will differ (different lead subset, different train split composition) |

## What does NOT match — real mismatches to document, not silently reconcile

1. **Conditioning label space is fundamentally different.**
   Their model conditions on a **71-dimensional multi-hot vector** over all PTB-XL SCP codes with count ≥10 (`label_embed_classes: 71` in `config_SSSD_ECG.json`, built from `label_all` in `ecg_utils.py`), not on the 5-superclass (or our 6-class incl. OTHER) diagnostic label we use. There is no built-in "NORM vs MI only" training mode — their `train.py` always trains on the full 71-code label space; restricting to a NORM/MI subset would require **either** filtering the training data to NORM/MI-only records (still conditioning on 71-dim codes, most of which would be near-constant), **or** a real code change to their label embedding to take a 2-class one-hot instead — a nontrivial modification to `wavenet_config.label_embed_classes` and however the label tensor is constructed in `train.py`.

2. **They only diffuse 8 of the 12 leads; the other 4 are linearly derived, not generated.**
   `train.py` selects `index_8 = [I, V1, V2, V3, V4, V5, V6, aVF]` as the actual model input/output (`in_channels: 8`, `out_channels: 8`). The remaining 4 leads (`II, III, aVR, aVL`) are excluded from the diffusion model entirely — presumably reconstructed afterward via the standard Einthoven/Goldberger linear lead relationships (not confirmed in code read so far; `index_4` is selected but I did not find where it's used downstream in the portion of `train.py` read). **Our model diffuses and evaluates all 12 leads directly.** Any head-to-head classifier evaluation must be clear about this: if 4 of SSSD-ECG's "generated" leads are actually deterministic linear functions of the other 8, that's an advantage/artifact worth calling out, not something to normalize away.

3. **Lead channel order differs.**
   Their `channel_stoi_default`: `{i:0, ii:1, v1:2, v2:3, v3:4, v4:5, v5:6, v6:7, iii:8, avr:9, avl:10, avf:11}`. Ours (`config.yaml`): `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`. Different index-to-lead mapping — if their generated `.npy` output is ever fed into our `classification_validation.py`, the channels must be explicitly reordered to our convention first, or the classifier will silently score the wrong leads against the wrong channel indices.

4. **Diffusion schedule differs substantially.** `T=200` (vs our `T=1000`), linear-style `beta_0=1e-4, beta_T=0.02` (vs our cosine schedule). Not a data-pipeline mismatch, but relevant context: their model is a much shorter diffusion chain over a structured state-space (S4) backbone, not a Transformer — architecturally unrelated to ours beyond both being conditional diffusion.

5. **No train/val split control exposed.** `train.py` loads pre-baked `ptbxl_train_data.npy` / `ptbxl_train_labels.npy` with no CLI/config option to pass our exact `strat_fold`-based train/val split (`step01`/`step02` in this repo use PTB-XL's `strat_fold` column). Matching splits exactly would require regenerating their `.npy` files ourselves from `prepare_data_ptb_xl()` using our fold assignment — feasible (the function takes a df with `strat_fold`), but is a real implementation step, not a flag.

6. **Batch size / iteration count differ from ours** (`batch_size: 8`, `n_iters: 100000`, `lr: 2e-4`, iteration-based not epoch-based) — not a blocker, just means "same number of epochs" isn't a meaningful comparison; would need to pick a stopping point independently (e.g. by validation loss plateau) rather than matching our 200-epoch config.

## Assessment

A **literal NORM/MI-only, exact-split head-to-head is not a drop-in run** — it requires:
- Regenerating their PTB-XL `.npy` files with our exact train/val fold split (moderate effort, their code supports it).
- Either accepting the 71-dim label space as-is (training on NORM/MI-filtered records but keeping the 71-code label format) or modifying `wavenet_config.label_embed_classes`/label construction for true 2-class conditioning (real code change to their repo).
- Reordering their generated leads to our channel convention, and explicitly deciding what to do about the 4 linearly-derived leads before scoring with `classification_validation.py`.

None of these are large, but together they're a half-day-to-a-day task, not a "clone and run" comparison — and item 2 (derived vs. generated leads) means even a clean run wouldn't be a perfectly fair architecture-for-architecture comparison without a footnote.

## Recommendation

I have not started any training. Given the effort above, there are two reasonable paths:

- **(A)** Proceed with the real head-to-head: regenerate their data with our split, decide on label-space handling, train, reorder leads, evaluate with our `classification_validation.py`. Real apples-to-apples number, ~half day to a day of work plus training time.
- **(B)** Cite their published paper numbers as a reference point with an explicit caveat (different label space, different lead-generation scope, no split control) instead of re-running their model.

I'd lean toward (B) for the paper unless the head-to-head number is important enough to justify the extra implementation work and the caveats in item 2 above — but this is your call, not mine. **STOP — waiting for your decision before doing anything further on Task 2.**
