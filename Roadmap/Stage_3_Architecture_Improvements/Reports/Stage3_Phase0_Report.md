# Stage 3 / Phase 0 -- Report

**Stage 2 is a closed record and is not edited by this report.** This
document lives in Stage 3's own namespace and REFERENCES
`../../Stage_2_Architecture_Investigation/Reports/Stage2_Decision_Report.md`
and `../../Stage_2_Architecture_Investigation/Reports/Stage2_Evidence_Matrix.md`
-- it does not modify either. An earlier version of this work was
appended directly to `Stage2_Decision_Report.md` as a dated addendum;
that was reverted (restored to its pre-Phase-0 state, byte-identical
diff against commit `6d10c46~1`) once flagged as the wrong provenance
model -- Phase 0 is new evidence collected after Stage 2's own closure,
not a retroactive edit to a frozen document.

Implements `Stage3_Phase0_PreRegistration.md`'s locked thresholds. Full
raw output at
`Roadmap/Stage_3_Architecture_Improvements/Outputs/stage3_phase0_task0_1_dilution_ratio/task0_1_raw.json`
and `.../stage3_phase0_task0_2_final_norm_unproj_ablation/task0_2_raw.json`.

## Task 0.1 (dilution-ratio test) -- SUPPORTED

`conditioning_delta(block_k) / total_output_norm(block_k)`, computed
directly (reusing Item 1's own `magnitude_and_consistency` pooling
convention from `Stage2_Architecture_Investigation/Code/common/metrics.py`,
not reimplemented), declines from 0.135 (block 1) to 0.043 (block 6) --
a 67.9% net decline, Wilcoxon signed-rank block1-vs-block6 p=6.1e-05
(n=15 cells), well clear of the pre-registered thresholds (p<0.05 and
decline>=30% required for SUPPORTED).

**This directly tests `Stage2_Decision_Report.md` Conclusion 5b's
dilution hypothesis** (there marked Low-to-Moderate confidence,
untested as a unified metric) -- Phase 0 is the first measurement of
`conditioning-delta / total-output-norm` as one tracked quantity across
blocks. Readers combining Stage 2's conclusions with Stage 3's Phase 0
results should treat 5b as now having direct, high-confidence
supporting evidence from this report -- **Stage 2's own document is not
edited to say so**, per the provenance principle above.

## Task 0.2 (final_norm/unproj causal ablation) -- IMPLICATES, borderline

Conditioning-specific signal retention through `final_norm`->`unproj`
(`retention_ratio_conditioning` = 0.038) is 0.41x the whole-tensor
retention rate through the same two layers (`retention_ratio_whole` =
0.093) -- below the pre-registered 0.5x threshold, but not by a wide
margin (0.41 vs. 0.5), unlike Task 0.1's clean margin. Reported as a
real but moderate-confidence result, not as decisively as Task 0.1's.

**Task 0.2's margin (0.41 vs 0.5 threshold) is real but not as clean as
0.1's -- treated as a lower-priority signal, scheduled for Wave 3, not
dropped.**

## Decision Gate A outcome -- both mechanisms implicated (AND, not a correction)

`Stage2_Decision_Report.md` Sec. 6 Recommendation 5 and
`Stage3_Roadmap.md` Decision Gate A were framed as testing which ONE of
two candidate mechanisms (block-level dilution vs. final_norm/unproj
suppression) applies. Real data shows **both effects present
simultaneously, an AND, not an either-or** -- they are not mutually
exclusive alternatives. Gate A's candidate-selection logic follows
accordingly: the gain-focused candidates (Phase 1's first five,
`S3-001` through `S3-005`) remain justified by Task 0.1's strong
result, AND a 6th candidate (`final_norm`/`unproj` modification,
`S3-006`, not yet implemented) is planned alongside them on the
strength of Task 0.2's result, not substituted in its place. Per Task
0.2's borderline margin, `S3-006` is scheduled for Wave 3 -- lower
priority than Waves 1-2, not excluded.

## Provenance

- Pre-registration: `Stage3_Phase0_PreRegistration.md`
- Scripts: `Code/stage3_phase0/task0_1_dilution_ratio.py`,
  `Code/stage3_phase0/task0_2_final_norm_unproj_ablation.py`
- Raw output: `Outputs/stage3_phase0_task0_1_dilution_ratio/task0_1_raw.json`,
  `Outputs/stage3_phase0_task0_2_final_norm_unproj_ablation/task0_2_raw.json`
- References (not edited): `../../Stage_2_Architecture_Investigation/Reports/Stage2_Decision_Report.md`,
  `../../Stage_2_Architecture_Investigation/Reports/Stage2_Evidence_Matrix.md`
