# Stage 3 -- Status

Tracks each Phase 0 task and Phase 1 candidate's status, verdict, and
provenance, mirroring `../Stage_2_Architecture_Investigation/STAGE2_STATUS.md`'s
format. This is the single source of truth for "what has actually
happened in Stage 3" -- `Stage3_Roadmap.md` describes the plan;
`Stage3_Phase0_Report.md` is Phase 0's own findings report; this table
is the at-a-glance status index, kept current as work lands.

| Phase/Candidate | Status | Verdict/Notes | Commit |
|---|---|---|---|
| Phase 0 Task 0.1 (dilution-ratio) | Done | SUPPORTED -- 67.9% net decline block1->block6, Wilcoxon p=6.1e-05 (n=15), well clear of both locked thresholds | `c912a45` (pre-reg) / `50aaf5a` (script + real run) |
| Phase 0 Task 0.2 (final_norm/unproj ablation) | Done | IMPLICATES final_norm/unproj as a fix target, but borderline (retention ratio-of-ratios 0.41 vs. 0.5 threshold) | `50aaf5a` |
| Phase 0 Decision Gate A | Resolved | Both mechanisms implicated (AND, not either-or) -- gain-focused candidates (S3-001..005) justified by 0.1; S3-006 (final_norm/unproj) added, lower priority given 0.2's borderline margin | `4a55874` (Stage3_Phase0_Report.md, provenance-corrected) |
| S3-001 (baseline) | Implemented, smoke-tested | Not yet trained -- shape/gradient-flow + 2-iter optimizer smoke tests both PASS | `85e1b1a` / `ea6ce0d` |
| S3-002 (layerscale) | Implemented, smoke-tested | Not yet trained -- both smoke tests PASS | `85e1b1a` / `ea6ce0d` |
| S3-003 (late_gain) | Implemented, smoke-tested | Not yet trained -- both smoke tests PASS | `85e1b1a` / `ea6ce0d` |
| S3-004 (residual_scaling) | Implemented, smoke-tested | Not yet trained -- both smoke tests PASS | `85e1b1a` / `ea6ce0d` |
| S3-005 (hybrid) | Implemented, smoke-tested | Not yet trained -- both smoke tests PASS | `85e1b1a` / `ea6ce0d` |
| S3-006 (final_norm/unproj) | Implemented, smoke-tested, ready for Wave 3 (2026-07-05) | Deprioritized -- Wave 3, per Task 0.2's borderline margin. Learnable per-channel gain (`final_gamma`) at the final_norm->unproj boundary; both smoke tests PASS; registered in `run_stage3_queue.py`'s `VARIANT_BY_RUN_ID`, ready to queue. | -- |
| Wave 1 GPU training (S3-001/002/003) | **Blocked** | Local MPS device confirmed out-of-memory on a single production-batch-size (32) training step (real error, not assumed) -- requires the remote GPU server per standing project workflow; no direct SSH/GPU access from this session | -- |

## Infrastructure (supporting all of the above, not itself a tracked phase)

| Component | Status | Commit |
|---|---|---|
| `mentor_eval/run_all.py` baseline manifest logging | Done, verified end-to-end | `1d51f8f` |
| `stage3_metadata.py` (per-result metadata.json) | Done, verified end-to-end | `8507fa8` |
| `run_stage3_queue.py` (sequential runner + automated gate) | Done, verified (gate correctly reads real baseline, fails loudly with no candidate metrics yet) | `8507fa8` |
| `run_optimizer_smoke_tests.py` (2-iter optimizer check, all 5) | Done, all PASS | `ea6ce0d` |

## Optimizer note (2026-07-05)

gamma1/gamma2/boost gain parameters were unintentionally included in the
default weight-decay group until commit `0294330` (name-based fix) and
`432395c` (tripwire guard). S3-002 and S3-003 trained under the UNFIXED
optimizer -- their gain-parameter values partially reflect this competing
pressure between task gradient and unnecessary decay. S3-004 onward
trained under the fix. This caveat must be carried into any comparison
table or report that cites S3-002/S3-003's numbers.

## S3-002 retrain decision policy (pre-committed 2026-07-05, before Wave 1/2 results are known)

S3-002 (LayerScale) will be retrained under the corrected optimizer
(post-0294330/432395c) ONLY if one of the following holds once
S3-004 and S3-005 results are in:
  (a) S3-002's original (pre-fix) run already beats baseline on >=1
      primary metric despite training under the weight-decay bug, OR
  (b) Neither S3-004 nor S3-005 improves on baseline on any primary
      metric, making LayerScale worth a clean re-test before ruling
      out gain-based approaches entirely.

This policy is written down now, before Wave 2's results are known, to
prevent the retrain decision being made reactively in a way that could
be biased toward a preferred outcome.
