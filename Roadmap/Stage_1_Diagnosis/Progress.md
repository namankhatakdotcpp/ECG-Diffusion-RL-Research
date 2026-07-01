# Stage 1 Progress

Last updated: 2026-07-02

- [x] Read all reports/markdown/PDF roadmap
- [x] Read core code (diffusion model, CFG, training loop)
- [x] Deep read of mentor_eval evaluation/diagnostic modules
- [x] Architecture.md drafted
- [x] Roadmap/ workspace scaffolded
- [ ] Experiment 1: Baseline Reproduction (code ready, awaiting GPU server)
- [ ] Experiment 1.5: Checkpoint Verification (code ready, needs Exp1's per-epoch checkpoints)
- [ ] Experiment 2: Dataset Scaling (code ready, awaiting GPU server)
- [ ] Experiment 2.5: Training Curves (built into Experiment 2's script)
- [ ] Experiment 3: Directional Conditioning Analysis (code ready, awaiting GPU server)
- [ ] Experiment 3.5: Layer-wise Direction Probe (code ready, needs Exp1's checkpoint)
- [x] Experiment 4: MentorClassifier Verification (COMPLETE - ran locally, no GPU needed - see Reports/classifier_validation_report.md)
- [x] Experiment 4.5: Feature Drift Visualization (real+noise half COMPLETE, ran locally; generated-sample overlay pending Exp1's checkpoint)
- [ ] Experiment 5: Decision Report (blocked on 1, 1.5, 2, 2.5, 3, 3.5)
- [x] Master runner (`run_stage1.sh`) + results collector (`collect_stage1_results.py`) written and tested against currently-available outputs

## Experiment 4 key result
AFIB attraction ratio peaks at 3.58x chance at sigma=0.5 (89.5% of all noise-driven misclassifications land on AFIB); role shifts to NSTEMI at extreme noise (sigma=2.0). AFIB is confirmed unreliable as a conditioning-quality signal.

## Next action
Run `bash Roadmap/Stage_1_Diagnosis/run_stage1.sh` on the GPU server. It
trains the baseline, runs 1.5/2(+2.5)/3/3.5 automatically, skips Experiment 4
(already done), refreshes 4.5 with the generated-sample overlay, and writes
`Reports/Stage1_Results_Digest.md`. Hand that digest + the Outputs/Figures
trees back for the narrative reports and Stage1_Final_Report.md.
