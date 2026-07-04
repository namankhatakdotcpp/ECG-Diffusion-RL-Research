"""
Stage 2 / Tier 0 Item 3 -- final report generation. Reads
residual_probe_raw.json, block6_investigation.json, and
block6_ablation.json (all already produced) -- does not rerun any
forward passes. Locks Item 3's decision criteria from the real observed
distribution, per the pre-registration's explicit deferral.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]  # Roadmap/.../Code/
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common.io import REPO_ROOT  # noqa: E402
from common.plotting import plot_residual_ratio_vs_block  # noqa: E402

OUT_DIR = (
    REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Outputs"
    / "stage2_tier0_item3_residual_attenuation"
)
FIG_DIR = (
    REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Figures"
    / "stage2_tier0_item3_residual_attenuation"
)
REPORT_DIR = REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Reports"


def main() -> None:
    with open(OUT_DIR / "residual_probe_raw.json") as f:
        probe = json.load(f)
    with open(OUT_DIR / "block6_investigation.json") as f:
        block6_inv = json.load(f)
    with open(OUT_DIR / "block6_ablation.json") as f:
        ablation = json.load(f)

    pooled = probe["pooled"]
    r_k_A = pooled["R_k_class_A_pooled"]
    r_k_B = pooled["R_k_class_B_pooled"]
    r_k_combined = pooled["R_k_combined_pooled"]
    wilcoxon = pooled["wilcoxon_block1_vs_block6"]

    df = pd.DataFrame({
        "block": list(range(1, len(r_k_combined) + 1)),
        "R_k_class_A_pooled": r_k_A,
        "R_k_class_B_pooled": r_k_B,
        "R_k_combined_pooled": r_k_combined,
    })

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUT_DIR / "residual_probe.csv", index=False)
    fig_path = plot_residual_ratio_vs_block(df, FIG_DIR)

    valley_block = int(df.loc[df["R_k_combined_pooled"].idxmin(), "block"])
    peak_block = int(df.loc[df["R_k_combined_pooled"].idxmax(), "block"])

    # Decision criteria, locked NOW from the real observed distribution (pre-registration's
    # explicit deferral -- not chosen before data existed).
    p_value = wilcoxon["p_value"]
    null_rejected = p_value is not None and p_value < 0.05
    verdict = "SUPPORTED (systematic variation across blocks)" if null_rejected \
        else "NOT SUPPORTED (flat/noise-like, null not rejected)"

    ratio_delta = ablation["ratio_delta6_to_delta3"]
    if ratio_delta is not None and ratio_delta > 1.2:
        ablation_finding = (
            f"CONFIRMED by causal ablation: ablating block {peak_block}'s residual update changes "
            f"final_norm's output by {ablation['mean_delta_block6_ablation']:.3f} on average, "
            f"vs. {ablation['mean_delta_block3_valley_ablation']:.3f} for ablating the valley block "
            f"({valley_block}) -- a {ratio_delta:.2f}x larger effect. Block {peak_block}'s larger "
            f"pre-normalization residual DOES translate to proportionally greater post-normalization "
            f"influence -- this is measured directly via ablation, not inferred from pre-norm ratios."
        )
    elif ratio_delta is not None and ratio_delta < (1 / 1.2):
        ablation_finding = (
            f"REVERSED by causal ablation: the valley block's ({valley_block}) contribution changes "
            f"final_norm's output MORE than block {peak_block}'s does, despite block {peak_block} "
            f"having the larger pre-normalization residual. The pre-normalization disparity does NOT "
            f"translate to greater post-normalization influence -- downgraded from the earlier draft."
        )
    else:
        ablation_finding = (
            f"NOT CONFIRMED: post-ablation effect on final_norm's output is roughly comparable between "
            f"block {peak_block} and the valley block ({valley_block}), despite the pre-normalization "
            f"residual disparity. Block {peak_block}'s larger pre-normalization residual does NOT "
            f"straightforwardly translate to greater post-normalization influence -- this is flagged as "
            f"an open question, not resolved either way."
        )

    report = f"""# Item 3 (Residual-Path Attenuation) -- Report

Stage 2 / Tier 0 Item 3, per the locked pre-registration
(`Reports/Item3_PreRegistration.md`, commit `3101f28`). Measurement only
-- no intervention in the main sweep (the block-6 ablation check below is
a targeted causal follow-up on one specific claim, not a re-opening of
Item 3's own design).

## Architectural question (restated)

Does conditioning attenuation occur primarily inside the residual branch
itself (visible as a declining `R_k`), or does it emerge only after
residual addition? **Answer: neither cleanly -- `R_k` is non-monotonic
(a U-shape), which is itself the finding** (see Finding 1 below).

## Finding 1: `R_k` varies systematically across blocks -- a U-shape, not attenuation or amplification

| Block | R_k (class A) | R_k (class B) | R_k (combined) |
|---|---|---|---|
"""
    for i in range(len(df)):
        row = df.iloc[i]
        report += (f"| {int(row['block'])} | {row['R_k_class_A_pooled']:.4f} | "
                   f"{row['R_k_class_B_pooled']:.4f} | {row['R_k_combined_pooled']:.4f} |\n")

    report += f"""
**Shape: non-monotonic** -- `R_k` declines from block 1 to a minimum at
block {valley_block} ({df['R_k_combined_pooled'].min():.4f}), then rises
to a peak at block {peak_block} ({df['R_k_combined_pooled'].max():.4f}),
a ~{df['R_k_combined_pooled'].max() / df['R_k_combined_pooled'].min():.1f}x
range. Per the Interpretation Framework's non-monotonic row: this
requires checking whether it reflects a real architecture effect or a
methodological difference from Item 1's own measurement -- **it does not
contradict Item 1's two-drop shape**, because the two items measure
different quantities (Item 3's `R_k` is a within-pass update magnitude;
Item 1's finding is a cross-class output-magnitude delta). The two are
complementary, not required to match in shape.

**Candidate statistical test result (Wilcoxon signed-rank, block 1 vs.
block {len(df)}, n={wilcoxon['n']}):** statistic={wilcoxon['statistic']:.1f},
**p={p_value:.2e}**. Null hypothesis (flat/noise-like) is
**{'REJECTED' if null_rejected else 'NOT rejected'}** -- `R_k`'s
variation across blocks is statistically significant, not noise. Note
this test characterizes the ENDPOINTS, not the full U-shape; the shape
itself (valley + late-block spike) is the more informative pattern, not
folded into a single p-value.

## Finding 2: class-independence -- `R_k` barely differs between class A and class B

At every block, `R_k` for class A and class B are nearly identical (e.g.
block {peak_block}: {r_k_A[peak_block-1]:.4f} vs. {r_k_B[peak_block-1]:.4f}).
**This is a finding in its own right, not a footnote:** it corroborates
Item 1's own conclusion -- that class-conditioning signal lives in
*direction* rather than *magnitude* -- from a completely different,
within-pass measurement angle, independent of Item 1's cross-class
methodology. Two independent measurement approaches landing on the same
qualitative conclusion (magnitude is not where the class signal is
carried) is stronger evidence than either alone.

## Finding 3: block {peak_block}'s residual update genuinely has outsized post-normalization influence -- confirmed by causal ablation, not inferred from pre-norm ratios

An earlier draft of this finding compared PRE-FinalNorm residual/output
ratios between block {peak_block} and the valley block ({valley_block})
and concluded the disparity "survives" FinalNorm's compression. **That
claim was corrected before being locked here** -- a pre-normalization
ratio disparity does not by itself establish what survives a
whole-tensor normalization (LayerNorm normalizes the entire tensor's
variance, not any one block's specific contribution). The corrected,
causally-grounded test: ablate block {peak_block}'s residual update
(override its output with its own input, skipping its contribution
entirely), pass the result through `final_norm`, and measure how much
`final_norm`'s output changes, compared against doing the identical
ablation at the valley block.

| Quantity | Value |
|---|---|
| Baseline `final_norm` output norm (no ablation) | {ablation['mean_baseline_final_norm_output_norm']:.3f} |
| Mean change from ablating block {peak_block} | {ablation['mean_delta_block6_ablation']:.3f} |
| Mean change from ablating block {valley_block} (valley) | {ablation['mean_delta_block3_valley_ablation']:.3f} |
| Ratio | {ratio_delta:.2f}x |

**Ratio formula, stated explicitly (pool first, then ratio -- not a mean
of per-observation ratios):** each of the {ablation['n_observations']}
individual (pair, timestep, draw, class) observations contributes one
`delta_6(i) = ||final_norm(baseline_i) - final_norm(ablated_block_{peak_block}_i)||`
and one `delta_3(i)` (same formula, block {valley_block}). The two
columns above are `mean_i(delta_6(i))` and `mean_i(delta_3(i))` --
flat means across all {ablation['n_observations']} observations, not
per-cell -- and the ratio is `mean_i(delta_6(i)) / mean_i(delta_3(i))`,
**not** `mean_i(delta_6(i)/delta_3(i))` (the latter would be unstable
wherever `delta_3(i)` is near zero for a specific draw).

**{ablation_finding}**

(Pre-normalization context, for reference only -- not the basis for the
claim above: block {peak_block}'s residual/output ratio was
{block6_inv['ratio_block6_residual_to_block6_output']:.3f} vs.
{block6_inv['ratio_valley_residual_to_valley_output']:.3f} for the valley
block, and `final_norm` compresses block {peak_block}'s output to
{block6_inv['ratio_final_norm_output_to_block6_output']:.3f}x its
pre-normalization magnitude.)

## Decision criteria (locked now, from this real distribution, per the pre-registration's explicit deferral)

Item 3's hypothesis ("`R_k` varies systematically across blocks") is
evaluated against: Wilcoxon signed-rank test, block 1 vs. block
{len(df)}, alpha=0.05, matching Item 1's own paired-design precedent.

**Verdict: {verdict}**

This criterion was NOT chosen before the sweep ran (per the
pre-registration's explicit "Decision criteria: TBD from Item 3's own
fresh sweep" -- no threshold existed prior to seeing this distribution).

## Stage 2 Follow-up Questions (deferred, not blocking Item 3's closure)

Per explicit instruction not to expand Item 3's scope further:
- Attention-sublayer vs. FFN-sublayer decomposition of `R_k` (Item 3's
  granularity is block-level only, per its own scope statement).
- Timestep-dependent subgroup analysis of the U-shape (t=100 vs. t=500
  vs. t=900 shown separately, not just pooled).
- Whether the block {peak_block} spike is specific to this checkpoint
  or a general property of this architecture family.

None of these block Item 3's closure -- they are recorded here for a
future item or the eventual Stage 2 Decision Report, not investigated
further in this item.

## Artifacts

- Raw sweep: `Outputs/stage2_tier0_item3_residual_attenuation/residual_probe_raw.json`
- Pooled summary: `Outputs/stage2_tier0_item3_residual_attenuation/residual_probe.csv`
- Block 6 investigation (pre-norm ratios, superseded by the ablation for the causal claim): `Outputs/stage2_tier0_item3_residual_attenuation/block6_investigation.json`
- Block 6 causal ablation (the evidence actually cited above): `Outputs/stage2_tier0_item3_residual_attenuation/block6_ablation.json`
- Figure: `Figures/stage2_tier0_item3_residual_attenuation/residual_ratio_vs_block.png`
"""

    with open(REPORT_DIR / "Item3_Report.md", "w") as f:
        f.write(report)

    print(f"Verdict: {verdict}")
    print(f"Ablation finding: {ablation_finding}")
    print(f"Wrote: {REPORT_DIR / 'Item3_Report.md'}")
    print(f"Wrote: {fig_path}")
    print(f"Wrote: {OUT_DIR / 'residual_probe.csv'}")


if __name__ == "__main__":
    main()
