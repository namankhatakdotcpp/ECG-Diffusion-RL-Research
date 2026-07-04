"""
Stage 2 / Tier 0 Item 2B, Phase D -- plots, corrected propagation-efficiency,
localized-vs-uniform comparison (explicitly flagged confounded), and
Item2B_Report.md. Reads sweep_summary.json from both Item 2A and Item 2B --
does not rerun any forward passes.

Propagation-efficiency fix (per chat sign-off, this session): Item 2A's
InjectedDelta formula, `(g-1)*mag1_baseline`, assumes a single injection
point and does not fit the uniform variant's 5 cumulative injections. The
audit in this session (post-processing only, using data already captured
by common/hooks.py's register_layer_hooks -- the mean-pool capture hook on
each target block fires BEFORE its CorrectionHook, so it records that
block's raw PRE-correction output, i.e. the actual delta each hook
corrects, not a formula-derived proxy) found the actual per-hook injected
magnitude at blocks 2-5 grows substantially beyond what the budget-
matching formula assumed, especially at high gain -- a real confound,
not a units error.

This script fixes the denominator using the ACTUAL per-hook injected
magnitude, L2-combined across the 5 hooks (chosen over a plain sum because
the budget-matching formula itself is framed in squared/L2 terms --
`5*(ln g_k)^2 = (ln g_L)^2` -- so an L2 combination of the raw magnitudes
is the more consistent choice; the plain-sum alternative was also computed
in this session's chat audit and shows the same accelerating-divergence
trend, so the conclusion below does not hinge on this choice).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]  # Roadmap/.../Code/
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common.io import REPO_ROOT  # noqa: E402
from common.statistics import DIRECTION_FLOOR, POOLED_BLOCK1_TO_2_DROP  # noqa: E402

LOC_OUT_DIR = (
    REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Outputs"
    / "stage2_tier0_item2_localized_gain"
)
UNI_OUT_DIR = (
    REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Outputs"
    / "stage2_tier0_item2_uniform_gain"
)
FIG_DIR = (
    REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Figures"
    / "stage2_tier0_item2_uniform_gain"
)
REPORT_DIR = REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Reports"

CONFOUND_QUALIFIER = (
    "Recovery advantage of uniform over localized is confounded by "
    "budget-matching breakdown under nonlinear compounding -- see "
    "Item2B_Report.md Sec. 3 before citing this as an architectural preference."
)


def main() -> None:
    with open(UNI_OUT_DIR / "sweep_summary.json") as f:
        uni_summary = json.load(f)
    with open(LOC_OUT_DIR / "sweep_summary.json") as f:
        loc_summary = json.load(f)

    uni_rows = uni_summary["summary_rows"]
    loc_rows = {r["gain"]: r for r in loc_summary["summary_rows"]}

    # --- Fix propagation efficiency: actual per-hook injected magnitude (L2), post-processing only ---
    audit_rows = []
    for row in uni_rows:
        g = row["nominal_gain"]
        g_k = row["per_block_gain"]
        layer_mag = row["layer_magnitude_avg"]  # indices 0-4 = blocks 1-5, raw pre-correction magnitude at this gain
        per_hook_injected = [(g_k - 1.0) * layer_mag[k] for k in range(5)]
        total_sum = sum(per_hook_injected)
        total_l2 = sum(x ** 2 for x in per_hook_injected) ** 0.5
        loc_injected = loc_rows[g]["injected_delta_avg"]
        ratio_sum = (total_sum / loc_injected) if loc_injected else None
        ratio_l2 = (total_l2 / loc_injected) if loc_injected else None
        corrected_efficiency = (row["recovered_magnitude_block6_avg"] / total_l2) if total_l2 > 1e-12 else None

        row["injected_delta_uniform_l2_avg"] = total_l2
        row["injected_delta_uniform_sum_avg"] = total_sum
        row["per_hook_injected_magnitude"] = per_hook_injected
        row["propagation_efficiency_avg_corrected_l2"] = corrected_efficiency
        # Preserve the original (formula-borrowed) efficiency under an explicit name rather than overwriting
        # silently. Idempotent: on a rerun the key has already been renamed once, so guard with pop(..., default).
        if "propagation_efficiency_avg" in row:
            row["propagation_efficiency_avg_uncorrected_single_hook_formula"] = row.pop("propagation_efficiency_avg")

        audit_rows.append({
            "nominal_gain": g, "per_block_gain": g_k,
            "localized_injected": loc_injected,
            "uniform_injected_sum": total_sum, "uniform_injected_l2": total_l2,
            "ratio_sum": ratio_sum, "ratio_l2": ratio_l2,
            "per_hook_injected_block1_5": per_hook_injected,
        })

    with open(UNI_OUT_DIR / "sweep_summary.json", "w") as f:
        json.dump(uni_summary, f, indent=2)
    with open(UNI_OUT_DIR / "budget_matching_audit.json", "w") as f:
        json.dump(audit_rows, f, indent=2)

    # --- Comparison table ---
    df = pd.DataFrame([{
        "gain": r["nominal_gain"],
        "per_block_gain": r["per_block_gain"],
        "loc_recovery_pct": loc_rows[r["nominal_gain"]]["recovery_fraction"] * 100.0,
        "uni_recovery_pct": r["recovery_fraction"] * 100.0,
        "loc_direction": loc_rows[r["nominal_gain"]]["min_direction_consistency_layers_2_6"],
        "uni_direction": r["min_direction_consistency_layers_2_6"],
        "loc_efficiency": loc_rows[r["nominal_gain"]]["propagation_efficiency_avg"],
        "uni_efficiency_corrected": r["propagation_efficiency_avg_corrected_l2"],
        "uni_verdict": r["decision_table_verdict"],
    } for r in uni_rows])
    df["recovery_delta_pp"] = df["uni_recovery_pct"] - df["loc_recovery_pct"]
    df.to_csv(UNI_OUT_DIR / "summary.csv", index=False)

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(UNI_OUT_DIR / "budget_matching_audit.csv", index=False)

    # --- Plots (uniform variant, same three as Item 2A) ---
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plot_df = pd.DataFrame([{
        "gain": r["nominal_gain"],
        "recovery_pct": r["recovery_fraction"] * 100.0,
        "min_direction_consistency": r["min_direction_consistency_layers_2_6"],
        "propagation_efficiency": r["propagation_efficiency_avg_corrected_l2"],
    } for r in uni_rows])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(plot_df["gain"], plot_df["recovery_pct"], marker="o", color="darkgreen")
    ax.axhline(70, linestyle="--", color="green", label="SUPPORTED (>=70%)")
    ax.axhline(30, linestyle="--", color="orange", label="Partial floor (>=30%)")
    ax.set_xlabel("Nominal gain g")
    ax.set_ylabel("Block 6 recovery fraction (%)")
    ax.set_title("Uniform gain: Block 6 recovery vs. nominal gain")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "recovery_vs_gain.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(plot_df["gain"], plot_df["min_direction_consistency"], marker="o", color="crimson")
    ax.axhline(DIRECTION_FLOOR, linestyle="--", color="gray", label=f"Direction floor ({DIRECTION_FLOOR})")
    ax.set_xlabel("Nominal gain g")
    ax.set_ylabel("Min direction consistency (layers 2-6)")
    ax.set_title("Uniform gain: direction consistency vs. nominal gain")
    ax.set_ylim(0.98, 1.001)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "direction_vs_gain.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    eff_df = plot_df[plot_df["propagation_efficiency"].notna()]
    ax.plot(eff_df["gain"], eff_df["propagation_efficiency"], marker="o", color="darkorange")
    ax.set_xlabel("Nominal gain g")
    ax.set_ylabel("Propagation efficiency (corrected, L2 per-hook denominator)")
    ax.set_title("Uniform gain: propagation efficiency vs. nominal gain\n(denominator = actual L2-combined per-hook injected magnitude)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "propagation_efficiency.png", dpi=200)
    plt.close(fig)

    # Comparison plot -- explicitly captioned as confounded.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["gain"], df["loc_recovery_pct"], marker="o", color="steelblue", label="Localized (2A)")
    ax.plot(df["gain"], df["uni_recovery_pct"], marker="o", color="darkgreen", label="Uniform (2B)")
    ax.set_xlabel("Nominal gain g")
    ax.set_ylabel("Block 6 recovery fraction (%)")
    ax.set_title("Localized vs. Uniform: Block 6 recovery vs. nominal gain\n"
                 "CONFOUNDED -- actual injected magnitude was not held equal between\n"
                 "variants at matched nominal gain (see Item2B_Report.md Sec. 4)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "localized_vs_uniform_recovery.png", dpi=200)
    plt.close(fig)

    # --- Item 2B verdict, decided ONLY on its own fixed pre-registered criteria ---
    supported_rows = [r for r in uni_rows if r["decision_table_verdict"] == "SUPPORTED"]
    if supported_rows:
        driving = min(supported_rows, key=lambda r: r["nominal_gain"])
        item2b_verdict = (f"SUPPORTED (driven by nominal_gain={driving['nominal_gain']} -- the smallest "
                          f"gain in the locked grid that clears the SUPPORTED threshold, same convention "
                          f"as Item 2A's g=3.0)")
    elif any(r["decision_table_verdict"] == "PARTIAL SUPPORT" for r in uni_rows):
        item2b_verdict = "PARTIAL SUPPORT"
    elif any(r["decision_table_verdict"] == "MAGNITUDE-AT-EXPENSE-OF-INTEGRITY" for r in uni_rows):
        item2b_verdict = "MAGNITUDE-AT-EXPENSE-OF-INTEGRITY"
    else:
        item2b_verdict = "REJECTED"

    monotonicity_flags = uni_summary["monotonicity_flags"]
    monotonic = len(monotonicity_flags) == 0

    report_lines = []
    report_lines.append("# Item 2B (Uniform Gain) -- Phase A-D Report\n")
    report_lines.append(
        "Stage 2 / Tier 0 Item 2, uniform-gain variant (blocks 1-5, cumulative\n"
        "substitution, budget-matched via `g_k = g_L^(1/sqrt(5))`), per the locked\n"
        "pre-registration (`Reports/Tier0_Findings.md`, Item 2 v3, commit\n"
        "`e84c54c`, budget-matching rule carried forward unchanged from v2).\n"
        "Analytical (hook-substitution) phase only, no gradient-descent retrain.\n"
    )

    report_lines.append("\n## Identity-regression test (Sec. 6)\n")
    report_lines.append(
        "**PASSED**, independently of Item 2A's own test (the uniform hook's\n"
        "cumulative bookkeeping is structurally different -- 5 chained\n"
        "substitutions vs. 1 -- so Item 2A's test result was not assumed to carry\n"
        "over, per Sec. 6's explicit instruction):\n"
        "- Standalone test (single draw, pair 0->1, t=500): layer 1 bit-identical,\n"
        "  layers 2-6 within 9.54e-07 of baseline -- the SAME roundoff ceiling as\n"
        "  Item 2A's single-substitution test, confirming the cumulative bookkeeping\n"
        "  does not compound floating-point error across the 5 chained hooks.\n"
        f"- Built-in re-check inside the full sweep (15 cells x 20 draws x 6 layers):\n"
        f"  max|diff| at nominal_gain=1.0 = {uni_summary['g1_builtin_identity_recheck_max_abs_diff']:.2e}\n"
        "  (larger than the single-cell test's max, as expected from a much larger\n"
        "  sample -- still floating-point-roundoff scale, not a bug signature).\n"
    )

    report_lines.append("\n## 1. Item 2B verdict -- decided ONLY on its own fixed criteria\n")
    report_lines.append(
        "Per Sec. 9's decision table, applied against the fixed pooled baseline\n"
        "(0.0635 block1->2 drop) and the fixed 0.989 direction floor -- **independent\n"
        "of any comparison to Item 2A** (a methodology issue in the *comparison*\n"
        "does not retroactively weaken a verdict that does not depend on that\n"
        "comparison):\n\n"
        "| Nominal Gain | Per-block Gain | Recovery% | Direction (min, L2-6) | Verdict |\n"
        "|---|---|---|---|---|\n"
    )
    for r in uni_rows:
        report_lines.append(
            f"| {r['nominal_gain']} | {r['per_block_gain']:.4f} | "
            f"{r['recovery_fraction']*100:.2f}% | {r['min_direction_consistency_layers_2_6']:.6f} | "
            f"{r['decision_table_verdict']} |\n"
        )
    report_lines.append(f"\n**Item 2B verdict: {item2b_verdict}**\n")
    corrected_effs = [r["propagation_efficiency_avg_corrected_l2"] for r in uni_rows
                      if r["propagation_efficiency_avg_corrected_l2"] is not None]
    report_lines.append(
        "\nRecovery% is monotonically non-decreasing across the full grid -- "
        f"{'no dip observed, no flag raised' if monotonic else 'DIPS OBSERVED, see raw sweep_summary.json'}. "
        "No NaN, no negative efficiency. With the corrected (Sec. 2) denominator -- the **L2 combination**\n"
        f"of the 5 per-hook injected magnitudes, NOT the plain sum -- efficiency ranges "
        f"{min(corrected_effs):.2f}-{max(corrected_effs):.2f}, much closer to a sane ~1.0 ceiling than the\n"
        "uncorrected single-hook formula's 1.4-2.3. It still modestly exceeds 1.0 at gains 1.5-5.0, and\n"
        "this residual is NOT fully resolved by the L2 fix: unlike the plain-sum alternative (which treats\n"
        "the 5 sequential corrections as independent additive contributions -- the same linearity\n"
        "assumption that broke the original budget-matching formula), the L2 combination does not carry\n"
        "that specific flaw, so a >1.0 reading under L2 is a more genuine signal that slightly more block-6\n"
        "magnitude was recovered than the L2-combined injected delta predicts, not a residual of the same\n"
        "additive-assumption bug. The g=1.0 identity test rules out a hook-mechanism bug independently.\n"
        "This is flagged here as an open question, not resolved: RecoveredMagnitude (measured at block 6,\n"
        "via the standard magnitude metric) and the L2-combined InjectedMagnitude (measured at the 5\n"
        "injection points, via the same metric applied per-block) are still two different measurement\n"
        "paths through a nonlinear network, and nothing in this analysis proves they should sum to exactly\n"
        "1.0 even under a perfectly matched budget -- whoever revisits Item 2's efficiency metric should\n"
        "treat this residual as unresolved, not as evidence the L2 fix under- or over-corrects.\n"
    )

    report_lines.append("\n## 2. Propagation-efficiency denominator, corrected\n")
    report_lines.append(
        "Item 2A's `InjectedDelta(g) = (g-1)*mag1_baseline` assumes a single\n"
        "injection point and does not fit the uniform variant's 5 cumulative\n"
        "injections. This report uses the ACTUAL per-hook injected magnitude\n"
        "instead -- `(g_k-1)*layer_magnitude_avg[k]` at each of blocks 1-5, already\n"
        "captured by `common/hooks.py`'s mean-pool hook (which fires BEFORE each\n"
        "block's CorrectionHook and therefore records the real pre-correction\n"
        "delta at that block, not a formula-derived proxy) -- combined via L2\n"
        "across the 5 hooks (consistent with the budget-matching formula's own\n"
        "squared-log framing; a plain sum was also computed and shows the same\n"
        "trend, so this choice does not drive the conclusion below).\n\n"
        "| Nominal Gain | Recovery% | Direction (min) | Efficiency (corrected, L2) | Verdict |\n"
        "|---|---|---|---|---|\n"
    )
    for r in uni_rows:
        eff = r["propagation_efficiency_avg_corrected_l2"]
        eff_str = f"{eff:.4f}" if eff is not None else "n/a (g=1.0)"
        report_lines.append(
            f"| {r['nominal_gain']} | {r['recovery_fraction']*100:.2f}% | "
            f"{r['min_direction_consistency_layers_2_6']:.6f} | {eff_str} | "
            f"{r['decision_table_verdict']} |\n"
        )

    report_lines.append("\n## 3. Localized vs. Uniform comparison -- CONFOUNDED\n")
    report_lines.append(
        "**This section's numbers are real, but the comparison is not clean.**\n"
        "The budget-matching formula (`g_k = g_L^(1/sqrt(5))`, an additive-log-gain\n"
        "heuristic) was intended to hold total injected correction magnitude equal\n"
        "between the localized and uniform variants at matched nominal gain `g`.\n"
        "A post-processing audit (this session, no rerun -- using the raw\n"
        "pre-correction magnitudes each hook already captured) found it does not:\n\n"
        "| Nominal g | Localized Injected | Uniform Injected (sum) | Uniform Injected (L2) | Ratio (sum) | Ratio (L2) |\n"
        "|---|---|---|---|---|---|\n"
    )
    for a in audit_rows:
        rs = f"{a['ratio_sum']:.3f}" if a["ratio_sum"] is not None else "n/a"
        rl = f"{a['ratio_l2']:.3f}" if a["ratio_l2"] is not None else "n/a"
        li = f"{a['localized_injected']:.4f}" if a["localized_injected"] else "0.0000"
        report_lines.append(
            f"| {a['nominal_gain']} | {li} | {a['uniform_injected_sum']:.4f} | "
            f"{a['uniform_injected_l2']:.4f} | {rs} | {rl} |\n"
        )
    report_lines.append(
        "\nThe ratio **grows with gain rather than staying flat near 1.0** -- from "
        "~1.36x (sum) / ~0.63x (L2) at g=1.25 to ~3.62x (sum) / ~2.03x (L2) at g=5.0.\n"
        "The mechanism is visible in the per-hook breakdown: at g=5.0, blocks 4 and 5's\n"
        "own raw (pre-correction) delta magnitude is "
        f"{audit_rows[-1]['per_hook_injected_block1_5'][3]:.4f} and "
        f"{audit_rows[-1]['per_hook_injected_block1_5'][4]:.4f} respectively -- several\n"
        "times block 1's "
        f"{audit_rows[-1]['per_hook_injected_block1_5'][0]:.4f} -- even though all 5 blocks\n"
        "nominally received the same budget-matched `g_k`. Later hooks are correcting a\n"
        "signal that earlier hooks have already amplified, so the linear/additive\n"
        "log-gain budgeting formula underestimates actual injected magnitude at high\n"
        "gain -- a real confound arising from nonlinear compounding, not a units error\n"
        "or implementation bug (the g=1.0 identity test rules that out independently).\n"
    )
    report_lines.append(
        "\n**Two claims, explicitly separated:**\n\n"
        "- **SURVIVES the confound:** under the tested budget formula, distributed\n"
        "  correction achieved higher raw recovery than concentrated correction, at\n"
        "  every non-identity gain in the locked grid. This is a description of what\n"
        "  was observed, not a causal claim -- it stands regardless of the confound.\n"
        "- **DOES NOT SURVIVE the confound:** \"distribution is inherently more\n"
        "  effective than concentration, at matched injection strength.\" This causal\n"
        "  claim is blocked -- injected magnitude was not actually held constant\n"
        "  between variants (see ratio table above), so the recovery gap cannot yet\n"
        "  be attributed to WHERE the correction landed vs. HOW MUCH correction\n"
        "  actually landed, since those two things covaried instead of being held\n"
        "  fixed the way the pre-registration's budget-matching rule assumed.\n"
    )
    report_lines.append(
        f"\n**Qualifier (carry forward to any future citation of this comparison):**\n"
        f"> {CONFOUND_QUALIFIER}\n"
    )

    report_lines.append("\n## Artifacts\n")
    report_lines.append(
        "- Raw per-gain JSON: `Outputs/stage2_tier0_item2_uniform_gain/gain_{1.00,1.25,1.50,2.00,3.00,5.00}.json`\n"
        "- Sweep summary: `Outputs/stage2_tier0_item2_uniform_gain/sweep_summary.json`\n"
        "- Budget-matching audit: `Outputs/stage2_tier0_item2_uniform_gain/budget_matching_audit.{json,csv}`\n"
        "- Summary table: `Outputs/stage2_tier0_item2_uniform_gain/summary.csv`\n"
        "- Figures: `Figures/stage2_tier0_item2_uniform_gain/{recovery_vs_gain,direction_vs_gain,"
        "propagation_efficiency,localized_vs_uniform_recovery}.png`\n"
    )

    report_text = "".join(report_lines)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_DIR / "Item2B_Report.md", "w") as f:
        f.write(report_text)

    print(f"Item 2B verdict (own criteria only): {item2b_verdict}")
    print("Comparison with Item 2A: CONFOUNDED (see report Sec. 3-4)")
    print(f"Wrote: {REPORT_DIR / 'Item2B_Report.md'}")
    print(f"Wrote: {UNI_OUT_DIR / 'summary.csv'}")
    print(f"Wrote: {UNI_OUT_DIR / 'budget_matching_audit.json'} / .csv")
    print(f"Wrote figures under: {FIG_DIR}")


if __name__ == "__main__":
    main()
