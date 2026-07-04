"""
Stage 2 Tier 0 -- shared falsification-criteria / decision-table logic.

LIFTED (copied, not moved) from
stage2_tier0_item2_localized_gain/item2_gain_sweep.py (Item 2A) -- the
decision table (Tier0_Findings.md, Item 2 v3, Sec. 9) and the
authorized monotonicity gate addition (per-item chat sign-off after
Item 2A's A2). Both apply unchanged to Item 2B (uniform variant) and any
later item that reuses this decision table.
"""

from __future__ import annotations

DIRECTION_FLOOR = 0.989
RECOVERY_SUPPORTED = 0.70
RECOVERY_PARTIAL = 0.30
POOLED_BLOCK1_TO_2_DROP = 0.0635  # Item 1 pooled n=15 stat, Tier0_Findings.md Sec. 9


def decision_table_verdict(recovery_fraction: float, min_direction_consistency: float) -> str:
    """Item 2 v3 Sec. 9's four-way table, applied per gain value
    independently (no monotonicity/unanimity requirement across a sweep --
    the closing clarification in Tier0_Findings.md, commit e84c54c)."""
    if min_direction_consistency < DIRECTION_FLOOR:
        return "MAGNITUDE-AT-EXPENSE-OF-INTEGRITY"
    elif recovery_fraction >= RECOVERY_SUPPORTED:
        return "SUPPORTED"
    elif recovery_fraction >= RECOVERY_PARTIAL:
        return "PARTIAL SUPPORT"
    else:
        return "REJECTED"


def monotonicity_check(gain_grid: list[float], recoveries: list[float],
                        directions: list[float], efficiencies: list[float | None]) -> list[dict]:
    """Authorized gate addition (chat sign-off after Item 2A's A2, before
    A3): flags non-monotonic Block-6-recovery dips, then classifies each
    dip by whether direction consistency and propagation efficiency ALSO
    dip at the same gain step.
      - Correlated dip across all three metrics -> plausible real
        non-linear interaction; not a bug, note and proceed.
      - Recovery dips alone (direction/efficiency stay smooth) -> bug
        signature; caller should stop and investigate.
    Returns a list of flag dicts (empty list = fully monotonic, no flags)."""
    flags = []
    for i in range(1, len(recoveries)):
        if recoveries[i] < recoveries[i - 1]:
            dir_dips = directions[i] < directions[i - 1]
            eff_dips = (efficiencies[i] is not None and efficiencies[i - 1] is not None
                        and efficiencies[i] < efficiencies[i - 1])
            correlated = dir_dips and eff_dips
            flags.append({
                "gain_from": gain_grid[i - 1], "gain_to": gain_grid[i],
                "recovery_drop": recoveries[i - 1] - recoveries[i],
                "direction_also_dipped": dir_dips, "efficiency_also_dipped": eff_dips,
                "classification": "plausible_nonlinear_interaction" if correlated else "BUG_SIGNATURE",
            })
    return flags
