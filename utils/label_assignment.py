"""
utils/label_assignment.py — shared SCP-code -> class selection logic.

Used by both step03_eda_and_class_mapping.py (_assign_primary, for EDA and
class-taxonomy determination) and step04_transformer_diffusion.py
(_load_class_labels, for training labels), so the two stay in lockstep by
construction rather than by two independently-maintained copies of the
same selection rule.

Background: Roadmap/Stage_0_Pipeline_Audit/Reports/Pipeline_Code_Audit.md
Findings 4 and 5. Finding 4 verified (by actually diffing all 21,799
records) that step03 and step04's PRE-FIX implementations agreed with
each other on every record under the current data — but Finding 5 also
confirmed the shared selection rule they both used was undocumented and
order-dependent: ties in SCP-code confidence were broken by whichever
code happened to appear first in that record's scp_codes dict (an
artifact of source-CSV field order, not a clinical decision), affecting
3,185 / 21,799 records (14.6%) with multiple diagnostic codes.

Tie-break rule (fix for Finding 5): ties at the maximum confidence value
are now broken by TIE_BREAK_PRIORITY below — a record with tied top-
confidence evidence for a more specific/actionable diagnosis is not
down-labeled toward a less specific one. This is a clinical-severity
ordering, not an arbitrary one (e.g. alphabetical), because this is a
medical dataset and an arbitrary tie-break is worse than a principled one
even when both are equally deterministic.
"""

from __future__ import annotations

from typing import Optional

# Highest priority first. A class not in this list (e.g. a custom/extra
# superclass some future config might introduce) ranks last, tied with OTHER.
TIE_BREAK_PRIORITY: list[str] = ["MI", "STTC", "CD", "HYP", "NORM", "OTHER"]


def _priority_rank(cls: str) -> int:
    """Lower = higher priority. Unknown classes rank last (same as OTHER)."""
    try:
        return TIE_BREAK_PRIORITY.index(cls)
    except ValueError:
        return len(TIE_BREAK_PRIORITY)


def assign_primary_class(scp_dict: dict[str, float], code_map: dict[str, str]) -> Optional[str]:
    """
    Pick the class for one record given its {SCP_code: confidence} dict
    and a code -> class map (either step03's raw scp_statements.csv-derived
    map, or step04's already-collapsed class_mapping.json — both are valid
    inputs, this function is agnostic to which stage of collapsing the map
    represents).

    Deterministic: highest confidence wins. Ties at the maximum confidence
    are broken by TIE_BREAK_PRIORITY, not by dict-iteration order.

    Returns None if no code in scp_dict maps to a known class in code_map.
    Callers decide their own OTHER-vs-drop policy for that case (this is
    intentionally NOT decided here — see Pipeline_Code_Audit.md Finding 7,
    which documents that step03 and step04 currently differ on exactly
    this point, a separate, lower-severity, deliberately-deferred issue).
    """
    candidates: list[tuple[float, str]] = []
    for code, conf in scp_dict.items():
        cls = code_map.get(str(code).upper())
        if cls:
            candidates.append((conf, cls))

    if not candidates:
        return None

    max_conf = max(conf for conf, _ in candidates)
    tied = [cls for conf, cls in candidates if conf == max_conf]
    tied.sort(key=_priority_rank)
    return tied[0]
