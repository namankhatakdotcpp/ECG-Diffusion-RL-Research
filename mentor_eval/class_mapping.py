"""
mentor_eval/class_mapping.py — PTB-XL SCP code -> 4-class mentor-review mapping.

PTB-XL has no native STEMI / NSTEMI labels (it only encodes MI by anatomical
site and ischemia by lead group, not by ST-segment elevation magnitude).
The mapping below is an explicit, documented PROXY, not a clinical ground
truth:

  Normal  <- NORM
  STEMI   <- acute injury / infarction codes (INJ*, and MI-by-site codes)
  NSTEMI  <- ischemia-without-infarction codes (ISC*)
  AFIB    <- AFIB, AFLT (rhythm codes, independent of the diagnostic_class
             system used elsewhere in this repo)

"Coronary Heart Disease" was dropped as a 5th class: PTB-XL has no
diagnostic category for it that doesn't already fully overlap with the
STEMI/NSTEMI proxy codes above (decided with the project owner).

NOTE: the existing trained diffusion model (step03/step04) uses a DIFFERENT
class scheme (NORM, MI, STTC, CD, HYP, OTHER) where AFIB was merged into
OTHER (only 103 AFIB records, below min_class_samples=200). The model has
no AFIB-conditioned generation. Anything in this package that needs
generated AFIB samples will flag that gap rather than substituting OTHER.

Usage:
    from mentor_eval.class_mapping import (
        MENTOR_CLASSES, load_ptbxl_database, assign_mentor_class,
        filter_to_mentor_classes, mapping_table_df,
    )
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Optional

import pandas as pd

MENTOR_CLASSES: list[str] = ["Normal", "STEMI", "NSTEMI", "AFIB"]

# ──────────────────────────────────────────────────────────────────────────────
# Explicit SCP code -> mentor class table (sanity-check this against
# data/ptbxl/scp_statements.csv before trusting downstream figures/metrics)
# ──────────────────────────────────────────────────────────────────────────────

SCP_TO_MENTOR_CLASS: dict[str, str] = {
    # Normal
    "NORM": "Normal",
    # STEMI proxy — acute subendocardial injury + MI-by-anatomical-site codes
    "INJAS": "STEMI", "INJAL": "STEMI", "INJIN": "STEMI",
    "INJLA": "STEMI", "INJIL": "STEMI",
    "AMI": "STEMI", "ASMI": "STEMI", "ALMI": "STEMI",
    "IMI": "STEMI", "ILMI": "STEMI", "IPLMI": "STEMI", "IPMI": "STEMI",
    "LMI": "STEMI", "PMI": "STEMI",
    # NSTEMI proxy — ischemic ST-T changes without infarction
    "ISCAL": "NSTEMI", "ISCAN": "NSTEMI", "ISCAS": "NSTEMI",
    "ISCIN": "NSTEMI", "ISCIL": "NSTEMI", "ISCLA": "NSTEMI", "ISC_": "NSTEMI",
    # Atrial fibrillation / flutter (rhythm codes)
    "AFIB": "AFIB", "AFLT": "AFIB",
}

# Reason string per code, for the printed/CSV sanity-check table.
_REASON: dict[str, str] = {
    "NORM": "direct match (normal ECG)",
    **{c: "acute injury pattern (proxy for STEMI, not a true ST-elevation measurement)"
       for c in ("INJAS", "INJAL", "INJIN", "INJLA", "INJIL")},
    **{c: "MI by anatomical site (proxy for STEMI)"
       for c in ("AMI", "ASMI", "ALMI", "IMI", "ILMI", "IPLMI", "IPMI", "LMI", "PMI")},
    **{c: "ischemic ST-T change without infarction (proxy for NSTEMI)"
       for c in ("ISCAL", "ISCAN", "ISCAS", "ISCIN", "ISCIL", "ISCLA", "ISC_")},
    "AFIB": "direct match (atrial fibrillation, rhythm code)",
    "AFLT": "direct match (atrial flutter, rhythm code)",
}


def mapping_table_df() -> pd.DataFrame:
    """Return the SCP->mentor-class mapping as a DataFrame for review/export."""
    rows = [
        {"scp_code": code, "mentor_class": cls, "reason": _REASON[code]}
        for code, cls in sorted(SCP_TO_MENTOR_CLASS.items())
    ]
    return pd.DataFrame(rows)


def parse_scp_codes(raw) -> dict[str, float]:
    """Parse the scp_codes column (stringified dict) into {code: confidence}."""
    try:
        return ast.literal_eval(str(raw))
    except (ValueError, SyntaxError):
        try:
            return json.loads(str(raw).replace("'", '"'))
        except Exception:
            return {}


def assign_mentor_class(scp_codes: dict[str, float]) -> Optional[str]:
    """Pick the highest-confidence SCP code that maps to one of MENTOR_CLASSES.

    Returns None if no code in this record maps to any of the 4 classes —
    such records are excluded from this analysis pipeline (per spec item 2).
    """
    best_cls, best_conf = None, -1.0
    for code, conf in scp_codes.items():
        cls = SCP_TO_MENTOR_CLASS.get(str(code).upper())
        if cls is not None and conf > best_conf:
            best_cls, best_conf = cls, conf
    return best_cls


def load_ptbxl_database(ptbxl_dir: Path) -> pd.DataFrame:
    """Load ptbxl_database.csv indexed by ecg_id."""
    db_path = Path(ptbxl_dir) / "ptbxl_database.csv"
    if not db_path.exists():
        raise FileNotFoundError(f"ptbxl_database.csv not found at {db_path}")
    return pd.read_csv(str(db_path), index_col="ecg_id")


# ──────────────────────────────────────────────────────────────────────────────
# Bridge to the TRAINED diffusion model's class scheme (NORM, MI, STTC, CD,
# HYP, OTHER — see step03_eda_and_class_mapping.py). This is a SEPARATE
# mapping from SCP_TO_MENTOR_CLASS above: the model was never trained to
# condition on "STEMI"/"NSTEMI"/"AFIB" directly. Used only when generating
# samples (item 4/6/8/9) — to ask the model for the closest class it knows.
# ──────────────────────────────────────────────────────────────────────────────

MENTOR_TO_TRAINED_CLASS: dict[str, Optional[str]] = {
    "Normal": "NORM",
    # STEMI proxy codes (INJ*, MI-by-site) fall under the trained model's "MI" superclass.
    "STEMI": "MI",
    # NSTEMI proxy codes (ISC*) fall under the trained model's "STTC" superclass.
    "NSTEMI": "STTC",
    # AFIB has no dedicated trained class — it was merged into OTHER in step03
    # (only 103 records, below min_class_samples=200). The model cannot
    # generate AFIB specifically; conditioning on OTHER would also pull in
    # every other low-frequency code lumped into OTHER. Flag, don't substitute.
    "AFIB": None,
}


def filter_to_mentor_classes(ptbxl_db: pd.DataFrame) -> pd.DataFrame:
    """Add a 'mentor_class' column and drop rows that don't map to any of the
    4 mentor-review classes. Returns a new DataFrame (does not mutate input).
    """
    classes = []
    for raw in ptbxl_db["scp_codes"]:
        scp = parse_scp_codes(raw)
        classes.append(assign_mentor_class(scp))
    out = ptbxl_db.copy()
    out["mentor_class"] = classes
    return out[out["mentor_class"].notna()].copy()
