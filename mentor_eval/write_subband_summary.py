"""
mentor_eval/write_subband_summary.py — generate
outputs/mentor_review/subband_analysis/SUMMARY.md by inspecting what
actually exists on disk (never hardcodes "done"), same convention as
mentor_eval/write_summary.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mentor_eval.subband_features import SUBBAND_NAMES, SUBBAND_CLINICAL_LABEL, subband_frequency_ranges


def _section(title: str, lines: list[str]) -> str:
    return f"## {title}\n\n" + "\n".join(lines) + "\n\n"


def write_subband_summary(cfg) -> Path:
    base = Path(cfg.paths.outputs.results).parent / "mentor_review" / "subband_analysis"
    out_path = base / "SUMMARY.md"

    md = []
    md.append("# Mentor Review — Subband (MEES-style) Analysis Summary\n")
    md.append(
        "This is an additional set of figures/metrics on top of the main `mentor_eval/` "
        "pipeline (see `../SUMMARY.md`), following the multiscale-energy approach of "
        "Sharma, Tripathy & Dandapat (IEEE TBME 2015) — wavelet subband decomposition, "
        "within-class variation box plots, an annotated single-beat figure, and "
        "subband-level similarity metrics. Nothing already built was replaced; this "
        "folder is purely additive.\n"
    )

    fs = 100.0
    ranges = subband_frequency_ranges(fs)
    band_lines = [
        f"- **{b}**: {ranges[b][0]:.2f}-{ranges[b][1]:.2f} Hz — {SUBBAND_CLINICAL_LABEL[b]}"
        for b in SUBBAND_NAMES
    ]
    md.append(_section(
        "Wavelet configuration (read this first)",
        [
            "Library: PyWavelets (`pywt`). Wavelet: `bior4.4` (Daubechies/CDF 9/7 "
            "biorthogonal — matches the paper). Decomposition level: **J=3**, not the "
            "paper's J=6 — re-derived because PTB-XL here is sampled at 100 Hz vs. the "
            "paper's 1000 Hz; using J=6 at our 10x-lower rate would push real QRS energy "
            "out of the named subbands entirely. See `mentor_eval/subband_features.py` "
            "module docstring for the full derivation.",
            "",
        ] + band_lines,
    ))

    # ── Items 1+2 ────────────────────────────────────────────────────────────
    table_path = base / "subband_energy_table.csv"
    lines = []
    if table_path.exists():
        df = pd.read_csv(table_path)
        lines.append(f"{len(df)} rows (class x subband x lead). Full table: `subband_energy_table.csv`.")
        lines.append("\nMean energy per class, averaged across leads:")
        pivot = df.groupby(["class", "subband"])["mean_energy"].mean().unstack()
        lines.append("```\n" + pivot.to_string() + "\n```")
    else:
        lines.append("_Not yet run._ `python -m mentor_eval.subband_decomposition`")

    box_pngs = sorted(base.glob("boxplot_*.png"))
    if box_pngs:
        lines.append(f"\nReal-vs-generated box plots: {len(box_pngs)} subbands — " +
                      ", ".join(p.name for p in box_pngs))
    else:
        lines.append(
            "\n_Box plots (item 2) blocked._ Needs `outputs/models/diffusion_best.pt` "
            "to generate samples for comparison."
        )
    md.append(_section("1+2. Subband energy table & within-class box plots (mirrors Fig. 4/7)", lines))

    # ── Item 3 ───────────────────────────────────────────────────────────────
    beat_pngs = sorted(base.glob("annotated_beat_*.png"))
    lines = []
    if beat_pngs:
        lines.append(f"{len(beat_pngs)} annotated beat figures: " + ", ".join(p.name for p in beat_pngs))
        lines.append(
            "Each reuses the QRS/ST/T boundaries already detected by "
            "`mentor_eval/zoomed_clinical.py` (no redetection) and draws dashed "
            "ellipse annotations directly on the trace, mirroring Sharma Fig. 2."
        )
    else:
        lines.append("_Not yet run._ `python -m mentor_eval.subband_annotated_beat`")
    md.append(_section("3. Annotated single-beat figure (mirrors Fig. 2)", lines))

    # ── Item 4 ───────────────────────────────────────────────────────────────
    sim_path = base / "subband_similarity_metrics.csv"
    lines = []
    if sim_path.exists():
        df = pd.read_csv(sim_path)
        lines.append("```\n" + df.to_string(index=False) + "\n```")
        readme_path = base / "subband_similarity_README.md"
        if readme_path.exists():
            content = readme_path.read_text()
            callout_line = [l for l in content.splitlines() if l.startswith("Largest real-vs-generated")]
            if callout_line:
                lines.append(f"\n**{callout_line[0]}**")
        lines.append("\nSee `subband_similarity_README.md` for how to read each number.")
    else:
        lines.append(
            "_Blocked._ Needs generated samples (checkpoint). "
            "Run `python -m mentor_eval.subband_similarity_metrics` on the GPU server."
        )
    md.append(_section("4. Subband-level similarity metrics (extends item 8)", lines))

    # ── Item 5 ───────────────────────────────────────────────────────────────
    cv_dir = base.parent / "classification_validation"
    real_plain = cv_dir / "confusion_matrix_real_plain.txt"
    gen_plain = cv_dir / "confusion_matrix_generated_plain.txt"
    lines = []
    if real_plain.exists():
        lines.append("Plain-table confusion matrix (real data) — same numbers as the "
                      "colored heatmap in `../classification_validation/confusion_matrix_real.png`, "
                      "presented as a plain numeric grid (Sharma Table IV/V style):")
        lines.append("```\n" + real_plain.read_text().strip() + "\n```")
    else:
        lines.append("_Not yet run._ Run `python -m mentor_eval.classification_validation` "
                      "(the plain table is written automatically alongside the colored heatmap).")
    if gen_plain.exists():
        lines.append("\nPlain-table confusion matrix (generated data):")
        lines.append("```\n" + gen_plain.read_text().strip() + "\n```")
    md.append(_section(
        "5. Confusion matrix — plain-table style (additive, alongside the existing colored heatmap)",
        lines,
    ))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md))
    return out_path


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils import load_config
    write_subband_summary(load_config())
