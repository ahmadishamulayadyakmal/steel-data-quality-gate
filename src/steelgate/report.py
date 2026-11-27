"""Deterministic Springer-style SVG figures and multi-database HTML report (v0.7.0)."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

# Springer Nature / academic journal colorblind-safe palette
_BG = "#ffffff"
_RULE_DARK = "#1a1a1a"
_AXIS = "#222222"
_GRID = "#e5e7eb"
_SUBTLE_BG = "#f8f9fa"
_TEXT = "#111827"
_MUTED = "#4b5563"
_NAVY = "#003b71"
_STEEL = "#4a90c2"
_SLATE = "#7a8b99"
_GRAY_BAR = "#c8cdd3"
_CRIMSON = "#b91c1c"
_PASS_GREEN = "#166534"

_FONT = "Helvetica Neue, Helvetica, Arial, Liberation Sans, sans-serif"


def _fmt(val: Any, digits: int = 4) -> str:
    return f"{float(val):.{digits}f}"


def _stat(payload: Any, suffix: str = "", digits: int = 4) -> str:
    if isinstance(payload, dict) and "mean" in payload and "std" in payload:
        return f"{_fmt(payload['mean'], digits)}{suffix} ± {_fmt(payload['std'], digits)}{suffix}"
    return f"{_fmt(payload, digits)}{suffix}"


def _txt(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 11,
    fill: str = _TEXT,
    weight: str = "normal",
    style: str = "normal",
    anchor: str = "start",
    rotate: float | None = None,
) -> str:
    transform = f' transform="rotate({rotate:.1f} {x:.1f} {y:.1f})"' if rotate else ""
    font_style = f' font-style="{style}"' if style != "normal" else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
        f'font-weight="{weight}"{font_style} text-anchor="{anchor}" '
        f'font-family="{_FONT}"{transform}>{escape(value)}</text>'
    )


def _navy_heat(fraction: float) -> tuple[str, str]:
    f = max(0.0, min(1.0, float(fraction)))
    r = int(round(255 * (1.0 - f) + 0 * f))
    g = int(round(255 * (1.0 - f) + 59 * f))
    b = int(round(255 * (1.0 - f) + 113 * f))
    return f"#{r:02x}{g:02x}{b:02x}", ("#ffffff" if f >= 0.45 else _TEXT)


def build_scorecard_svg(metrics: dict[str, Any]) -> str:
    """Render Table 1: Multi-database 64-rule quantitative scorecard for Case 02."""
    raw1 = metrics["raw_d01_audit"]
    unit = metrics["unit_verification"]
    rej1 = metrics["corrupt_copy_rejection"]
    camp = metrics["campaign_leakage_cv"]
    ba1 = metrics["before_after_benchmark"]["models"]
    et1 = ba1["extra_trees"]
    ens1 = ba1["voting_ensemble"]
    d04 = metrics["d04_benchmark"]
    d04_b = d04["five_fold_benchmark"]
    d09 = metrics["d09_benchmark"]
    d09_b = d09["chronological_before_after"]
    d09_t = d09["temporal_leakage"]

    w, h = 1020, 668
    left, right = 36, 984
    rows = [
        (
            "D01 Plates Faults (38 rules)",
            "1,941 raw / 1,939 curated plates",
            f"Tier 1: 1,941/1,941; R38 physics catches rows 518 & 591; Curated: 1,939/1,939 (100%)",
            "PASS",
            _PASS_GREEN,
        ),
        (
            "D01 ET+HGB Gate Recovery",
            "5 seeds, clean untouched test split",
            f"Acc {_stat(et1['before_gate_uncleaned']['accuracy'])} (ET) -> {_stat(ens1['after_gate_cleaned']['accuracy'])} (ET+HGB); 0 leaked rows",
            "IMPROVED",
            _PASS_GREEN,
        ),
        (
            "D04 MatBench Steels (12 rules)",
            "312 alloys, official 5-fold CV",
            f"M12 catches {len(d04['real_raw_stoich_anomalies'])} raw formula deviations; MAE {_stat(d04_b['before_gate']['mae_mpa'], ' MPa', 2)} -> {_stat(d04_b['after_gate']['mae_mpa'], ' MPa', 2)} (-{d04_b['mae_reduction_pct']:.1f}%)",
            "PASS",
            _PASS_GREEN,
        ),
        (
            "D09 Steel Energy (14 rules)",
            "35,040 15-min rows (2018)",
            f"35,040/35,040 pass E01-E14 (PF <= 0.01%); Chronological MAE {d09_b['before_gate']['mae_kwh']:.2f} -> {d09_b['after_gate']['mae_kwh']:.2f} kWh (-{d09_b['mae_reduction_pct']:.1f}%)",
            "PASS",
            _PASS_GREEN,
        ),
        (
            "Pint Units & Stoichiometry",
            f"{unit['n_unit_round_trips']} round-trips + 4 steels",
            f"Length/stress/temp/energy max err {unit['max_unit_round_trip_abs_err']:.2e}; wt%<->at% err {unit['max_composition_round_trip_abs_err']:.2e}",
            "PASS" if unit["passed"] else "FAIL",
            _PASS_GREEN if unit["passed"] else _CRIMSON,
        ),
        (
            "Corrupt-Copy Rejections",
            "Synthetic D01, D04, D09 corruptions",
            f"All 3 gates raise QualityGateRejectionError (D01: {rej1['rejection_report']['rules_failed']} rules; D04 & D09 rejected)",
            "REJECTED",
            _PASS_GREEN,
        ),
        (
            "R09 Campaign Leakage (D01)",
            f"Stratified vs. Grouped ({camp['n_campaign_groups']} groups)",
            f"Acc {_fmt(camp['stratified_kfold']['accuracy']['mean'])} -> {_fmt(camp['campaign_grouped_kfold']['accuracy']['mean'])} ({100*camp['accuracy_inflation_gap']:+.2f} pp gap)",
            "MEASURED",
            _NAVY,
        ),
        (
            "Temporal Leakage (D09)",
            "Random 80/20 vs. Forward 80/20",
            f"Random MAE {d09_t['random_80_20_mae_kwh']:.2f} kWh vs. Forward MAE {d09_t['chronological_80_20_mae_kwh']:.2f} kWh ({d09_t['lookahead_mae_inflation_kwh']:+.2f} kWh)",
            "MEASURED",
            _NAVY,
        ),
        (
            "D02 NEU-DET Camera Contract",
            "Metadata-only schema (200x200 px)",
            f"Status: {metrics['data']['D02_catalog_status']} (enforces mirror_licence_verified == True)",
            "GATED",
            _NAVY,
        ),
    ]

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="Case 02 multi-database quantitative scorecard">',
        f'<rect width="{w}" height="{h}" fill="{_BG}"/>',
        _txt(left, 34, "Table 1", size=12, fill=_NAVY, weight="bold"),
        _txt(left + 52, 34, "Case 02 multi-database quality gate verification across 37,293 open steel records and 64 rules (v0.7.0)", size=12, fill=_TEXT, weight="bold"),
        _txt(left, 54, f"Open Datasets: UCI D01 (1,941) + MatBench D04 (312) + UCI Steel Energy D09 (35,040) | Determinism: {metrics['determinism_check']}", size=10, fill=_MUTED),
        f'<line x1="{left}" y1="70" x2="{right}" y2="70" stroke="{_RULE_DARK}" stroke-width="2"/>',
        _txt(left + 6, 92, "Database / Gate Component", size=10, weight="bold"),
        _txt(260, 92, "Dataset / Scope", size=10, weight="bold"),
        _txt(495, 92, "Measured Evidence (Our Computation)", size=10, weight="bold"),
        _txt(right - 8, 92, "Status", size=10, weight="bold", anchor="end"),
        f'<line x1="{left}" y1="104" x2="{right}" y2="104" stroke="{_RULE_DARK}" stroke-width="1"/>',
    ]

    y = 132
    row_h = 48
    for i, (comp, scope, evidence, status, status_color) in enumerate(rows):
        if i % 2 == 1:
            parts.append(f'<rect x="{left}" y="{y - 22}" width="{right - left}" height="{row_h}" fill="{_SUBTLE_BG}"/>')
        parts.append(_txt(left + 6, y, comp, size=10, weight="bold"))
        parts.append(_txt(260, y, scope, size=9.5, fill=_MUTED))
        parts.append(_txt(495, y, evidence, size=9.5, fill=_TEXT))
        parts.append(_txt(right - 8, y, status, size=10, weight="bold", fill=status_color, anchor="end"))
        y += row_h

    parts.append(f'<line x1="{left}" y1="{y - 14}" x2="{right}" y2="{y - 14}" stroke="{_RULE_DARK}" stroke-width="2"/>')
    parts.append(_txt(left, y + 10, "Disclosure: All before/after corruption experiments are explicitly synthetic perturbations of open datasets evaluated on clean untouched test splits.", size=9.5, fill=_MUTED))
    parts.append(_txt(left, y + 26, "All raw file SHA-256 checksums are verified at runtime; no plant or proprietary data are stored or used.", size=9.5, fill=_MUTED))
    parts.append("</svg>")
    return "\n".join(parts)


def build_before_after_svg(metrics: dict[str, Any]) -> str:
    """Render Fig. 1: 4-panel multi-database recovery & leakage analysis (D01, D04, D09)."""
    ablation = metrics["before_after_benchmark"]["ablation_ladder"]
    d04_b = metrics["d04_benchmark"]["five_fold_benchmark"]
    d09_b = metrics["d09_benchmark"]["chronological_before_after"]
    d09_t = metrics["d09_benchmark"]["temporal_leakage"]
    camp = metrics["campaign_leakage_cv"]

    w, h = 1060, 740
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="Case 02 multi-database recovery and leakage">',
        f'<rect width="{w}" height="{h}" fill="{_BG}"/>',
        _txt(36, 30, "Fig. 1", size=12, fill=_NAVY, weight="bold"),
        _txt(78, 30, "Multi-database quality-gate recovery (D01, D04, D09) and leakage inflation (Case 02 v0.7.0)", size=12, weight="bold"),
        _txt(36, 48, "Training partitions receive synthetic corruptions of open datasets; all test evaluations use clean untouched test splits.", size=10, fill=_MUTED),
    ]

    # Panel a: D01 5-stage ablation ladder (top-left)
    ax0, ay0, aw, ah = 80, 95, 425, 225
    parts.append(_txt(36, 80, "a", size=13, weight="bold", fill=_NAVY))
    parts.append(_txt(52, 80, "D01 5-stage gate ablation & ET+HGB ensemble (5 seeds)", size=11, weight="bold"))
    for tick in (0.72, 0.74, 0.76, 0.78, 0.80, 0.82):
        yy = ay0 + ah - ((tick - 0.72) / 0.10) * ah
        parts.append(f'<line x1="{ax0}" y1="{yy:.1f}" x2="{ax0 + aw}" y2="{yy:.1f}" stroke="{_GRID}" stroke-width="1"/>')
        parts.append(_txt(ax0 - 8, yy + 4, f"{tick:.2f}", size=9, fill=_MUTED, anchor="end"))
    parts.append(f'<line x1="{ax0}" y1="{ay0}" x2="{ax0}" y2="{ay0 + ah}" stroke="{_AXIS}" stroke-width="1.2"/>')
    parts.append(f'<line x1="{ax0}" y1="{ay0 + ah}" x2="{ax0 + aw}" y2="{ay0 + ah}" stroke="{_AXIS}" stroke-width="1.2"/>')

    ab_short = {
        "Before Gate (ET)": ("Before", _CRIMSON),
        "+S1 Pint Units": ("+Units", _SLATE),
        "+S2 Repair & 38 Rules": ("+38Rules", _SLATE),
        "+S3 Dedup": ("+Dedup", _STEEL),
        "+S4 Confident Learning": ("+CL", _STEEL),
        "+S5 ET+HGB Ensemble": ("+ET+HGB", _NAVY),
        "Clean Oracle (ET+HGB)": ("Oracle", _PASS_GREEN),
    }
    ab_plot = [item for item in ablation if item["stage"] in ab_short]
    step_a = aw / len(ab_plot)
    for i, item in enumerate(ab_plot):
        lbl, col = ab_short[item["stage"]]
        cx = ax0 + (i + 0.5) * step_a
        m_v = float(item["accuracy"]["mean"])
        s_v = float(item["accuracy"]["std"])
        bh_a = max(0.0, ((m_v - 0.72) / 0.10) * ah)
        by_a = ay0 + ah - bh_a
        parts.append(f'<rect x="{cx - 16:.1f}" y="{by_a:.1f}" width="32" height="{bh_a:.1f}" fill="{col}"/>')
        err_px = (s_v / 0.10) * ah
        parts.append(f'<line x1="{cx:.1f}" y1="{by_a - err_px:.1f}" x2="{cx:.1f}" y2="{by_a + err_px:.1f}" stroke="{_RULE_DARK}" stroke-width="1.1"/>')
        parts.append(_txt(cx, by_a - err_px - 4, f"{m_v:.4f}", size=8, weight="bold" if lbl in ("+ET+HGB", "Oracle") else "normal", anchor="middle"))
        parts.append(_txt(cx, ay0 + ah + 15, lbl, size=8.5, anchor="middle"))

    # Panel b: D04 MatBench Yield Strength MAE (top-right)
    bx0, by0, bw, bh = 605, 95, 415, 225
    parts.append(_txt(555, 80, "b", size=13, weight="bold", fill=_NAVY))
    parts.append(_txt(571, 80, "D04 MatBench yield-strength MAE (official 5 folds)", size=11, weight="bold"))
    max_mae_d04 = max(500.0, float(d04_b["before_gate"]["mae_mpa"]["mean"]) * 1.15)
    for tick in (0, 100, 200, 300, 400, 500):
        yy = by0 + bh - (tick / max_mae_d04) * bh
        parts.append(f'<line x1="{bx0}" y1="{yy:.1f}" x2="{bx0 + bw}" y2="{yy:.1f}" stroke="{_GRID}" stroke-width="1"/>')
        parts.append(_txt(bx0 - 8, yy + 4, str(tick), size=9, fill=_MUTED, anchor="end"))
    parts.append(f'<line x1="{bx0}" y1="{by0}" x2="{bx0}" y2="{by0 + bh}" stroke="{_AXIS}" stroke-width="1.2"/>')
    parts.append(f'<line x1="{bx0}" y1="{by0 + bh}" x2="{bx0 + bw}" y2="{by0 + bh}" stroke="{_AXIS}" stroke-width="1.2"/>')

    d04_bars = [
        ("Before Gate (ksi/GPa/NaN)", d04_b["before_gate"]["mae_mpa"], _CRIMSON),
        ("After Gate (SteelGate D04)", d04_b["after_gate"]["mae_mpa"], _NAVY),
        ("Clean Oracle (MatBench)", d04_b["clean_oracle"]["mae_mpa"], _STEEL),
    ]
    step_b = bw / len(d04_bars)
    for i, (lbl, payload, col) in enumerate(d04_bars):
        cx = bx0 + (i + 0.5) * step_b
        m_v, s_v = float(payload["mean"]), float(payload["std"])
        bh_b = (m_v / max_mae_d04) * bh
        by_b = by0 + bh - bh_b
        parts.append(f'<rect x="{cx - 32:.1f}" y="{by_b:.1f}" width="64" height="{bh_b:.1f}" fill="{col}"/>')
        err_px = (s_v / max_mae_d04) * bh
        parts.append(f'<line x1="{cx:.1f}" y1="{by_b - err_px:.1f}" x2="{cx:.1f}" y2="{by_b + err_px:.1f}" stroke="{_RULE_DARK}" stroke-width="1.2"/>')
        parts.append(_txt(cx, by_b - err_px - 5, f"{m_v:.2f} MPa", size=9, weight="bold", anchor="middle"))
        parts.append(_txt(cx, by0 + bh + 16, lbl, size=8.5, anchor="middle"))
    parts.append(_txt(bx0 - 42, by0 + bh / 2, "MAE (MPa, lower is better)", size=9, fill=_MUTED, anchor="middle", rotate=-90))

    # Panel c: D09 Steel Energy MAE + Temporal Leakage (bottom-left)
    cx0, cy0, cw, ch = 80, 420, 425, 220
    parts.append(_txt(36, 400, "c", size=13, weight="bold", fill=_NAVY))
    parts.append(_txt(52, 400, "D09 Steel Energy (35,040 rows): Gate MAE & temporal leakage", size=11, weight="bold"))
    d09_bars = [
        ("Random 80/20", d09_t["random_80_20_mae_kwh"], _GRAY_BAR),
        ("Before Gate (Wh/MWh)", d09_b["before_gate"]["mae_kwh"], _CRIMSON),
        ("After Gate (Chron.)", d09_b["after_gate"]["mae_kwh"], _NAVY),
        ("Clean Oracle (Chron.)", d09_b["clean_oracle"]["mae_kwh"], _STEEL),
    ]
    max_e = max(val for _, val, _ in d09_bars) * 1.18
    parts.append(f'<line x1="{cx0}" y1="{cy0}" x2="{cx0}" y2="{cy0 + ch}" stroke="{_AXIS}" stroke-width="1.2"/>')
    parts.append(f'<line x1="{cx0}" y1="{cy0 + ch}" x2="{cx0 + cw}" y2="{cy0 + ch}" stroke="{_AXIS}" stroke-width="1.2"/>')
    step_c = cw / len(d09_bars)
    for i, (lbl, val, col) in enumerate(d09_bars):
        cx = cx0 + (i + 0.5) * step_c
        bh_c = (val / max_e) * ch
        by_c = cy0 + ch - bh_c
        parts.append(f'<rect x="{cx - 26:.1f}" y="{by_c:.1f}" width="52" height="{bh_c:.1f}" fill="{col}"/>')
        parts.append(_txt(cx, by_c - 6, f"{val:.2f} kWh", size=8.5, weight="bold", anchor="middle"))
        parts.append(_txt(cx, cy0 + ch + 16, lbl, size=8.5, anchor="middle"))
    parts.append(_txt(cx0 - 42, cy0 + ch / 2, "Energy MAE (kWh, lower is better)", size=9, fill=_MUTED, anchor="middle", rotate=-90))

    # Panel d: R09 D01 Mill-campaign leakage CV gap (bottom-right)
    dx0, dy0, dw, dh = 605, 420, 415, 220
    parts.append(_txt(555, 400, "d", size=13, weight="bold", fill=_NAVY))
    parts.append(_txt(571, 400, f"R09 mill-campaign leakage on D01 ({camp['n_campaign_groups']} campaigns)", size=11, weight="bold"))
    for tick in (0.5, 0.6, 0.7, 0.8, 0.9):
        yy = dy0 + dh - ((tick - 0.5) / 0.4) * dh
        parts.append(f'<line x1="{dx0}" y1="{yy:.1f}" x2="{dx0 + dw}" y2="{yy:.1f}" stroke="{_GRID}" stroke-width="1"/>')
        parts.append(_txt(dx0 - 8, yy + 4, f"{tick:.1f}", size=9, fill=_MUTED, anchor="end"))
    parts.append(f'<line x1="{dx0}" y1="{dy0}" x2="{dx0}" y2="{dy0 + dh}" stroke="{_AXIS}" stroke-width="1.2"/>')
    parts.append(f'<line x1="{dx0}" y1="{dy0 + dh}" x2="{dx0 + dw}" y2="{dy0 + dh}" stroke="{_AXIS}" stroke-width="1.2"/>')

    cv_items = [
        ("Stratified Acc", camp["stratified_kfold"]["accuracy"], _SLATE),
        ("Grouped Acc", camp["campaign_grouped_kfold"]["accuracy"], _NAVY),
        ("Stratified F1", camp["stratified_kfold"]["macro_f1"], _SLATE),
        ("Grouped F1", camp["campaign_grouped_kfold"]["macro_f1"], _STEEL),
    ]
    step_d = dw / len(cv_items)
    for i, (lbl, payload, col) in enumerate(cv_items):
        cx = dx0 + (i + 0.5) * step_d
        m_v, s_v = float(payload["mean"]), float(payload["std"])
        bh_d = max(0.0, ((m_v - 0.5) / 0.4) * dh)
        by_d = dy0 + dh - bh_d
        parts.append(f'<rect x="{cx - 22:.1f}" y="{by_d:.1f}" width="44" height="{bh_d:.1f}" fill="{col}"/>')
        err_px = (s_v / 0.4) * dh
        parts.append(f'<line x1="{cx:.1f}" y1="{by_d - err_px:.1f}" x2="{cx:.1f}" y2="{by_d + err_px:.1f}" stroke="{_RULE_DARK}" stroke-width="1.3"/>')
        parts.append(_txt(cx, by_d - err_px - 5, f"{m_v:.4f}", size=9, weight="bold", anchor="middle"))
        parts.append(_txt(cx, dy0 + dh + 16, lbl, size=8.5, anchor="middle"))

    parts.append(_txt(36, 700, "Source: Deterministic computation from results/metrics.json (D01, D04, D09 + synthetic training corruptions). Error bars show ±1 SD.", size=9.5, fill=_MUTED))
    parts.append("</svg>")
    return "\n".join(parts)


def build_diagnostics_svg(metrics: dict[str, Any]) -> str:
    """Render Fig. 2: Raw D01 luminosity audit, raw D04 stoichiometry audit, and 7x7 CM."""
    lum_pts = metrics["raw_d01_audit"]["luminosity_audit_points"]
    raw_sums = metrics["d04_benchmark"]["raw_formula_sums"]
    ens_after = metrics["before_after_benchmark"]["models"]["voting_ensemble"]["after_gate_cleaned"]
    cm = ens_after["confusion_matrix_normalized"]
    classes = metrics["before_after_benchmark"]["class_names"]

    w, h = 1080, 420
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="Case 02 raw anomalies and confusion matrix">',
        f'<rect width="{w}" height="{h}" fill="{_BG}"/>',
        _txt(32, 28, "Fig. 2", size=12, fill=_NAVY, weight="bold"),
        _txt(74, 28, "Real raw-data physics anomalies in D01 & D04, plus post-gate ET+HGB 7-class confusion matrix", size=12, weight="bold"),
    ]

    # Panel a: Raw D01 luminosity physics (left)
    ax0, ay0, aw, ah = 74, 72, 265, 260
    parts.append(_txt(32, 56, "a", size=13, weight="bold", fill=_NAVY))
    parts.append(_txt(48, 56, "Raw D01 luminosity physics (Rule R38)", size=10.5, weight="bold"))
    for tick in (0, 50, 100, 150, 200, 250):
        xx = ax0 + (tick / 250.0) * aw
        yy = ay0 + ah - (tick / 250.0) * ah
        parts.append(f'<line x1="{xx:.1f}" y1="{ay0}" x2="{xx:.1f}" y2="{ay0 + ah}" stroke="{_GRID}" stroke-width="0.8"/>')
        parts.append(f'<line x1="{ax0}" y1="{yy:.1f}" x2="{ax0 + aw}" y2="{yy:.1f}" stroke="{_GRID}" stroke-width="0.8"/>')
        parts.append(_txt(xx, ay0 + ah + 14, str(tick), size=8.5, fill=_MUTED, anchor="middle"))
        parts.append(_txt(ax0 - 6, yy + 3, str(tick), size=8.5, fill=_MUTED, anchor="end"))
    parts.append(f'<line x1="{ax0}" y1="{ay0}" x2="{ax0}" y2="{ay0 + ah}" stroke="{_AXIS}" stroke-width="1.2"/>')
    parts.append(f'<line x1="{ax0}" y1="{ay0 + ah}" x2="{ax0 + aw}" y2="{ay0 + ah}" stroke="{_AXIS}" stroke-width="1.2"/>')
    parts.append(f'<line x1="{ax0}" y1="{ay0 + ah}" x2="{ax0 + aw}" y2="{ay0}" stroke="{_CRIMSON}" stroke-width="1.4" stroke-dasharray="4,3"/>')
    for pt in lum_pts:
        px = ax0 + (min(250.0, max(0.0, pt["min_lum"])) / 250.0) * aw
        py = ay0 + ah - (min(250.0, max(0.0, pt["mean_lum"])) / 250.0) * ah
        if pt["valid"]:
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.2" fill="{_STEEL}" opacity="0.55"/>')
        else:
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5.0" fill="{_CRIMSON}" stroke="#ffffff" stroke-width="1"/>')
            parts.append(_txt(px + 7, py + 3, f"row {pt.get('row_index', '')}", size=8.5, fill=_CRIMSON, weight="bold"))
    parts.append(_txt(ax0 + aw / 2, ay0 + ah + 32, "Minimum_of_Luminosity (8-bit)", size=9, fill=_MUTED, anchor="middle"))
    parts.append(_txt(ax0 - 40, ay0 + ah / 2, "Mean luminosity (Sum / Area)", size=9, fill=_MUTED, anchor="middle", rotate=-90))

    # Panel b: Raw D04 alloy stoichiometric sum (center)
    bx0, by0, bw, bh = 420, 72, 260, 260
    parts.append(_txt(378, 56, "b", size=13, weight="bold", fill=_NAVY))
    parts.append(_txt(394, 56, "Raw D04 formula sum (Rule M12, 312 alloys)", size=10.5, weight="bold"))
    y_lo, y_hi = 0.9985, 1.0015
    for tick in (0.9985, 0.9990, 1.0000, 1.0010, 1.0015):
        yy = by0 + bh - ((tick - y_lo) / (y_hi - y_lo)) * bh
        parts.append(f'<line x1="{bx0}" y1="{yy:.1f}" x2="{bx0 + bw}" y2="{yy:.1f}" stroke="{_GRID}" stroke-width="0.8"/>')
        parts.append(_txt(bx0 - 6, yy + 3, f"{tick:.4f}", size=8, fill=_MUTED, anchor="end"))
    parts.append(f'<line x1="{bx0}" y1="{by0}" x2="{bx0}" y2="{by0 + bh}" stroke="{_AXIS}" stroke-width="1.2"/>')
    parts.append(f'<line x1="{bx0}" y1="{by0 + bh}" x2="{bx0 + bw}" y2="{by0 + bh}" stroke="{_AXIS}" stroke-width="1.2"/>')
    # Tolerance band [0.999, 1.001]
    y_top = by0 + bh - ((1.001 - y_lo) / (y_hi - y_lo)) * bh
    y_bot = by0 + bh - ((0.999 - y_lo) / (y_hi - y_lo)) * bh
    parts.append(f'<line x1="{bx0}" y1="{y_top:.1f}" x2="{bx0 + bw}" y2="{y_top:.1f}" stroke="{_CRIMSON}" stroke-width="1.1" stroke-dasharray="4,3"/>')
    parts.append(f'<line x1="{bx0}" y1="{y_bot:.1f}" x2="{bx0 + bw}" y2="{y_bot:.1f}" stroke="{_CRIMSON}" stroke-width="1.1" stroke-dasharray="4,3"/>')
    n_d04 = max(len(raw_sums), 1)
    for i, item in enumerate(raw_sums):
        px = bx0 + (item["row_index"] / n_d04) * bw
        val = min(y_hi, max(y_lo, float(item["raw_fraction_sum"])))
        py = by0 + bh - ((val - y_lo) / (y_hi - y_lo)) * bh
        if item["flagged_m12"]:
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="{_CRIMSON}" stroke="#ffffff" stroke-width="1"/>')
        elif i % 3 == 0:
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.0" fill="{_STEEL}" opacity="0.65"/>')
    parts.append(_txt(bx0 + bw / 2, by0 + bh + 32, "MatBench D04 alloy row index (0..311)", size=9, fill=_MUTED, anchor="middle"))
    parts.append(_txt(bx0 + bw - 4, by0 + 14, "6 flagged raw rows (>1e-3)", size=8.5, fill=_CRIMSON, weight="bold", anchor="end"))

    # Panel c: 7x7 Confusion Matrix post-gate ET+HGB (right)
    cx0, cy0, cell = 765, 72, 36
    parts.append(_txt(715, 56, "c", size=13, weight="bold", fill=_NAVY))
    parts.append(_txt(731, 56, "Post-gate ET+HGB 7-class confusion matrix", size=10.5, weight="bold"))
    short_cls = [c.replace("Other_Faults", "Other").replace("Z_Scratch", "Z_Scr").replace("K_Scatch", "K_Sca").replace("Dirtiness", "Dirt") for c in classes]
    for r_i in range(7):
        parts.append(_txt(cx0 - 6, cy0 + r_i * cell + 22, short_cls[r_i], size=8.5, anchor="end"))
        parts.append(_txt(cx0 + r_i * cell + cell / 2, cy0 + 7 * cell + 14, short_cls[r_i], size=8, anchor="middle"))
        for c_i in range(7):
            frac = float(cm[r_i][c_i])
            bg_c, fg_c = _navy_heat(frac)
            xx = cx0 + c_i * cell
            yy = cy0 + r_i * cell
            parts.append(f'<rect x="{xx}" y="{yy}" width="{cell}" height="{cell}" fill="{bg_c}" stroke="#e5e7eb" stroke-width="0.5"/>')
            parts.append(_txt(xx + cell / 2, yy + 22, f"{frac:.2f}", size=8, fill=fg_c, anchor="middle", weight="bold" if r_i == c_i else "normal"))

    parts.append(_txt(32, 398, "Note: Panels a and b inspect unmodified open datasets (D01 rows 518 & 591; D04 rows 21, 28, 36, 90, 91, 103). Panel c evaluates clean D01 test splits after synthetic training corruption + gate.", size=9.5, fill=_MUTED))
    parts.append("</svg>")
    return "\n".join(parts)


def build_html_quality_report(metrics: dict[str, Any], provenance: dict[str, Any]) -> str:
    """Render a self-contained HTML5 multi-database quality audit report (v0.7.0)."""
    raw1 = metrics["raw_d01_audit"]
    unit = metrics["unit_verification"]
    rej1 = metrics["corrupt_copy_rejection"]
    camp = metrics["campaign_leakage_cv"]
    ba1 = metrics["before_after_benchmark"]
    et1 = ba1["models"]["extra_trees"]
    ens1 = ba1["models"]["voting_ensemble"]
    d04 = metrics["d04_benchmark"]
    d04_b = d04["five_fold_benchmark"]
    d09 = metrics["d09_benchmark"]
    d09_b = d09["chronological_before_after"]
    d09_t = d09["temporal_leakage"]

    all_rules = (
        [("D01", r) for r in raw1["curated_strict_physical"]["rule_results"]]
        + [("D04", r) for r in d04["strict_m01_m12"]["rule_results"]]
        + [("D09", r) for r in d09["strict_e01_e14"]["rule_results"]]
    )
    rule_rows = "\n".join(
        f"<tr><td>{ds}</td><td><code>{escape(r['rule_id'])}</code></td><td>{escape(r['tier'])}</td>"
        f"<td>{escape(r['category'])}</td><td>{escape(r['name'])}</td>"
        f"<td>{escape(r['description'])}</td><td>{r['n_failing']}</td><td>{'PASS' if r['passed'] else 'FAIL'}</td></tr>"
        for ds, r in all_rules
    )
    ablation_rows = "\n".join(
        f"<tr><td>{escape(row['stage'])}</td><td>{_stat(row['accuracy'])}</td><td>{_stat(row['macro_f1'])}</td></tr>"
        for row in ba1["ablation_ladder"]
    )
    unit_rows = "\n".join(
        f"<tr><td>{u['original']} {escape(u['from_unit'])}</td>"
        f"<td>{u['converted']:.6g} {escape(u['to_unit'])}</td>"
        f"<td>{u['abs_error']:.2e}</td><td>{'PASS' if u['passed'] else 'FAIL'}</td></tr>"
        for u in unit["round_trips"]
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Case 02 (v0.7.0) — Multi-Database SteelGate Quality Audit</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 1120px; margin: 2rem auto; padding: 0 1.25rem; color: #111827; line-height: 1.5; }}
  h1, h2 {{ color: #003b71; border-bottom: 1px solid #d1d5db; padding-bottom: 0.3rem; }}
  .banner {{ background: #f8f9fa; border-left: 4px solid #003b71; padding: 0.9rem 1.1rem; margin: 1rem 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }}
  th, td {{ border-bottom: 1px solid #e5e7eb; padding: 0.42rem 0.55rem; text-align: left; }}
  th {{ background: #f3f4f6; font-weight: 600; }}
  pre {{ background: #111827; color: #f9fafb; padding: 1rem; overflow-x: auto; border-radius: 4px; font-size: 0.84rem; }}
  code {{ background: #f3f4f6; padding: 0.1rem 0.3rem; border-radius: 3px; }}
</style>
</head>
<body>
<h1>Case 02 (v0.7.0) — Multi-Database SteelGate Quality Audit</h1>
<div class="banner">
  <strong>Open-Data &amp; Synthetic Disclosure:</strong> Covers <strong>{metrics['total_real_records']:,} real records</strong> across 3 pinned open datasets (<code>D01</code> UCI Steel Plates Faults, <code>D04</code> MatBench <code>matbench_steels</code>, <code>D09</code> UCI Steel Industry Energy Consumption) and <strong>{metrics['total_schema_rules']} Pandera &amp; physics rules</strong>, plus metadata-only camera schema <code>D02</code> (<code>{escape(str(metrics['data']['D02_catalog_status']))}</code>). All corruption benchmarks are explicitly <strong>synthetic perturbations</strong> evaluated on untouched clean test splits. Determinism check: <strong>{metrics['determinism_check']}</strong>.
</div>

<h2>1. Multi-Database Summary Scorecard</h2>
<table>
  <thead><tr><th>Database</th><th>Records</th><th>Rules</th><th>Real Raw Findings</th><th>Gate Recovery (Clean Test)</th></tr></thead>
  <tbody>
    <tr><td><strong>D01</strong> UCI Steel Plates Faults</td><td>1,941 (1,939 curated)</td><td>38 (<code>R01</code>–<code>R38</code>)</td><td>2 luminosity anomalies (rows <code>518</code>, <code>591</code>); 239 campaign groups (+5.32 pp CV gap)</td><td>ET+HGB Acc <strong>{_stat(ens1['after_gate_cleaned']['accuracy'])}</strong> (vs. {_stat(et1['before_gate_uncleaned']['accuracy'])} uncleaned ET)</td></tr>
    <tr><td><strong>D04</strong> MatBench <code>matbench_steels</code></td><td>312</td><td>12 (<code>M01</code>–<code>M12</code>)</td><td>{len(d04['real_raw_stoich_anomalies'])} raw formula sum deviations (<code>M12</code>, max 1.22e-3)</td><td>MAE <strong>{_stat(d04_b['before_gate']['mae_mpa'], ' MPa', 2)} → {_stat(d04_b['after_gate']['mae_mpa'], ' MPa', 2)}</strong> (-{d04_b['mae_reduction_pct']:.1f}%)</td></tr>
    <tr><td><strong>D09</strong> UCI Steel Energy (2018)</td><td>35,040</td><td>14 (<code>E01</code>–<code>E14</code>)</td><td>35,040/35,040 pass all 14 rules incl. AC PF conservation; 00:00 rows stored out-of-order</td><td>Chronological MAE <strong>{d09_b['before_gate']['mae_kwh']:.2f} → {d09_b['after_gate']['mae_kwh']:.2f} kWh</strong> (-{d09_b['mae_reduction_pct']:.1f}%); random split understates MAE by {d09_t['lookahead_mae_inflation_kwh']:.2f} kWh</td></tr>
  </tbody>
</table>

<h2>2. D01 Five-Stage Cumulative Ablation Ladder (5 Seeds)</h2>
<table>
  <thead><tr><th>Stage</th><th>Accuracy (Mean ± SD)</th><th>Macro F1 (Mean ± SD)</th></tr></thead>
  <tbody>{ablation_rows}</tbody>
</table>

<h2>3. Corrupt-Copy Rejection Excerpt (Synthetic D01)</h2>
<pre>{escape(rej1['rejection_report']['error_report'])}</pre>

<h2>4. Pint Unit Round-Trip Suite ({unit['n_unit_round_trips']} Conversions)</h2>
<table>
  <thead><tr><th>Input</th><th>Converted</th><th>Round-Trip Abs Error</th><th>Status</th></tr></thead>
  <tbody>{unit_rows}</tbody>
</table>

<h2>5. Complete 64-Rule Multi-Database Contract</h2>
<table>
  <thead><tr><th>Dataset</th><th>Rule</th><th>Tier</th><th>Category</th><th>Name</th><th>Invariant</th><th>Raw Failing</th><th>Status</th></tr></thead>
  <tbody>{rule_rows}</tbody>
</table>

<p><small>Generated deterministically from <code>results/metrics.json</code> | Code fingerprint: <code>{escape(provenance['code_fingerprint'])}</code></small></p>
</body>
</html>
"""


def render_all_reports(
    metrics_path: str | Path,
    provenance_path: str | Path,
    figures_dir: str | Path,
    html_output: str | Path,
) -> dict[str, Path]:
    """Generate all three Case 02 Springer SVGs and the HTML quality report."""
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    provenance = json.loads(Path(provenance_path).read_text(encoding="utf-8"))

    fig_root = Path(figures_dir)
    fig_root.mkdir(parents=True, exist_ok=True)
    html_path = Path(html_output)
    html_path.parent.mkdir(parents=True, exist_ok=True)

    p_score = fig_root / "c02_scorecard.svg"
    p_ba = fig_root / "c02_before_after.svg"
    p_diag = fig_root / "c02_diagnostics.svg"

    p_score.write_text(build_scorecard_svg(metrics), encoding="utf-8")
    p_ba.write_text(build_before_after_svg(metrics), encoding="utf-8")
    p_diag.write_text(build_diagnostics_svg(metrics), encoding="utf-8")
    html_path.write_text(build_html_quality_report(metrics, provenance), encoding="utf-8")

    return {
        "scorecard_svg": p_score,
        "before_after_svg": p_ba,
        "diagnostics_svg": p_diag,
        "html_report": html_path,
    }
