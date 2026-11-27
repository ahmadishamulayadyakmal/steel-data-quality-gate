"""Unit tests for the 64-rule multi-database Pandera schemas (D01, D04, D09, D02)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from steelgate.loaders import D09_ENERGY_COLUMNS, UCI_ALL_COLUMNS, UCI_FAULT_COLUMNS
from steelgate.schemas import (
    D04_QUALITY_RULES,
    D09_QUALITY_RULES,
    QualityGateRejectionError,
    UCI_QUALITY_RULES,
    validate_d04_or_reject,
    validate_d09_or_reject,
    validate_neu_det_manifest,
    validate_or_reject,
)


def make_valid_uci_frame(n_rows: int = 12) -> pd.DataFrame:
    """Create a mathematically self-consistent synthetic UCI D01 plate table."""
    rows: list[dict[str, float | int]] = []
    for i in range(n_rows):
        x0 = 10 + i * 5
        x1 = x0 + 20 + (i % 3) * 10
        y0 = 1000 + i * 500
        y1 = y0 + 15 + (i % 4) * 5
        dx = x1 - x0
        dy = y1 - y0
        area = max(10, int(0.6 * dx * dy))
        x_perim = dx + 6
        y_perim = dy + 4
        min_lum, max_lum, mean_lum = 90, 150, 120.0
        lum_sum = int(round(area * mean_lum))
        conv = 1600
        a300 = 1 if i % 2 == 0 else 0
        a400 = 1 - a300
        row: dict[str, float | int] = {
            "X_Minimum": x0,
            "X_Maximum": x1,
            "Y_Minimum": y0,
            "Y_Maximum": y1,
            "Pixels_Areas": area,
            "X_Perimeter": x_perim,
            "Y_Perimeter": y_perim,
            "Sum_of_Luminosity": lum_sum,
            "Minimum_of_Luminosity": min_lum,
            "Maximum_of_Luminosity": max_lum,
            "Length_of_Conveyer": conv,
            "TypeOfSteel_A300": a300,
            "TypeOfSteel_A400": a400,
            "Steel_Plate_Thickness": 80,
            "Edges_Index": 0.45,
            "Empty_Index": round(1.0 - area / (dx * dy), 6),
            "Square_Index": round(min(dx, dy) / max(dx, dy), 6),
            "Outside_X_Index": round(dx / conv, 6),
            "Edges_X_Index": round(dx / x_perim, 6),
            "Edges_Y_Index": round(dy / y_perim, 6),
            "Outside_Global_Index": 1.0 if dy > dx else (0.5 if dy == dx else 0.0),
            "LogOfAreas": round(math.log10(area), 6),
            "Log_X_Index": round(math.log10(dx), 6),
            "Log_Y_Index": round(math.log10(dy), 6),
            "Orientation_Index": round((dy - dx) / max(dx, dy), 6),
            "Luminosity_Index": round((lum_sum / area) / 128.0 - 1.0, 6),
            "SigmoidOfAreas": 0.5,
        }
        active_fault = UCI_FAULT_COLUMNS[i % len(UCI_FAULT_COLUMNS)]
        for fc in UCI_FAULT_COLUMNS:
            row[fc] = 1 if fc == active_fault else 0
        rows.append(row)
    return pd.DataFrame(rows, columns=list(UCI_ALL_COLUMNS))


def make_valid_d04_frame(n_rows: int = 30) -> pd.DataFrame:
    """Create a valid synthetic MatBench D04 alloy table."""
    rows = []
    for i in range(n_rows):
        c_frac = 0.005 + 0.001 * (i % 6)
        cr_frac = 0.05 + 0.01 * (i % 5)
        fe_frac = round(1.0 - c_frac - cr_frac, 6)
        rows.append(
            {
                "id": f"syn-steel-{i:03d}",
                "composition": f"Fe{fe_frac:.6f}Cr{cr_frac:.6f}C{c_frac:.6f}",
                "yield_strength_mpa": 950.0 + 35.0 * (i % 10) + 5.0 * i,
            }
        )
    return pd.DataFrame(rows)


def make_valid_d09_frame(n_rows: int = 20) -> pd.DataFrame:
    """Create a valid synthetic UCI Steel Energy D09 table satisfying E01-E14."""
    base_ts = pd.Timestamp("2018-01-01 00:00:00")
    rows = []
    for i in range(n_rows):
        ts = base_ts + pd.Timedelta(minutes=15 * i)
        p = 30.0 + (i % 5) * 10.0
        q_lag = 12.0 + (i % 3) * 4.0
        pf_lag = round(100.0 * p / math.sqrt(p * p + q_lag * q_lag), 2)
        rows.append(
            {
                "date": ts.strftime("%d/%m/%Y %H:%M"),
                "Usage_kWh": p,
                "Lagging_Current_Reactive.Power_kVarh": q_lag,
                "Leading_Current_Reactive_Power_kVarh": 0.0,
                "CO2(tCO2)": round(0.00045 * p, 2),
                "Lagging_Current_Power_Factor": pf_lag,
                "Leading_Current_Power_Factor": 100.0,
                "NSM": ts.hour * 3600 + ts.minute * 60,
                "WeekStatus": "Weekend" if ts.day_name() in ("Saturday", "Sunday") else "Weekday",
                "Day_of_week": ts.day_name(),
                "Load_Type": ("Light_Load", "Medium_Load", "Maximum_Load")[i % 3],
            }
        )
    return pd.DataFrame(rows, columns=list(D09_ENERGY_COLUMNS))


def test_total_rule_count_is_64_across_three_databases():
    assert len(UCI_QUALITY_RULES) == 38
    assert len(D04_QUALITY_RULES) == 12
    assert len(D09_QUALITY_RULES) == 14
    assert len(UCI_QUALITY_RULES) + len(D04_QUALITY_RULES) + len(D09_QUALITY_RULES) == 64


def test_uci_schema_passes_valid_and_rejects_violations():
    df = make_valid_uci_frame(14)
    report = validate_or_reject(df, strict_physical=True)
    assert report["passed"] is True
    assert report["total_rules"] == 38

    bad = df.copy()
    bad.loc[0, "Sum_of_Luminosity"] = 10
    with pytest.raises(QualityGateRejectionError) as exc_info:
        validate_or_reject(bad, strict_physical=True)
    failed_ids = {r["rule_id"] for r in exc_info.value.report["rule_results"] if not r["passed"]}
    assert "R38" in failed_ids


def test_d04_schema_passes_valid_and_rejects_violations():
    df = make_valid_d04_frame(12)
    rep = validate_d04_or_reject(df, strict_physical=True)
    assert rep["passed"] is True
    assert rep["total_rules"] == 12

    bad = df.copy()
    bad.loc[0, "yield_strength_mpa"] = -50.0
    bad.loc[1, "composition"] = "Cu0.9Fe0.1"  # fails M05 Fe-base >= 50%
    bad.loc[2, "composition"] = "Fe0.80Cr0.15"  # fails M12 sum != 1.0
    with pytest.raises(QualityGateRejectionError) as exc_info:
        validate_d04_or_reject(bad, strict_physical=True)
    failed_ids = {r["rule_id"] for r in exc_info.value.report["rule_results"] if not r["passed"]}
    assert {"M05", "M09", "M12"}.issubset(failed_ids)


def test_d09_schema_passes_valid_and_rejects_violations():
    df = make_valid_d09_frame(12)
    rep = validate_d09_or_reject(df, strict_physical=True)
    assert rep["passed"] is True
    assert rep["total_rules"] == 14

    bad = df.copy()
    bad.loc[0, "Usage_kWh"] = -12.0
    bad.loc[1, "Lagging_Current_Power_Factor"] = 50.0  # fails E14 AC physics
    bad.loc[2, "NSM"] = 123
    with pytest.raises(QualityGateRejectionError) as exc_info:
        validate_d09_or_reject(bad, strict_physical=True)
    failed_ids = {r["rule_id"] for r in exc_info.value.report["rule_results"] if not r["passed"]}
    assert {"E04", "E10", "E14"}.issubset(failed_ids)


def test_neu_det_manifest_schema_passes_and_rejects_unverified_mirror():
    valid = pd.DataFrame(
        [
            {
                "image_id": "crazing_001.jpg",
                "defect_class": "crazing",
                "width_px": 200,
                "height_px": 200,
                "bbox_xmin": 10,
                "bbox_xmax": 110,
                "bbox_ymin": 20,
                "bbox_ymax": 150,
                "sha256": "a" * 64,
                "mirror_licence_verified": True,
            }
        ]
    )
    assert validate_neu_det_manifest(valid)["passed"] is True

    bad = valid.copy()
    bad.loc[0, "mirror_licence_verified"] = False
    bad.loc[0, "width_px"] = 256
    res = validate_neu_det_manifest(bad)
    assert res["passed"] is False
    assert res["error_count"] >= 2
