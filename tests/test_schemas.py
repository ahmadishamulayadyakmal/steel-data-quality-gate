"""Unit tests for the 38-rule UCI Pandera schema and NEU-DET D02 manifest schema."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from steelgate.loaders import UCI_ALL_COLUMNS, UCI_FAULT_COLUMNS
from steelgate.schemas import (
    QualityGateRejectionError,
    UCI_QUALITY_RULES,
    validate_neu_det_manifest,
    validate_or_reject,
    validate_uci_frame,
)


def make_valid_uci_frame(n_rows: int = 12) -> pd.DataFrame:
    """Create a mathematically self-consistent synthetic UCI plate table."""
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
        min_lum = 90
        max_lum = 150
        mean_lum = 120.0
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


def test_uci_schema_has_38_rules_and_passes_valid_frame():
    assert len(UCI_QUALITY_RULES) == 38
    df = make_valid_uci_frame(14)
    report = validate_or_reject(df, strict_physical=True)
    assert report["passed"] is True
    assert report["total_rules"] == 38
    assert report["rules_failed"] == 0
    assert report["rows_passing"] == 14


@pytest.mark.parametrize(
    "mutator,expected_rule",
    [
        (lambda d: d.__setitem__("Pixels_Areas", d["Pixels_Areas"].astype(float).mask(d.index == 0, np.nan)), "R05"),
        (lambda d: d.loc.__setitem__((0, "X_Maximum"), 5), "R19"),
        (lambda d: d.loc.__setitem__((0, "Maximum_of_Luminosity"), 50), "R22"),
        (lambda d: d.loc.__setitem__((0, "TypeOfSteel_A400"), int(d.loc[0, "TypeOfSteel_A300"])), "R23"),
        (lambda d: d.loc.__setitem__((0, "Pastry"), 1 - int(d.loc[0, "Pastry"])), "R24"),
        (lambda d: d.loc.__setitem__((0, "LogOfAreas"), 9.9), "R27"),
        (lambda d: d.loc.__setitem__((0, "Sum_of_Luminosity"), 10), "R38"),
    ],
)
def test_uci_schema_rejects_targeted_violations(mutator, expected_rule):
    df = make_valid_uci_frame(10)
    mutator(df)
    with pytest.raises(QualityGateRejectionError) as exc_info:
        validate_or_reject(df, strict_physical=True)
    report = exc_info.value.report
    assert report["passed"] is False
    failed_ids = {r["rule_id"] for r in report["rule_results"] if not r["passed"]}
    assert expected_rule in failed_ids
    assert "STATUS: REJECTED" in report["error_report"].upper()


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
