"""Pandera schema contracts (38 UCI rules + NEU-DET camera manifest) for Case 02."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import pandera as pa

from steelgate.loaders import UCI_ALL_COLUMNS, UCI_FAULT_COLUMNS, UCI_FEATURE_COLUMNS

NEU_DET_CLASSES: tuple[str, ...] = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
)

_BINARY_COLS = ("TypeOfSteel_A300", "TypeOfSteel_A400") + UCI_FAULT_COLUMNS
_UNIT_INTERVAL_COLS = (
    "Edges_Index",
    "Empty_Index",
    "Square_Index",
    "Outside_X_Index",
    "Edges_X_Index",
    "Edges_Y_Index",
    "SigmoidOfAreas",
)


class QualityGateRejectionError(ValueError):
    """Raised when an ingested dataset fails the Pandera/physical quality gate."""

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        super().__init__(report["error_report"])


@dataclass(frozen=True)
class QualityRule:
    """Single deterministic data-quality invariant evaluated row-wise."""

    rule_id: str
    tier: str
    category: str
    name: str
    description: str
    mask_fn: Callable[[pd.DataFrame], pd.Series]


def _num_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").astype("float64")


def _finite_ge(col: str, low: float) -> Callable[[pd.DataFrame], pd.Series]:
    def _fn(df: pd.DataFrame) -> pd.Series:
        s = _num_series(df, col)
        return np.isfinite(s) & (s >= low)

    return _fn


def _finite_between(col: str, low: float, high: float) -> Callable[[pd.DataFrame], pd.Series]:
    def _fn(df: pd.DataFrame) -> pd.Series:
        s = _num_series(df, col)
        return np.isfinite(s) & (s >= low) & (s <= high)

    return _fn


def _finite_isin(col: str, allowed: tuple[float, ...]) -> Callable[[pd.DataFrame], pd.Series]:
    def _fn(df: pd.DataFrame) -> pd.Series:
        s = _num_series(df, col)
        return np.isfinite(s) & s.isin(allowed)

    return _fn


def _all_cols_between(
    cols: tuple[str, ...], low: float, high: float
) -> Callable[[pd.DataFrame], pd.Series]:
    def _fn(df: pd.DataFrame) -> pd.Series:
        mask = pd.Series(True, index=df.index, dtype="bool")
        for col in cols:
            s = _num_series(df, col)
            mask &= np.isfinite(s) & (s >= low) & (s <= high)
        return mask

    return _fn


def _all_cols_binary(cols: tuple[str, ...]) -> Callable[[pd.DataFrame], pd.Series]:
    def _fn(df: pd.DataFrame) -> pd.Series:
        mask = pd.Series(True, index=df.index, dtype="bool")
        for col in cols:
            s = _num_series(df, col)
            mask &= np.isfinite(s) & s.isin((0.0, 1.0))
        return mask

    return _fn


def _safe_log10_match(num_col: str, target_col: str, tol: float = 1e-3) -> Callable[[pd.DataFrame], pd.Series]:
    def _fn(df: pd.DataFrame) -> pd.Series:
        val = _num_series(df, num_col)
        tgt = _num_series(df, target_col)
        valid = np.isfinite(val) & np.isfinite(tgt) & (val > 0.0)
        out = pd.Series(False, index=df.index, dtype="bool")
        with np.errstate(divide="ignore", invalid="ignore"):
            diff = np.abs(tgt[valid] - np.log10(val[valid]))
        out.loc[valid] = diff <= tol
        return out

    return _fn


def _safe_diff_log10_match(
    hi_col: str, lo_col: str, target_col: str, tol: float = 1e-3
) -> Callable[[pd.DataFrame], pd.Series]:
    def _fn(df: pd.DataFrame) -> pd.Series:
        hi = _num_series(df, hi_col)
        lo = _num_series(df, lo_col)
        tgt = _num_series(df, target_col)
        delta = hi - lo
        valid = np.isfinite(delta) & np.isfinite(tgt) & (delta > 0.0)
        out = pd.Series(False, index=df.index, dtype="bool")
        with np.errstate(divide="ignore", invalid="ignore"):
            diff = np.abs(tgt[valid] - np.log10(delta[valid]))
        out.loc[valid] = diff <= tol
        return out

    return _fn


def _check_outside_x_index(df: pd.DataFrame) -> pd.Series:
    dx = _num_series(df, "X_Maximum") - _num_series(df, "X_Minimum")
    conv = _num_series(df, "Length_of_Conveyer")
    tgt = _num_series(df, "Outside_X_Index")
    valid = np.isfinite(dx) & np.isfinite(conv) & np.isfinite(tgt) & (dx > 0.0) & (conv > 0.0)
    out = pd.Series(False, index=df.index, dtype="bool")
    with np.errstate(divide="ignore", invalid="ignore"):
        out.loc[valid] = np.abs(tgt[valid] - dx[valid] / conv[valid]) <= 1e-3
    return out


def _check_empty_index(df: pd.DataFrame) -> pd.Series:
    dx = _num_series(df, "X_Maximum") - _num_series(df, "X_Minimum")
    dy = _num_series(df, "Y_Maximum") - _num_series(df, "Y_Minimum")
    area = _num_series(df, "Pixels_Areas")
    tgt = _num_series(df, "Empty_Index")
    bbox = dx * dy
    valid = (
        np.isfinite(bbox)
        & np.isfinite(area)
        & np.isfinite(tgt)
        & (dx > 0.0)
        & (dy > 0.0)
        & (bbox > 0.0)
    )
    out = pd.Series(False, index=df.index, dtype="bool")
    with np.errstate(divide="ignore", invalid="ignore"):
        out.loc[valid] = np.abs(tgt[valid] - (1.0 - area[valid] / bbox[valid])) <= 1e-3
    return out


def _check_square_index(df: pd.DataFrame) -> pd.Series:
    dx = _num_series(df, "X_Maximum") - _num_series(df, "X_Minimum")
    dy = _num_series(df, "Y_Maximum") - _num_series(df, "Y_Minimum")
    tgt = _num_series(df, "Square_Index")
    valid = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(tgt) & (dx > 0.0) & (dy > 0.0)
    out = pd.Series(False, index=df.index, dtype="bool")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.minimum(dx[valid], dy[valid]) / np.maximum(dx[valid], dy[valid])
        out.loc[valid] = np.abs(tgt[valid] - ratio) <= 1e-3
    return out


def _check_edges_x_index(df: pd.DataFrame) -> pd.Series:
    dx = _num_series(df, "X_Maximum") - _num_series(df, "X_Minimum")
    perim = _num_series(df, "X_Perimeter")
    tgt = _num_series(df, "Edges_X_Index")
    valid = np.isfinite(dx) & np.isfinite(perim) & np.isfinite(tgt) & (dx > 0.0) & (perim > 0.0)
    out = pd.Series(False, index=df.index, dtype="bool")
    with np.errstate(divide="ignore", invalid="ignore"):
        out.loc[valid] = np.abs(tgt[valid] - dx[valid] / perim[valid]) <= 1e-3
    return out


def _check_edges_y_index(df: pd.DataFrame) -> pd.Series:
    dy = _num_series(df, "Y_Maximum") - _num_series(df, "Y_Minimum")
    perim = _num_series(df, "Y_Perimeter")
    tgt = _num_series(df, "Edges_Y_Index")
    valid = np.isfinite(dy) & np.isfinite(perim) & np.isfinite(tgt) & (dy > 0.0) & (perim > 0.0)
    out = pd.Series(False, index=df.index, dtype="bool")
    with np.errstate(divide="ignore", invalid="ignore"):
        out.loc[valid] = np.abs(tgt[valid] - dy[valid] / perim[valid]) <= 1e-3
    return out


def _check_outside_global_index(df: pd.DataFrame) -> pd.Series:
    dx = _num_series(df, "X_Maximum") - _num_series(df, "X_Minimum")
    dy = _num_series(df, "Y_Maximum") - _num_series(df, "Y_Minimum")
    tgt = _num_series(df, "Outside_Global_Index")
    valid = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(tgt) & (dx > 0.0) & (dy > 0.0)
    out = pd.Series(False, index=df.index, dtype="bool")
    expected = np.where(dy[valid] > dx[valid], 1.0, np.where(dy[valid] == dx[valid], 0.5, 0.0))
    out.loc[valid] = np.abs(tgt[valid] - expected) <= 1e-6
    return out


def _check_orientation_index(df: pd.DataFrame) -> pd.Series:
    dx = _num_series(df, "X_Maximum") - _num_series(df, "X_Minimum")
    dy = _num_series(df, "Y_Maximum") - _num_series(df, "Y_Minimum")
    tgt = _num_series(df, "Orientation_Index")
    valid = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(tgt) & (dx > 0.0) & (dy > 0.0)
    out = pd.Series(False, index=df.index, dtype="bool")
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = (dy[valid] - dx[valid]) / np.maximum(dx[valid], dy[valid])
        out.loc[valid] = np.abs(tgt[valid] - expected) <= 1e-3
    return out


def _check_luminosity_index(df: pd.DataFrame) -> pd.Series:
    lum_sum = _num_series(df, "Sum_of_Luminosity")
    area = _num_series(df, "Pixels_Areas")
    tgt = _num_series(df, "Luminosity_Index")
    valid = np.isfinite(lum_sum) & np.isfinite(area) & np.isfinite(tgt) & (area > 0.0) & (lum_sum >= 0.0)
    out = pd.Series(False, index=df.index, dtype="bool")
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = (lum_sum[valid] / area[valid]) / 128.0 - 1.0
        out.loc[valid] = np.abs(tgt[valid] - expected) <= 1e-3
    return out


def _check_x_order(df: pd.DataFrame) -> pd.Series:
    hi = _num_series(df, "X_Maximum")
    lo = _num_series(df, "X_Minimum")
    return np.isfinite(hi) & np.isfinite(lo) & (hi > lo)


def _check_y_order(df: pd.DataFrame) -> pd.Series:
    hi = _num_series(df, "Y_Maximum")
    lo = _num_series(df, "Y_Minimum")
    return np.isfinite(hi) & np.isfinite(lo) & (hi > lo)


def _check_x_within_conveyer(df: pd.DataFrame) -> pd.Series:
    hi = _num_series(df, "X_Maximum")
    conv = _num_series(df, "Length_of_Conveyer")
    return np.isfinite(hi) & np.isfinite(conv) & (hi <= conv)


def _check_lum_order(df: pd.DataFrame) -> pd.Series:
    hi = _num_series(df, "Maximum_of_Luminosity")
    lo = _num_series(df, "Minimum_of_Luminosity")
    return np.isfinite(hi) & np.isfinite(lo) & (hi >= lo)


def _check_steel_grade_onehot(df: pd.DataFrame) -> pd.Series:
    a300 = _num_series(df, "TypeOfSteel_A300")
    a400 = _num_series(df, "TypeOfSteel_A400")
    return (
        np.isfinite(a300)
        & np.isfinite(a400)
        & a300.isin((0.0, 1.0))
        & a400.isin((0.0, 1.0))
        & ((a300 + a400) == 1.0)
    )


def _check_fault_label_exclusivity(df: pd.DataFrame) -> pd.Series:
    valid = pd.Series(True, index=df.index, dtype="bool")
    total = pd.Series(0.0, index=df.index, dtype="float64")
    for col in UCI_FAULT_COLUMNS:
        s = _num_series(df, col)
        valid &= np.isfinite(s) & s.isin((0.0, 1.0))
        total = total + s.fillna(0.0)
    return valid & (total == 1.0)


def _check_area_le_bbox(df: pd.DataFrame) -> pd.Series:
    dx = _num_series(df, "X_Maximum") - _num_series(df, "X_Minimum")
    dy = _num_series(df, "Y_Maximum") - _num_series(df, "Y_Minimum")
    area = _num_series(df, "Pixels_Areas")
    return np.isfinite(dx) & np.isfinite(dy) & np.isfinite(area) & (dx > 0.0) & (dy > 0.0) & (area <= dx * dy)


def _check_lum_upper_bound(df: pd.DataFrame) -> pd.Series:
    lum_sum = _num_series(df, "Sum_of_Luminosity")
    area = _num_series(df, "Pixels_Areas")
    max_lum = _num_series(df, "Maximum_of_Luminosity")
    return np.isfinite(lum_sum) & np.isfinite(area) & np.isfinite(max_lum) & (area > 0.0) & (lum_sum <= area * max_lum)


def _check_lum_lower_bound(df: pd.DataFrame) -> pd.Series:
    lum_sum = _num_series(df, "Sum_of_Luminosity")
    area = _num_series(df, "Pixels_Areas")
    min_lum = _num_series(df, "Minimum_of_Luminosity")
    return np.isfinite(lum_sum) & np.isfinite(area) & np.isfinite(min_lum) & (area > 0.0) & (lum_sum >= area * min_lum)


UCI_QUALITY_RULES: tuple[QualityRule, ...] = (
    QualityRule("R01", "Tier 1", "Domain & Nulls", "X_Minimum non-negative", "X_Minimum finite and >= 0", _finite_ge("X_Minimum", 0.0)),
    QualityRule("R02", "Tier 1", "Domain & Nulls", "X_Maximum positive", "X_Maximum finite and >= 1", _finite_ge("X_Maximum", 1.0)),
    QualityRule("R03", "Tier 1", "Domain & Nulls", "Y_Minimum non-negative", "Y_Minimum finite and >= 0", _finite_ge("Y_Minimum", 0.0)),
    QualityRule("R04", "Tier 1", "Domain & Nulls", "Y_Maximum positive", "Y_Maximum finite and >= 1", _finite_ge("Y_Maximum", 1.0)),
    QualityRule("R05", "Tier 1", "Domain & Nulls", "Pixels_Areas positive", "Pixels_Areas finite and >= 1 pixel^2", _finite_ge("Pixels_Areas", 1.0)),
    QualityRule("R06", "Tier 1", "Domain & Nulls", "X_Perimeter positive", "X_Perimeter finite and >= 1 pixel", _finite_ge("X_Perimeter", 1.0)),
    QualityRule("R07", "Tier 1", "Domain & Nulls", "Y_Perimeter positive", "Y_Perimeter finite and >= 1 pixel", _finite_ge("Y_Perimeter", 1.0)),
    QualityRule("R08", "Tier 1", "Domain & Nulls", "Sum_of_Luminosity non-negative", "Sum_of_Luminosity finite and >= 0", _finite_ge("Sum_of_Luminosity", 0.0)),
    QualityRule("R09", "Tier 1", "Domain & Nulls", "Minimum_of_Luminosity 8-bit", "Minimum_of_Luminosity in [0, 255]", _finite_between("Minimum_of_Luminosity", 0.0, 255.0)),
    QualityRule("R10", "Tier 1", "Domain & Nulls", "Maximum_of_Luminosity 8-bit", "Maximum_of_Luminosity in [0, 255]", _finite_between("Maximum_of_Luminosity", 0.0, 255.0)),
    QualityRule("R11", "Tier 1", "Domain & Nulls", "Length_of_Conveyer mill range", "Length_of_Conveyer in [500, 5000] px", _finite_between("Length_of_Conveyer", 500.0, 5000.0)),
    QualityRule("R12", "Tier 1", "Domain & Nulls", "Steel grade binary flags", "TypeOfSteel_A300 and A400 in {0, 1}", _all_cols_binary(("TypeOfSteel_A300", "TypeOfSteel_A400"))),
    QualityRule("R13", "Tier 1", "Unit & Range", "Plate thickness canonical mm", "Steel_Plate_Thickness in [10, 500] mm", _finite_between("Steel_Plate_Thickness", 10.0, 500.0)),
    QualityRule("R14", "Tier 1", "Domain & Nulls", "Normalized [0, 1] shape indices", "Edges/Empty/Square/Outside_X/Edges_X/Edges_Y/SigmoidOfAreas in [0, 1]", _all_cols_between(_UNIT_INTERVAL_COLS, 0.0, 1.0)),
    QualityRule("R15", "Tier 1", "Domain & Nulls", "Outside_Global_Index discrete", "Outside_Global_Index in {0.0, 0.5, 1.0}", _finite_isin("Outside_Global_Index", (0.0, 0.5, 1.0))),
    QualityRule("R16", "Tier 1", "Domain & Nulls", "Logarithmic indices non-negative", "LogOfAreas, Log_X_Index, Log_Y_Index >= 0", _all_cols_between(("LogOfAreas", "Log_X_Index", "Log_Y_Index"), 0.0, 100.0)),
    QualityRule("R17", "Tier 1", "Domain & Nulls", "Signed orientation/luminosity range", "Orientation_Index and Luminosity_Index in [-1, 1]", _all_cols_between(("Orientation_Index", "Luminosity_Index"), -1.0, 1.0)),
    QualityRule("R18", "Tier 1", "Label Integrity", "Fault target indicators binary", "All 7 fault indicator columns in {0, 1}", _all_cols_binary(UCI_FAULT_COLUMNS)),
    QualityRule("R19", "Tier 1", "Geometry Physics", "X_Maximum > X_Minimum", "Horizontal bounding box width dx > 0", _check_x_order),
    QualityRule("R20", "Tier 1", "Geometry Physics", "Y_Maximum > Y_Minimum", "Vertical bounding box height dy > 0", _check_y_order),
    QualityRule("R21", "Tier 1", "Geometry Physics", "X_Maximum <= Length_of_Conveyer", "Defect x-coordinate fits within conveyer width", _check_x_within_conveyer),
    QualityRule("R22", "Tier 1", "Luminosity Bounds", "Maximum_of_Luminosity >= Minimum_of_Luminosity", "Peak intensity >= minimum intensity", _check_lum_order),
    QualityRule("R23", "Tier 1", "Label Integrity", "Steel grade one-hot exclusivity", "TypeOfSteel_A300 + TypeOfSteel_A400 == 1", _check_steel_grade_onehot),
    QualityRule("R24", "Tier 1", "Label Integrity", "Fault class 1-of-7 exclusivity", "Sum of 7 fault indicator columns == 1", _check_fault_label_exclusivity),
    QualityRule("R25", "Tier 1", "Geometry Physics", "Pixels_Areas <= dx * dy", "Defect pixel area cannot exceed bounding box area", _check_area_le_bbox),
    QualityRule("R26", "Tier 1", "Luminosity Bounds", "Sum_of_Luminosity <= Area * Max_Lum", "Total luminosity cannot exceed Pixels_Areas * Maximum_of_Luminosity", _check_lum_upper_bound),
    QualityRule("R27", "Tier 1", "Math Invariant", "LogOfAreas == log10(Pixels_Areas)", "|LogOfAreas - log10(Pixels_Areas)| <= 1e-3", _safe_log10_match("Pixels_Areas", "LogOfAreas")),
    QualityRule("R28", "Tier 1", "Math Invariant", "Log_X_Index == log10(dx)", "|Log_X_Index - log10(X_Maximum - X_Minimum)| <= 1e-3", _safe_diff_log10_match("X_Maximum", "X_Minimum", "Log_X_Index")),
    QualityRule("R29", "Tier 1", "Math Invariant", "Log_Y_Index == log10(dy)", "|Log_Y_Index - log10(Y_Maximum - Y_Minimum)| <= 1e-3", _safe_diff_log10_match("Y_Maximum", "Y_Minimum", "Log_Y_Index")),
    QualityRule("R30", "Tier 1", "Math Invariant", "Outside_X_Index == dx / Conveyer", "|Outside_X_Index - dx / Length_of_Conveyer| <= 1e-3", _check_outside_x_index),
    QualityRule("R31", "Tier 1", "Math Invariant", "Empty_Index == 1 - Area / (dx*dy)", "|Empty_Index - (1 - Pixels_Areas / (dx*dy))| <= 1e-3", _check_empty_index),
    QualityRule("R32", "Tier 1", "Math Invariant", "Square_Index == min(dx,dy)/max(dx,dy)", "|Square_Index - min(dx,dy)/max(dx,dy)| <= 1e-3", _check_square_index),
    QualityRule("R33", "Tier 1", "Math Invariant", "Edges_X_Index == dx / X_Perimeter", "|Edges_X_Index - dx / X_Perimeter| <= 1e-3", _check_edges_x_index),
    QualityRule("R34", "Tier 1", "Math Invariant", "Edges_Y_Index == dy / Y_Perimeter", "|Edges_Y_Index - dy / Y_Perimeter| <= 1e-3", _check_edges_y_index),
    QualityRule("R35", "Tier 1", "Math Invariant", "Outside_Global_Index sign consistency", "1.0 if dy > dx, 0.5 if dy == dx, 0.0 if dy < dx", _check_outside_global_index),
    QualityRule("R36", "Tier 1", "Math Invariant", "Orientation_Index == (dy-dx)/max(dx,dy)", "|Orientation_Index - (dy - dx)/max(dx, dy)| <= 1e-3", _check_orientation_index),
    QualityRule("R37", "Tier 1", "Math Invariant", "Luminosity_Index == mean_lum/128 - 1", "|Luminosity_Index - ((Sum_of_Luminosity/Pixels_Areas)/128 - 1)| <= 1e-3", _check_luminosity_index),
    QualityRule("R38", "Tier 2", "Physical Conservation", "Sum_of_Luminosity >= Area * Min_Lum", "Total luminosity >= Pixels_Areas * Minimum_of_Luminosity (catches raw D01 rows 518, 591)", _check_lum_lower_bound),
)


def _df_check(rule: QualityRule) -> pa.Check:
    return pa.Check(
        rule.mask_fn,
        name=f"{rule.rule_id}: {rule.name}",
        error=f"{rule.rule_id} ({rule.description})",
    )


def build_uci_schema(*, strict_physical: bool = True) -> pa.DataFrameSchema:
    """Build the Pandera ``DataFrameSchema`` for UCI Steel Plates Faults (37 or 38 rules)."""
    columns = {
        col: pa.Column(
            nullable=False,
            checks=[
                pa.Check(
                    lambda s: pd.api.types.is_numeric_dtype(s) and bool(np.isfinite(pd.to_numeric(s, errors="coerce")).all()),
                    name=f"numeric_finite_{col}",
                    error=f"Column {col} must be finite numeric without nulls",
                )
            ],
        )
        for col in UCI_ALL_COLUMNS
    }
    active_rules = [
        rule for rule in UCI_QUALITY_RULES if strict_physical or rule.tier != "Tier 2"
    ]
    return pa.DataFrameSchema(
        columns=columns,
        checks=[_df_check(rule) for rule in active_rules],
        strict=True,
        coerce=False,
    )


def _format_error_report(
    *,
    rows_checked: int,
    rows_passing: int,
    rows_failing: int,
    missing_cols: list[str],
    extra_cols: list[str],
    rule_results: list[dict[str, Any]],
    pandera_error_count: int,
) -> str:
    lines = [
        "=== STEELGATE QUALITY REPORT ===",
        f"Rows checked: {rows_checked} | Passing: {rows_passing} | Failing: {rows_failing}",
        f"Pandera lazy schema errors recorded: {pandera_error_count}",
    ]
    if missing_cols:
        lines.append(f"Missing required columns ({len(missing_cols)}): {', '.join(missing_cols)}")
    if extra_cols:
        lines.append(f"Unexpected extra columns ({len(extra_cols)}): {', '.join(extra_cols)}")

    failed_rules = [r for r in rule_results if not r["passed"]]
    if not failed_rules and not missing_cols and not extra_cols:
        lines.append("Status: PASS — 100% of rows satisfy all active Pandera & physical rules.")
    else:
        lines.append(f"Status: REJECTED — {len(failed_rules)} rule(s) violated:")
        for item in failed_rules:
            sample = ", ".join(str(idx) for idx in item["sample_failing_rows"][:5])
            lines.append(
                f"  - [{item['rule_id']}] {item['name']} ({item['tier']}): "
                f"{item['n_failing']} failing row(s) [sample idx: {sample}]"
            )
    return "\n".join(lines)


def validate_uci_frame(df: pd.DataFrame, *, strict_physical: bool = True) -> dict[str, Any]:
    """Validate a DataFrame against all active Pandera & physical rules without mutating it."""
    schema = build_uci_schema(strict_physical=strict_physical)
    pandera_error_count = 0
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        pandera_error_count = int(len(exc.failure_cases))
    except pa.errors.SchemaError as exc:
        pandera_error_count = int(len(exc.failure_cases)) if exc.failure_cases is not None else 1

    missing_cols = [c for c in UCI_ALL_COLUMNS if c not in df.columns]
    extra_cols = [str(c) for c in df.columns if c not in UCI_ALL_COLUMNS]
    active_rules = [
        rule for rule in UCI_QUALITY_RULES if strict_physical or rule.tier != "Tier 2"
    ]

    overall_pass_mask = pd.Series(True, index=df.index, dtype="bool")
    if missing_cols or extra_cols:
        overall_pass_mask[:] = False

    rule_results: list[dict[str, Any]] = []
    for rule in active_rules:
        mask = rule.mask_fn(df).astype("bool")
        overall_pass_mask &= mask
        fail_idx = df.index[~mask].tolist()
        n_fail = int((~mask).sum())
        rule_results.append(
            {
                "rule_id": rule.rule_id,
                "tier": rule.tier,
                "category": rule.category,
                "name": rule.name,
                "description": rule.description,
                "n_failing": n_fail,
                "pass_rate": float(1.0 - n_fail / max(len(df), 1)),
                "passed": n_fail == 0,
                "sample_failing_rows": [int(i) if isinstance(i, (int, np.integer)) else str(i) for i in fail_idx[:10]],
            }
        )

    rows_checked = int(len(df))
    rows_passing = int(overall_pass_mask.sum())
    rows_failing = rows_checked - rows_passing
    rules_failed = sum(1 for r in rule_results if not r["passed"])
    rules_passed = len(rule_results) - rules_failed
    passed = (
        len(missing_cols) == 0
        and len(extra_cols) == 0
        and rows_failing == 0
        and rules_failed == 0
        and pandera_error_count == 0
    )

    failing_indices = [
        int(i) if isinstance(i, (int, np.integer)) else str(i)
        for i in df.index[~overall_pass_mask].tolist()
    ]
    error_report = _format_error_report(
        rows_checked=rows_checked,
        rows_passing=rows_passing,
        rows_failing=rows_failing,
        missing_cols=missing_cols,
        extra_cols=extra_cols,
        rule_results=rule_results,
        pandera_error_count=pandera_error_count,
    )

    return {
        "passed": bool(passed),
        "strict_physical": bool(strict_physical),
        "total_rules": len(rule_results),
        "rules_passed": rules_passed,
        "rules_failed": rules_failed,
        "rows_checked": rows_checked,
        "rows_passing": rows_passing,
        "rows_failing": rows_failing,
        "pass_rate": float(rows_passing / max(rows_checked, 1)),
        "missing_columns": missing_cols,
        "extra_columns": extra_cols,
        "pandera_error_count": pandera_error_count,
        "failing_row_indices": failing_indices,
        "rule_results": rule_results,
        "error_report": error_report,
        "_pass_mask": overall_pass_mask,
    }


def validate_or_reject(df: pd.DataFrame, *, strict_physical: bool = True) -> dict[str, Any]:
    """Validate ``df`` and raise ``QualityGateRejectionError`` if any check fails."""
    report = validate_uci_frame(df, strict_physical=strict_physical)
    if not report["passed"]:
        raise QualityGateRejectionError(report)
    return report


def serialize_validation_report(report: dict[str, Any]) -> dict[str, Any]:
    """Strip internal pandas masks before writing JSON metrics."""
    return {k: v for k, v in report.items() if not k.startswith("_")}


def build_neu_det_manifest_schema() -> pa.DataFrameSchema:
    """Build the Pandera contract for NEU-DET (D02) mill camera metadata manifests."""
    return pa.DataFrameSchema(
        columns={
            "image_id": pa.Column(str, pa.Check.str_length(min_value=1), nullable=False),
            "defect_class": pa.Column(str, pa.Check.isin(NEU_DET_CLASSES), nullable=False),
            "width_px": pa.Column(int, pa.Check.equal_to(200), coerce=True, nullable=False),
            "height_px": pa.Column(int, pa.Check.equal_to(200), coerce=True, nullable=False),
            "bbox_xmin": pa.Column(int, pa.Check.in_range(0, 199), coerce=True, nullable=False),
            "bbox_xmax": pa.Column(int, pa.Check.in_range(1, 200), coerce=True, nullable=False),
            "bbox_ymin": pa.Column(int, pa.Check.in_range(0, 199), coerce=True, nullable=False),
            "bbox_ymax": pa.Column(int, pa.Check.in_range(1, 200), coerce=True, nullable=False),
            "sha256": pa.Column(str, pa.Check.str_matches(r"^[0-9a-f]{64}$"), nullable=False),
            "mirror_licence_verified": pa.Column(bool, pa.Check.equal_to(True), coerce=True, nullable=False),
        },
        checks=[
            pa.Check(lambda d: d["bbox_xmax"] > d["bbox_xmin"], name="D02_bbox_x_order"),
            pa.Check(lambda d: d["bbox_ymax"] > d["bbox_ymin"], name="D02_bbox_y_order"),
            pa.Check(lambda d: d["bbox_xmax"] <= d["width_px"], name="D02_bbox_x_bounds"),
            pa.Check(lambda d: d["bbox_ymax"] <= d["height_px"], name="D02_bbox_y_bounds"),
        ],
        strict=True,
    )


def validate_neu_det_manifest(df: pd.DataFrame) -> dict[str, Any]:
    """Validate a NEU-DET (D02) camera metadata table against ``neu_det_manifest_v1``."""
    schema = build_neu_det_manifest_schema()
    try:
        schema.validate(df, lazy=True)
        return {"passed": True, "rows_checked": int(len(df)), "error_count": 0, "errors": []}
    except (pa.errors.SchemaErrors, pa.errors.SchemaError) as exc:
        cases = exc.failure_cases if exc.failure_cases is not None else pd.DataFrame()
        return {
            "passed": False,
            "rows_checked": int(len(df)),
            "error_count": int(len(cases)),
            "errors": cases.head(10).to_dict(orient="records"),
        }
