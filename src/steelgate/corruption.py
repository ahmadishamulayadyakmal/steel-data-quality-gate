"""Synthetic corruptions, deterministic index repair, and multi-database quality gates."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import KFold, StratifiedKFold

from steelgate.leakage import detect_duplicates
from steelgate.loaders import UCI_ALL_COLUMNS, UCI_FAULT_COLUMNS, UCI_FEATURE_COLUMNS
from steelgate.schemas import (
    serialize_validation_report,
    validate_d04_frame,
    validate_d04_or_reject,
    validate_d09_frame,
    validate_d09_or_reject,
    validate_or_reject,
    validate_uci_frame,
)
from steelgate.units import normalize_series_units, parse_formula_to_at_pct

_UNIT_FACTORS_FROM_MM: dict[str, float] = {
    "in": 1.0 / 25.4,
    "cm": 0.1,
    "um": 1000.0,
}

_STRENGTH_FACTORS_FROM_MPA: dict[str, float] = {
    "ksi": 1.0 / 6.894757293168361,
    "GPa": 0.001,
}

_ENERGY_FACTORS_FROM_KWH: dict[str, float] = {
    "Wh": 1000.0,
    "MWh": 0.001,
}

_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)?")


def _featurize_formulas_simple(formulas: pd.Series) -> np.ndarray:
    parsed = []
    vocab_set: set[str] = set()
    for f in formulas:
        comp, _ = parse_formula_to_at_pct(str(f))
        parsed.append(comp)
        vocab_set.update(comp.keys())
    vocab = sorted(vocab_set)
    mat = np.zeros((len(parsed), len(vocab)), dtype="float64")
    for i, comp in enumerate(parsed):
        for j, el in enumerate(vocab):
            mat[i, j] = comp.get(el, 0.0) / 100.0
    return mat


# ---------------------------------------------------------------------------
# D01: UCI Steel Plates Faults — Corruption, Repair, Confident Learning, Gate
# ---------------------------------------------------------------------------

def corrupt_mill_dataset(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    unit_mix_frac: float = 0.18,
    missing_frac: float = 0.04,
    physics_frac: float = 0.03,
    label_schema_frac: float = 0.02,
    label_noise_frac: float = 0.12,
    duplicate_frac: float = 0.10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create an explicitly **synthetic** corrupted copy of UCI D01."""
    rng = np.random.default_rng(seed)
    out = df.copy().reset_index(drop=True)
    n_base = len(out)

    out["Steel_Plate_Thickness"] = out["Steel_Plate_Thickness"].astype("float64")
    out["Thickness_Unit"] = "mm"

    n_unit = max(1, int(round(n_base * unit_mix_frac)))
    unit_idx = rng.choice(n_base, size=n_unit, replace=False)
    unit_choices = rng.choice(["in", "cm", "um"], size=n_unit, replace=True)
    unit_counts: dict[str, int] = {"mm": n_base - n_unit, "in": 0, "cm": 0, "um": 0}
    for idx, unit in zip(unit_idx, unit_choices):
        out.at[int(idx), "Steel_Plate_Thickness"] = float(
            out.at[int(idx), "Steel_Plate_Thickness"] * _UNIT_FACTORS_FROM_MM[str(unit)]
        )
        out.at[int(idx), "Thickness_Unit"] = str(unit)
        unit_counts[str(unit)] += 1

    perm = rng.permutation(n_base)
    n_miss = max(1, int(round(n_base * missing_frac)))
    n_phys = max(1, int(round(n_base * physics_frac)))
    n_lbl_bad = max(1, int(round(n_base * label_schema_frac)))

    miss_idx = perm[:n_miss]
    phys_idx = perm[n_miss : n_miss + n_phys]
    lbl_bad_idx = perm[n_miss + n_phys : n_miss + n_phys + n_lbl_bad]
    clean_pool = perm[n_miss + n_phys + n_lbl_bad :]

    miss_cols = ["Pixels_Areas", "Steel_Plate_Thickness", "Edges_Index", "Luminosity_Index"]
    for i, idx in enumerate(miss_idx):
        out.at[int(idx), miss_cols[i % len(miss_cols)]] = np.nan

    for i, idx in enumerate(phys_idx):
        r = int(idx)
        mode = i % 4
        if mode == 0:
            out.at[r, "X_Maximum"] = float(out.at[r, "X_Minimum"]) - 5.0
        elif mode == 1:
            out.at[r, "Pixels_Areas"] = -50.0
        elif mode == 2:
            out.at[r, "Maximum_of_Luminosity"] = max(0.0, float(out.at[r, "Minimum_of_Luminosity"]) - 20.0)
        else:
            out.at[r, "TypeOfSteel_A300"] = 1
            out.at[r, "TypeOfSteel_A400"] = 1

    fault_cols = list(UCI_FAULT_COLUMNS)
    for i, idx in enumerate(lbl_bad_idx):
        r = int(idx)
        if i % 2 == 0:
            for col in fault_cols:
                out.at[r, col] = 0
        else:
            current = [col for col in fault_cols if int(out.at[r, col]) == 1]
            other = next(col for col in fault_cols if col not in current)
            out.at[r, other] = 1

    n_noise = max(1, int(round(len(clean_pool) * label_noise_frac)))
    for idx in clean_pool[:n_noise]:
        r = int(idx)
        active = next(col for col in fault_cols if int(out.at[r, col]) == 1)
        alternatives = [col for col in fault_cols if col != active]
        flipped = str(rng.choice(alternatives))
        out.at[r, active] = 0
        out.at[r, flipped] = 1

    n_dup = max(1, int(round(n_base * duplicate_frac)))
    dup_rows = out.iloc[rng.choice(n_base, size=n_dup, replace=True)].copy()
    corrupted = pd.concat([out, dup_rows], ignore_index=True)
    corrupted = corrupted.iloc[rng.permutation(len(corrupted))].reset_index(drop=True)

    manifest: dict[str, Any] = {
        "synthetic": True,
        "source_dataset": "UCI Steel Plates Faults (D01) — synthetic corruption",
        "seed": int(seed),
        "n_source_rows": int(n_base),
        "n_corrupted_rows": int(len(corrupted)),
        "injected_failures": {
            "mixed_thickness_units_rows": int(n_unit),
            "thickness_unit_breakdown": unit_counts,
            "missing_value_rows": int(n_miss),
            "physical_geometry_violation_rows": int(n_phys),
            "label_schema_violation_rows": int(n_lbl_bad),
            "label_noise_flipped_rows": int(n_noise),
            "appended_duplicate_rows": int(n_dup),
        },
    }
    return corrupted, manifest


def repair_uci_derived_indices(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Deterministically recompute missing derived indices when primary sensors are valid."""
    out = df.copy()
    x0 = pd.to_numeric(out["X_Minimum"], errors="coerce")
    x1 = pd.to_numeric(out["X_Maximum"], errors="coerce")
    y0 = pd.to_numeric(out["Y_Minimum"], errors="coerce")
    y1 = pd.to_numeric(out["Y_Maximum"], errors="coerce")
    area = pd.to_numeric(out["Pixels_Areas"], errors="coerce")
    lsum = pd.to_numeric(out["Sum_of_Luminosity"], errors="coerce")

    dx = x1 - x0
    dy = y1 - y0
    valid_lum = (
        np.isfinite(dx) & np.isfinite(dy) & (dx > 0) & (dy > 0)
        & np.isfinite(area) & (area >= 1) & (area <= dx * dy)
        & np.isfinite(lsum) & (lsum >= 0)
    )
    li = pd.to_numeric(out["Luminosity_Index"], errors="coerce")
    rep_li = valid_lum & (~np.isfinite(li))
    repaired_count = int(rep_li.sum())
    if repaired_count > 0:
        out.loc[rep_li, "Luminosity_Index"] = ((lsum[rep_li] / area[rep_li]) / 128.0 - 1.0).round(4)
    return out, repaired_count


def apply_quality_gate(
    df: pd.DataFrame,
    *,
    strict_physical: bool = True,
    clean_label_noise: bool = True,
    repair_derived: bool = True,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the 4-stage SteelGate pipeline on UCI D01 (with Northcutt Confident Learning)."""
    working = df.copy()
    n_input = int(len(working))

    # Stage 1: Pint unit normalization
    unit_counts: dict[str, int] = {"mm": n_input}
    raw_thickness_stats: dict[str, float] = {}
    if "Steel_Plate_Thickness" in working.columns:
        numeric_raw = pd.to_numeric(working["Steel_Plate_Thickness"], errors="coerce")
        finite_raw = numeric_raw[np.isfinite(numeric_raw)]
        if len(finite_raw) > 0:
            raw_thickness_stats = {
                "min": float(finite_raw.min()),
                "median": float(finite_raw.median()),
                "max": float(finite_raw.max()),
            }
    if "Thickness_Unit" in working.columns:
        norm_thickness, unit_counts = normalize_series_units(
            working["Steel_Plate_Thickness"], working["Thickness_Unit"], target_unit="mm"
        )
        working["Steel_Plate_Thickness"] = norm_thickness
        working = working.drop(columns=["Thickness_Unit"])

    # Stage 2: Deterministic derived-index repair + 38-rule Pandera quarantine
    n_repaired = 0
    if repair_derived:
        working, n_repaired = repair_uci_derived_indices(working)

    pre_report = validate_uci_frame(working, strict_physical=strict_physical)
    pass_mask = pre_report["_pass_mask"]
    n_quarantined_schema = int((~pass_mask).sum())
    clean_df = working.loc[pass_mask].copy().reset_index(drop=True)

    for col in ("TypeOfSteel_A300", "TypeOfSteel_A400", *UCI_FAULT_COLUMNS):
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].round().astype("int64")

    # Stage 3: Deduplication & label-conflict removal
    dup_info = detect_duplicates(clean_df)
    n_before_dedup = len(clean_df)
    if dup_info["label_conflict_rows"] > 0:
        feat_cols = list(UCI_FEATURE_COLUMNS)
        lbl_cols = list(UCI_FAULT_COLUMNS)
        conflict_keys = clean_df.groupby(feat_cols)[lbl_cols].apply(lambda g: len(g.drop_duplicates()) > 1)
        bad_keys = set(conflict_keys[conflict_keys].index.tolist())
        if bad_keys:
            row_keys = [tuple(r) for r in clean_df[feat_cols].itertuples(index=False, name=None)]
            clean_df = clean_df.loc[[k not in bad_keys for k in row_keys]].copy()
    n_conflicts_dropped = n_before_dedup - len(clean_df)

    n_before_exact_dedup = len(clean_df)
    clean_df = clean_df.drop_duplicates(subset=list(UCI_ALL_COLUMNS), keep="first").reset_index(drop=True)
    n_exact_dedup = n_before_exact_dedup - len(clean_df)

    # Stage 4: Training-only Northcutt class-adaptive Confident Learning
    n_noise_quarantined = 0
    if clean_label_noise and len(clean_df) >= 50:
        y = np.argmax(clean_df[list(UCI_FAULT_COLUMNS)].to_numpy(dtype=int), axis=1)
        n_cls = len(UCI_FAULT_COLUMNS)
        class_counts = np.bincount(y, minlength=n_cls)
        if int(class_counts.min()) >= 5:
            X = clean_df[list(UCI_FEATURE_COLUMNS)].to_numpy(dtype="float64")
            oof_probs = np.zeros((len(clean_df), n_cls), dtype="float64")
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for tr_idx, va_idx in skf.split(X, y):
                clf = ExtraTreesClassifier(n_estimators=250, random_state=seed, n_jobs=1)
                clf.fit(X[tr_idx], y[tr_idx])
                oof_probs[va_idx] = clf.predict_proba(X[va_idx])

            t_j = np.array([oof_probs[y == j, j].mean() if (y == j).any() else 0.5 for j in range(n_cls)])
            pred_cls = np.argmax(oof_probs, axis=1)
            pred_prob = oof_probs.max(axis=1)
            given_prob = oof_probs[np.arange(len(clean_df)), y]
            suspect_mask = (pred_cls != y) & (pred_prob >= 0.90 * t_j[pred_cls]) & (given_prob < 0.18 * t_j[y])

            keep_mask = np.ones(len(clean_df), dtype=bool)
            remaining_counts = class_counts.copy()
            for idx in np.argsort(given_prob):
                if suspect_mask[idx]:
                    cls_id = int(y[idx])
                    if remaining_counts[cls_id] > 5:
                        keep_mask[idx] = False
                        remaining_counts[cls_id] -= 1
            n_noise_quarantined = int((~keep_mask).sum())
            clean_df = clean_df.loc[keep_mask].reset_index(drop=True)

    post_report = validate_or_reject(clean_df, strict_physical=strict_physical)

    norm_thickness_stats: dict[str, float] = {}
    if len(clean_df) > 0:
        s_clean = clean_df["Steel_Plate_Thickness"]
        norm_thickness_stats = {
            "min": float(s_clean.min()),
            "median": float(s_clean.median()),
            "max": float(s_clean.max()),
        }

    audit: dict[str, Any] = {
        "n_input_rows": n_input,
        "n_output_rows": int(len(clean_df)),
        "unit_counts": unit_counts,
        "raw_thickness_stats": raw_thickness_stats,
        "normalized_thickness_stats_mm": norm_thickness_stats,
        "n_repaired_derived_rows": n_repaired,
        "n_quarantined_schema": n_quarantined_schema,
        "n_conflicts_dropped": int(n_conflicts_dropped),
        "n_exact_duplicates_dropped": int(n_exact_dedup),
        "n_label_noise_quarantined": int(n_noise_quarantined),
        "pre_gate_validation": serialize_validation_report(pre_report),
        "post_gate_validation": serialize_validation_report(post_report),
    }
    return clean_df, audit


# ---------------------------------------------------------------------------
# D04: MatBench Steels — Synthetic Corruption & Quality Gate
# ---------------------------------------------------------------------------

def corrupt_d04_dataset(
    df: pd.DataFrame,
    *,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create an explicitly **synthetic** corrupted copy of MatBench D04."""
    rng = np.random.default_rng(seed)
    out = df.copy().reset_index(drop=True)
    n = len(out)
    out["yield_strength_mpa"] = out["yield_strength_mpa"].astype("float64")
    out["Strength_Unit"] = "MPa"

    perm = rng.permutation(n)
    n_ksi = max(1, int(round(0.20 * n)))
    n_gpa = max(1, int(round(0.10 * n)))
    n_bad = max(1, int(round(0.05 * n)))
    n_out = max(1, int(round(0.05 * n)))

    for idx in perm[:n_ksi]:
        r = int(idx)
        out.at[r, "yield_strength_mpa"] = float(out.at[r, "yield_strength_mpa"] * _STRENGTH_FACTORS_FROM_MPA["ksi"])
        out.at[r, "Strength_Unit"] = "ksi"
    for idx in perm[n_ksi : n_ksi + n_gpa]:
        r = int(idx)
        out.at[r, "yield_strength_mpa"] = float(out.at[r, "yield_strength_mpa"] * _STRENGTH_FACTORS_FROM_MPA["GPa"])
        out.at[r, "Strength_Unit"] = "GPa"
    for k, idx in enumerate(perm[n_ksi + n_gpa : n_ksi + n_gpa + n_bad]):
        out.at[int(idx), "yield_strength_mpa"] = -250.0 if k % 2 == 0 else np.nan
    for idx in perm[n_ksi + n_gpa + n_bad : n_ksi + n_gpa + n_bad + n_out]:
        out.at[int(idx), "yield_strength_mpa"] = float(out.at[int(idx), "yield_strength_mpa"]) + 2200.0

    n_dup = max(1, int(round(0.10 * n)))
    dups = out.iloc[rng.choice(n, size=n_dup, replace=True)].copy()
    corrupted = pd.concat([out, dups], ignore_index=True)
    corrupted = corrupted.iloc[rng.permutation(len(corrupted))].reset_index(drop=True)

    manifest = {
        "synthetic": True,
        "source_dataset": "MatBench matbench_steels (D04) — synthetic corruption",
        "seed": int(seed),
        "n_source_rows": int(n),
        "n_corrupted_rows": int(len(corrupted)),
        "injected_failures": {
            "ksi_unit_rows": int(n_ksi),
            "gpa_unit_rows": int(n_gpa),
            "negative_or_missing_rows": int(n_bad),
            "extreme_outlier_rows": int(n_out),
            "appended_duplicate_rows": int(n_dup),
        },
    }
    return corrupted, manifest


def apply_d04_quality_gate(
    df: pd.DataFrame,
    *,
    clean_outliers: bool = True,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the D04 alloy quality gate: Pint MPa norm -> M01-M11 quarantine -> dedup -> MAD filter."""
    working = df.copy()
    n_input = int(len(working))

    unit_counts: dict[str, int] = {"MPa": n_input}
    if "Strength_Unit" in working.columns:
        norm_s, unit_counts = normalize_series_units(
            working["yield_strength_mpa"], working["Strength_Unit"], target_unit="MPa"
        )
        working["yield_strength_mpa"] = norm_s
        working = working.drop(columns=["Strength_Unit"])

    # Evaluate structural & physical M01-M11 (allow normalized raw sums via strict_physical=False)
    pre_report = validate_d04_frame(working, strict_physical=False)
    clean_df = working.loc[pre_report["_pass_mask"]].copy().reset_index(drop=True)
    n_quarantined_schema = n_input - len(clean_df)

    n_before_dedup = len(clean_df)
    clean_df = clean_df.drop_duplicates(subset=["composition", "yield_strength_mpa"], keep="first").reset_index(drop=True)
    clean_df = clean_df.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)
    n_dedup = n_before_dedup - len(clean_df)

    n_outliers = 0
    if clean_outliers and len(clean_df) >= 25:
        X = _featurize_formulas_simple(clean_df["composition"])
        y = clean_df["yield_strength_mpa"].to_numpy(dtype="float64")
        oof = np.zeros(len(y), dtype="float64")
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for tr_i, va_i in kf.split(X):
            reg = ExtraTreesRegressor(n_estimators=150, random_state=seed, n_jobs=1)
            reg.fit(X[tr_i], y[tr_i])
            oof[va_i] = reg.predict(X[va_i])
        res = np.abs(y - oof)
        med = float(np.median(res))
        mad = max(float(np.median(np.abs(res - med))), 1.0)
        keep = res <= (med + 5.0 * 1.4826 * mad)
        n_outliers = int((~keep).sum())
        clean_df = clean_df.loc[keep].reset_index(drop=True)

    post_report = validate_d04_or_reject(clean_df, strict_physical=False)
    audit = {
        "n_input_rows": n_input,
        "n_output_rows": int(len(clean_df)),
        "unit_counts": unit_counts,
        "n_quarantined_schema": int(n_quarantined_schema),
        "n_duplicates_dropped": int(n_dedup),
        "n_oof_outliers_quarantined": int(n_outliers),
        "pre_gate_validation": serialize_validation_report(pre_report),
        "post_gate_validation": serialize_validation_report(post_report),
    }
    return clean_df, audit


# ---------------------------------------------------------------------------
# D09: UCI Steel Energy — Synthetic Corruption & Quality Gate
# ---------------------------------------------------------------------------

def corrupt_d09_dataset(
    df: pd.DataFrame,
    *,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create an explicitly **synthetic** corrupted copy of UCI Steel Energy D09."""
    rng = np.random.default_rng(seed)
    out = df.copy().reset_index(drop=True)
    n = len(out)
    out["Usage_kWh"] = out["Usage_kWh"].astype("float64")
    out["Energy_Unit"] = "kWh"

    perm = rng.permutation(n)
    n_wh = max(1, int(round(0.10 * n)))
    n_mwh = max(1, int(round(0.10 * n)))
    n_bad_kwh = max(1, int(round(0.03 * n)))
    n_bad_pf = max(1, int(round(0.02 * n)))

    wh_idx = perm[:n_wh]
    mwh_idx = perm[n_wh : n_wh + n_mwh]
    out.loc[wh_idx, "Usage_kWh"] = out.loc[wh_idx, "Usage_kWh"] * _ENERGY_FACTORS_FROM_KWH["Wh"]
    out.loc[wh_idx, "Energy_Unit"] = "Wh"
    out.loc[mwh_idx, "Usage_kWh"] = out.loc[mwh_idx, "Usage_kWh"] * _ENERGY_FACTORS_FROM_KWH["MWh"]
    out.loc[mwh_idx, "Energy_Unit"] = "MWh"

    bad_kwh_idx = perm[n_wh + n_mwh : n_wh + n_mwh + n_bad_kwh]
    half = len(bad_kwh_idx) // 2
    out.loc[bad_kwh_idx[:half], "Usage_kWh"] = -45.0
    out.loc[bad_kwh_idx[half:], "Usage_kWh"] = np.nan

    bad_pf_idx = perm[n_wh + n_mwh + n_bad_kwh : n_wh + n_mwh + n_bad_kwh + n_bad_pf]
    out.loc[bad_pf_idx, "Lagging_Current_Power_Factor"] = 142.0

    n_dup = max(1, int(round(0.08 * n)))
    dups = out.iloc[rng.choice(n, size=n_dup, replace=True)].copy()
    corrupted = pd.concat([out, dups], ignore_index=True)
    corrupted = corrupted.iloc[rng.permutation(len(corrupted))].reset_index(drop=True)

    manifest = {
        "synthetic": True,
        "source_dataset": "UCI Steel Industry Energy Consumption (D09) — synthetic corruption",
        "seed": int(seed),
        "n_source_rows": int(n),
        "n_corrupted_rows": int(len(corrupted)),
        "injected_failures": {
            "wh_unit_rows": int(n_wh),
            "mwh_unit_rows": int(n_mwh),
            "negative_or_missing_kwh_rows": int(n_bad_kwh),
            "invalid_power_factor_rows": int(n_bad_pf),
            "appended_duplicate_rows": int(n_dup),
        },
    }
    return corrupted, manifest


def apply_d09_quality_gate(
    df: pd.DataFrame,
    *,
    strict_physical: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the D09 time-series quality gate: Pint kWh norm -> E01-E13/14 quarantine -> dedup."""
    working = df.copy()
    n_input = int(len(working))

    unit_counts: dict[str, int] = {"kWh": n_input}
    if "Energy_Unit" in working.columns:
        norm_e, unit_counts = normalize_series_units(
            working["Usage_kWh"], working["Energy_Unit"], target_unit="kWh"
        )
        working["Usage_kWh"] = norm_e
        working = working.drop(columns=["Energy_Unit"])

    pre_report = validate_d09_frame(working, strict_physical=strict_physical)
    clean_df = working.loc[pre_report["_pass_mask"]].copy().reset_index(drop=True)
    n_quarantined_schema = n_input - len(clean_df)

    # Sort chronologically and ensure strict timestamp uniqueness
    clean_df["_dt"] = pd.to_datetime(clean_df["date"], format="%d/%m/%Y %H:%M")
    n_before_dedup = len(clean_df)
    clean_df = clean_df.sort_values("_dt").drop_duplicates(subset=["_dt"], keep="first").drop(columns=["_dt"]).reset_index(drop=True)
    n_dedup = n_before_dedup - len(clean_df)

    post_report = validate_d09_or_reject(clean_df, strict_physical=strict_physical)
    audit = {
        "n_input_rows": n_input,
        "n_output_rows": int(len(clean_df)),
        "unit_counts": unit_counts,
        "n_quarantined_schema": int(n_quarantined_schema),
        "n_timestamp_duplicates_dropped": int(n_dedup),
        "pre_gate_validation": serialize_validation_report(pre_report),
        "post_gate_validation": serialize_validation_report(post_report),
    }
    return clean_df, audit
