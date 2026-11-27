"""Synthetic mill-data corruption generator and multi-stage quality gate."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold

from steelgate.leakage import detect_duplicates
from steelgate.loaders import UCI_ALL_COLUMNS, UCI_FAULT_COLUMNS, UCI_FEATURE_COLUMNS
from steelgate.schemas import (
    serialize_validation_report,
    validate_or_reject,
    validate_uci_frame,
)
from steelgate.units import normalize_thickness_series

_UNIT_FACTORS_FROM_MM: dict[str, float] = {
    "in": 1.0 / 25.4,
    "cm": 0.1,
    "um": 1000.0,
}


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
    """Create an explicitly **synthetic** corrupted copy of UCI D01.

    Simulates realistic steel-mill ingestion failures: mixed thickness units
    (``mm``, ``in``, ``cm``, ``um``), missing measurements, physical/geometric
    violations, invalid fault-label vectors, operator mislabelling, and
    re-ingested duplicate inspection batches.
    """
    rng = np.random.default_rng(seed)
    out = df.copy().reset_index(drop=True)
    n_base = len(out)

    # Ensure float-capable thickness before mixing units & injecting NaNs
    out["Steel_Plate_Thickness"] = out["Steel_Plate_Thickness"].astype("float64")
    out["Thickness_Unit"] = "mm"

    # 1. Mixed thickness units (~18% of rows)
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

    # Disjoint pools for structural/physical/label corruption
    perm = rng.permutation(n_base)
    n_miss = max(1, int(round(n_base * missing_frac)))
    n_phys = max(1, int(round(n_base * physics_frac)))
    n_lbl_bad = max(1, int(round(n_base * label_schema_frac)))

    miss_idx = perm[:n_miss]
    phys_idx = perm[n_miss : n_miss + n_phys]
    lbl_bad_idx = perm[n_miss + n_phys : n_miss + n_phys + n_lbl_bad]
    clean_pool = perm[n_miss + n_phys + n_lbl_bad :]

    # 2. Missing values (NaN)
    miss_cols = ["Pixels_Areas", "Steel_Plate_Thickness", "Edges_Index", "Luminosity_Index"]
    for i, idx in enumerate(miss_idx):
        col = miss_cols[i % len(miss_cols)]
        out.at[int(idx), col] = np.nan

    # 3. Physical / geometric violations
    for i, idx in enumerate(phys_idx):
        r = int(idx)
        mode = i % 4
        if mode == 0:
            out.at[r, "X_Maximum"] = float(out.at[r, "X_Minimum"]) - 5.0
        elif mode == 1:
            out.at[r, "Pixels_Areas"] = -50.0
        elif mode == 2:
            out.at[r, "Maximum_of_Luminosity"] = max(
                0.0, float(out.at[r, "Minimum_of_Luminosity"]) - 20.0
            )
        else:
            out.at[r, "TypeOfSteel_A300"] = 1
            out.at[r, "TypeOfSteel_A400"] = 1

    # 4. Invalid fault-label rows (sum == 0 or sum == 2)
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

    # 5. Operator label noise (flip single active fault to another fault class)
    n_noise = max(1, int(round(len(clean_pool) * label_noise_frac)))
    noise_idx = clean_pool[:n_noise]
    for idx in noise_idx:
        r = int(idx)
        active = next(col for col in fault_cols if int(out.at[r, col]) == 1)
        alternatives = [col for col in fault_cols if col != active]
        flipped = str(rng.choice(alternatives))
        out.at[r, active] = 0
        out.at[r, flipped] = 1

    # 6. Duplicate inspection batch rows
    n_dup = max(1, int(round(n_base * duplicate_frac)))
    dup_src = rng.choice(n_base, size=n_dup, replace=True)
    dup_rows = out.iloc[dup_src].copy()
    corrupted = pd.concat([out, dup_rows], ignore_index=True)
    shuffle_order = rng.permutation(len(corrupted))
    corrupted = corrupted.iloc[shuffle_order].reset_index(drop=True)

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


def apply_quality_gate(
    df: pd.DataFrame,
    *,
    strict_physical: bool = True,
    clean_label_noise: bool = True,
    seed: int = 42,
    oof_given_prob_max: float = 0.12,
    oof_alt_prob_min: float = 0.65,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the 4-stage SteelGate ingestion pipeline on a raw or corrupted table.

    Stages:
      1. Pint unit normalization on ``Steel_Plate_Thickness`` if ``Thickness_Unit`` is present.
      2. Row-level Pandera + physical-invariant quarantine (38 rules).
      3. Exact deduplication and conflicting-label duplicate removal.
      4. Optional training-only out-of-fold confident-learning noise quarantine.
    """
    working = df.copy()
    n_input = int(len(working))

    # Stage 1: Unit normalization via Pint
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
        norm_thickness, unit_counts = normalize_thickness_series(
            working["Steel_Plate_Thickness"], working["Thickness_Unit"], target_unit="mm"
        )
        working["Steel_Plate_Thickness"] = norm_thickness
        working = working.drop(columns=["Thickness_Unit"])

    # Stage 2: Pandera + physical invariant quarantine
    pre_report = validate_uci_frame(working, strict_physical=strict_physical)
    pass_mask = pre_report["_pass_mask"]
    n_quarantined_schema = int((~pass_mask).sum())
    clean_df = working.loc[pass_mask].copy().reset_index(drop=True)

    # Restore integer types on discrete integer columns where valid
    for col in ("TypeOfSteel_A300", "TypeOfSteel_A400", *UCI_FAULT_COLUMNS):
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].round().astype("int64")

    # Stage 3: Deduplication & label-conflict removal
    dup_info = detect_duplicates(clean_df)
    n_before_dedup = len(clean_df)
    # Drop feature-identical rows with conflicting labels entirely
    if dup_info["label_conflict_rows"] > 0:
        feat_cols = list(UCI_FEATURE_COLUMNS)
        lbl_cols = list(UCI_FAULT_COLUMNS)
        conflict_keys = (
            clean_df.groupby(feat_cols)[lbl_cols]
            .apply(lambda g: len(g.drop_duplicates()) > 1)
        )
        bad_keys = set(conflict_keys[conflict_keys].index.tolist())
        if bad_keys:
            row_keys = [tuple(r) for r in clean_df[feat_cols].itertuples(index=False, name=None)]
            keep_mask = [k not in bad_keys for k in row_keys]
            clean_df = clean_df.loc[keep_mask].copy()
    n_conflicts_dropped = n_before_dedup - len(clean_df)

    n_before_exact_dedup = len(clean_df)
    clean_df = clean_df.drop_duplicates(subset=list(UCI_ALL_COLUMNS), keep="first").reset_index(drop=True)
    n_exact_dedup = n_before_exact_dedup - len(clean_df)

    # Stage 4: Optional training-only OOF confident-learning noise filter
    n_noise_quarantined = 0
    if clean_label_noise and len(clean_df) >= 50:
        y = np.argmax(clean_df[list(UCI_FAULT_COLUMNS)].to_numpy(dtype=int), axis=1)
        class_counts = np.bincount(y, minlength=len(UCI_FAULT_COLUMNS))
        if int(class_counts.min()) >= 5:
            X = clean_df[list(UCI_FEATURE_COLUMNS)].to_numpy(dtype="float64")
            oof_probs = np.zeros((len(clean_df), len(UCI_FAULT_COLUMNS)), dtype="float64")
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for tr_idx, va_idx in skf.split(X, y):
                clf = ExtraTreesClassifier(
                    n_estimators=200, random_state=seed, n_jobs=1
                )
                clf.fit(X[tr_idx], y[tr_idx])
                oof_probs[va_idx] = clf.predict_proba(X[va_idx])

            given_prob = oof_probs[np.arange(len(clean_df)), y]
            alt_prob = oof_probs.max(axis=1)
            suspect_mask = (given_prob < oof_given_prob_max) & (alt_prob > oof_alt_prob_min)

            # Never prune a class below 5 remaining samples
            keep_mask = np.ones(len(clean_df), dtype=bool)
            remaining_counts = class_counts.copy()
            for idx in np.where(suspect_mask)[0]:
                cls_id = int(y[idx])
                if remaining_counts[cls_id] > 5:
                    keep_mask[idx] = False
                    remaining_counts[cls_id] -= 1
            n_noise_quarantined = int((~keep_mask).sum())
            clean_df = clean_df.loc[keep_mask].reset_index(drop=True)

    # Final mandatory post-gate contract verification (must pass 100%)
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
        "n_quarantined_schema": n_quarantined_schema,
        "n_conflicts_dropped": int(n_conflicts_dropped),
        "n_exact_duplicates_dropped": int(n_exact_dedup),
        "n_label_noise_quarantined": int(n_noise_quarantined),
        "pre_gate_validation": serialize_validation_report(pre_report),
        "post_gate_validation": serialize_validation_report(post_report),
    }
    return clean_df, audit
