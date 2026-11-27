"""Pint-powered unit normalization, round-trip verification, and alloy balance checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pint

UREG = pint.UnitRegistry()

# Canonical IUPAC standard atomic weights (g/mol) for common steel alloy elements.
ATOMIC_WEIGHTS: dict[str, float] = {
    "Fe": 55.845,
    "C": 12.011,
    "Mn": 54.938044,
    "Si": 28.085,
    "Cr": 51.9961,
    "Ni": 58.6934,
    "Mo": 95.95,
    "V": 50.9415,
    "Nb": 92.90637,
    "Ti": 47.867,
    "Al": 26.9815385,
    "Cu": 63.546,
    "Co": 58.933194,
    "W": 183.84,
    "B": 10.81,
    "N": 14.007,
    "P": 30.973761998,
    "S": 32.06,
    "Sn": 118.710,
    "As": 74.921595,
}

UNIT_ALIASES: dict[str, str] = {
    "mm": "millimeter",
    "millimeter": "millimeter",
    "millimeters": "millimeter",
    "cm": "centimeter",
    "centimeter": "centimeter",
    "m": "meter",
    "meter": "meter",
    "um": "micrometer",
    "µm": "micrometer",
    "micron": "micrometer",
    "micrometer": "micrometer",
    "in": "inch",
    "inch": "inch",
    "inches": "inch",
    "mil": "milliinch",
    "milliinch": "milliinch",
    "thou": "thou",
    "mpa": "megapascal",
    "MPa": "megapascal",
    "gpa": "gigapascal",
    "GPa": "gigapascal",
    "pa": "pascal",
    "Pa": "pascal",
    "ksi": "kilopound_force_per_square_inch",
    "psi": "pound_force_per_square_inch",
    "degC": "degC",
    "C": "degC",
    "K": "kelvin",
    "degF": "degF",
    "F": "degF",
}


def resolve_unit(unit: str) -> str:
    """Resolve common metallurgical unit shorthand into canonical Pint unit names."""
    cleaned = str(unit).strip()
    if cleaned in UNIT_ALIASES:
        return UNIT_ALIASES[cleaned]
    if cleaned.lower() in UNIT_ALIASES:
        return UNIT_ALIASES[cleaned.lower()]
    raise ValueError(f"Unsupported or unrecognized unit: {unit!r}")


def convert_scalar(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a scalar value between two compatible physical units via Pint."""
    src = resolve_unit(from_unit)
    dst = resolve_unit(to_unit)
    return float(UREG.Quantity(float(value), src).to(dst).magnitude)


def verify_round_trip(
    value: float,
    from_unit: str,
    to_unit: str,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Verify ``from_unit -> to_unit -> from_unit`` numerical round-trip fidelity."""
    converted = convert_scalar(value, from_unit, to_unit)
    round_tripped = convert_scalar(converted, to_unit, from_unit)
    abs_error = abs(round_tripped - float(value))
    scale = max(abs(float(value)), 1.0)
    rel_error = abs_error / scale
    passed = abs_error <= max(atol, rtol * scale)
    return {
        "original": float(value),
        "from_unit": from_unit,
        "to_unit": to_unit,
        "converted": converted,
        "round_tripped": round_tripped,
        "abs_error": abs_error,
        "rel_error": rel_error,
        "passed": bool(passed),
    }


def normalize_thickness_series(
    values: pd.Series,
    units: pd.Series,
    *,
    target_unit: str = "mm",
) -> tuple[pd.Series, dict[str, int]]:
    """Normalize a mixed-unit thickness column into ``target_unit`` (default ``mm``).

    Preserves NaN values so downstream schema rules can explicitly flag missingness.
    Raises ``ValueError`` on unrecognized unit labels.
    """
    dst = resolve_unit(target_unit)
    normalized = pd.Series(np.nan, index=values.index, dtype="float64")
    unit_counts: dict[str, int] = {}

    for raw_unit in units.dropna().unique():
        raw_str = str(raw_unit).strip()
        src = resolve_unit(raw_str)
        mask = units.astype(str).str.strip() == raw_str
        unit_counts[raw_str] = int(mask.sum())
        subset = pd.to_numeric(values.loc[mask], errors="coerce")
        factor = float(UREG.Quantity(1.0, src).to(dst).magnitude)
        normalized.loc[mask] = subset * factor

    # Any row with a missing unit label keeps NaN or assumes missing contract
    return normalized, unit_counts


def check_composition_mass_balance(
    wt_pct: dict[str, float],
    *,
    tolerance: float = 0.05,
) -> dict[str, Any]:
    """Check ``Sigma wt% == 100`` within ``tolerance`` and non-negativity per element."""
    errors: list[str] = []
    total = 0.0
    for element, pct in wt_pct.items():
        if element not in ATOMIC_WEIGHTS:
            errors.append(f"Unknown alloy element: {element!r}")
            continue
        val = float(pct)
        if not np.isfinite(val) or val < 0.0 or val > 100.0:
            errors.append(f"Out-of-range wt% for {element}: {val}")
        total += val

    deviation = abs(total - 100.0)
    if deviation > tolerance:
        errors.append(
            f"Mass-balance violation: sum(wt%) = {total:.4f} (deviation {deviation:.4f} > {tolerance})"
        )
    return {
        "passed": len(errors) == 0,
        "sum_wt_pct": float(total),
        "deviation_wt_pct": float(deviation),
        "tolerance": float(tolerance),
        "errors": errors,
    }


def wt_pct_to_at_pct(wt_pct: dict[str, float]) -> dict[str, float]:
    """Convert weight percent (wt%) composition to atomic percent (at%)."""
    if not wt_pct:
        raise ValueError("Composition dictionary cannot be empty")
    moles: dict[str, float] = {}
    for element, pct in wt_pct.items():
        if element not in ATOMIC_WEIGHTS:
            raise ValueError(f"Unknown alloy element: {element!r}")
        val = float(pct)
        if not np.isfinite(val) or val < 0.0:
            raise ValueError(f"Negative or non-finite wt% for {element}: {val}")
        moles[element] = val / ATOMIC_WEIGHTS[element]
    total_moles = sum(moles.values())
    if total_moles <= 0.0:
        raise ValueError("Total composition must be strictly positive")
    return {el: 100.0 * mol / total_moles for el, mol in moles.items()}


def at_pct_to_wt_pct(at_pct: dict[str, float]) -> dict[str, float]:
    """Convert atomic percent (at%) composition to weight percent (wt%)."""
    if not at_pct:
        raise ValueError("Composition dictionary cannot be empty")
    masses: dict[str, float] = {}
    for element, pct in at_pct.items():
        if element not in ATOMIC_WEIGHTS:
            raise ValueError(f"Unknown alloy element: {element!r}")
        val = float(pct)
        if not np.isfinite(val) or val < 0.0:
            raise ValueError(f"Negative or non-finite at% for {element}: {val}")
        masses[element] = val * ATOMIC_WEIGHTS[element]
    total_mass = sum(masses.values())
    if total_mass <= 0.0:
        raise ValueError("Total composition must be strictly positive")
    return {el: 100.0 * mass / total_mass for el, mass in masses.items()}


def run_unit_verification_suite() -> dict[str, Any]:
    """Run deterministic Pint round-trips and alloy ``Sigma = 100%`` verification."""
    unit_cases = [
        (40.0, "mm", "in"),
        (150.0, "mm", "cm"),
        (300.0, "mm", "m"),
        (85.0, "mm", "um"),
        (120.0, "mm", "mil"),
        (450.0, "MPa", "ksi"),
        (210.0, "GPa", "MPa"),
        (355.0, "MPa", "psi"),
        (920.0, "degC", "K"),
        (250.0, "degC", "degF"),
    ]
    round_trips = [verify_round_trip(v, src, dst) for v, src, dst in unit_cases]

    reference_alloys: dict[str, dict[str, float]] = {
        "AISI_304": {"Fe": 70.12, "Cr": 19.0, "Ni": 9.25, "Mn": 1.0, "Si": 0.5, "C": 0.05, "N": 0.05, "P": 0.02, "S": 0.01},
        "AISI_4140": {"Fe": 97.10, "C": 0.40, "Mn": 0.85, "Cr": 0.95, "Mo": 0.20, "Si": 0.25, "Ni": 0.15, "P": 0.05, "S": 0.05},
        "API_5L_X70": {"Fe": 97.70, "C": 0.08, "Mn": 1.55, "Si": 0.30, "Nb": 0.05, "V": 0.05, "Ti": 0.02, "Cr": 0.10, "Ni": 0.10, "Mo": 0.05},
        "AISI_316L": {"Fe": 65.47, "Cr": 17.0, "Ni": 12.0, "Mo": 2.5, "Mn": 2.0, "Si": 0.9, "C": 0.03, "N": 0.06, "P": 0.02, "S": 0.02},
    }

    alloy_checks: dict[str, Any] = {}
    max_comp_rt_err = 0.0
    for grade, wt_comp in reference_alloys.items():
        balance = check_composition_mass_balance(wt_comp)
        at_comp = wt_pct_to_at_pct(wt_comp)
        wt_back = at_pct_to_wt_pct(at_comp)
        rt_err = max(abs(wt_back[el] - wt_comp[el]) for el in wt_comp)
        max_comp_rt_err = max(max_comp_rt_err, rt_err)
        alloy_checks[grade] = {
            "mass_balance_passed": balance["passed"],
            "sum_wt_pct": balance["sum_wt_pct"],
            "sum_at_pct": float(sum(at_comp.values())),
            "wt_at_wt_max_abs_err": float(rt_err),
        }

    invalid_alloy = check_composition_mass_balance({"Fe": 80.0, "Cr": 18.0, "Ni": 5.5})
    max_unit_err = max(item["abs_error"] for item in round_trips)
    all_passed = (
        all(item["passed"] for item in round_trips)
        and all(entry["mass_balance_passed"] for entry in alloy_checks.values())
        and max_comp_rt_err < 1e-12
        and (invalid_alloy["passed"] is False)
    )
    return {
        "passed": bool(all_passed),
        "n_unit_round_trips": len(round_trips),
        "max_unit_round_trip_abs_err": float(max_unit_err),
        "max_composition_round_trip_abs_err": float(max_comp_rt_err),
        "round_trips": round_trips,
        "alloy_checks": alloy_checks,
        "invalid_mass_balance_rejected": not invalid_alloy["passed"],
        "invalid_mass_balance_sum": invalid_alloy["sum_wt_pct"],
    }
