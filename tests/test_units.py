"""Unit round-trip, multi-unit normalization, formula parser, and alloy balance tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from steelgate.units import (
    at_pct_to_wt_pct,
    check_composition_mass_balance,
    convert_scalar,
    normalize_series_units,
    normalize_thickness_series,
    parse_formula_to_at_pct,
    run_unit_verification_suite,
    verify_round_trip,
    wt_pct_to_at_pct,
)


@pytest.mark.parametrize(
    "value,src,dst",
    [
        (40.0, "mm", "in"),
        (125.0, "mm", "cm"),
        (300.0, "mm", "um"),
        (450.0, "MPa", "ksi"),
        (210.0, "GPa", "MPa"),
        (920.0, "degC", "K"),
        (250.0, "degC", "degF"),
        (42.5, "kWh", "Wh"),
        (125.0, "kWh", "MWh"),
        (75.0, "kWh", "MJ"),
    ],
)
def test_pint_unit_round_trips_within_machine_precision(value, src, dst):
    result = verify_round_trip(value, src, dst, rtol=1e-12, atol=1e-12)
    assert result["passed"] is True
    assert result["abs_error"] < 1e-11


def test_normalize_series_units_thickness_stress_and_energy():
    values = pd.Series([100.0, 2.0, 15.0, 50000.0, np.nan])
    units = pd.Series(["mm", "in", "cm", "um", "mm"])
    normalized, counts = normalize_thickness_series(values, units, target_unit="mm")

    assert counts == {"mm": 2, "in": 1, "cm": 1, "um": 1}
    assert normalized.iloc[0] == pytest.approx(100.0)
    assert normalized.iloc[1] == pytest.approx(50.8)
    assert normalized.iloc[2] == pytest.approx(150.0)
    assert normalized.iloc[3] == pytest.approx(50.0)
    assert np.isnan(normalized.iloc[4])

    e_vals, _ = normalize_series_units(pd.Series([1000.0, 0.05]), pd.Series(["Wh", "MWh"]), target_unit="kWh")
    assert e_vals.iloc[0] == pytest.approx(1.0)
    assert e_vals.iloc[1] == pytest.approx(50.0)


def test_normalize_thickness_series_rejects_unknown_unit():
    with pytest.raises(ValueError, match="Unsupported"):
        normalize_thickness_series(pd.Series([10.0]), pd.Series(["furlong"]))


def test_parse_formula_and_alloy_sigma_100():
    at_comp, rsum = parse_formula_to_at_pct("Fe0.80Cr0.18C0.02")
    assert rsum == pytest.approx(1.0)
    assert at_comp["Fe"] == pytest.approx(80.0)

    with pytest.raises(ValueError, match="Unknown alloy element"):
        parse_formula_to_at_pct("Fe0.9Xz0.1")

    grade_304 = {"Fe": 70.12, "Cr": 19.0, "Ni": 9.25, "Mn": 1.0, "Si": 0.5, "C": 0.05, "N": 0.05, "P": 0.02, "S": 0.01}
    balance = check_composition_mass_balance(grade_304)
    assert balance["passed"] is True

    at_pct = wt_pct_to_at_pct(grade_304)
    assert sum(at_pct.values()) == pytest.approx(100.0)
    wt_back = at_pct_to_wt_pct(at_pct)
    for el, orig in grade_304.items():
        assert wt_back[el] == pytest.approx(orig, abs=1e-12)

    bad = check_composition_mass_balance({"Fe": 80.0, "Cr": 18.0, "Ni": 5.5})
    assert bad["passed"] is False
    assert convert_scalar(1.0, "inch", "mm") == pytest.approx(25.4)


def test_full_unit_verification_suite_passes():
    suite = run_unit_verification_suite()
    assert suite["passed"] is True
    assert suite["n_unit_round_trips"] == 13
    assert suite["invalid_mass_balance_rejected"] is True
