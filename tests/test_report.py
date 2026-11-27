"""Tests for Case 02 SVG figures, HTML quality report, and CLI commands."""

from __future__ import annotations

import json

from steelgate.cli import main
from steelgate.report import (
    build_before_after_svg,
    build_diagnostics_svg,
    build_html_quality_report,
    build_scorecard_svg,
)
from test_schemas import make_valid_uci_frame


def _sample_metrics_and_provenance() -> tuple[dict, dict]:
    classes = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
    per_cls_b = {c: {"mean": 0.60, "std": 0.02, "values": [0.60]} for c in classes}
    per_cls_a = {c: {"mean": 0.78, "std": 0.02, "values": [0.78]} for c in classes}
    cm = [[0.80 if i == j else 0.033 for j in range(7)] for i in range(7)]
    rules = [
        {
            "rule_id": f"R{i:02d}",
            "tier": "Tier 1",
            "category": "Domain & Nulls",
            "name": f"Rule {i}",
            "description": f"Invariant {i}",
            "n_failing": 0,
            "pass_rate": 1.0,
            "passed": True,
            "sample_failing_rows": [],
        }
        for i in range(1, 39)
    ]
    metrics = {
        "case": "C02",
        "determinism_check": True,
        "data": {"D01": {"sha256": "cb8eb9859198b63f053e443513036b401746fa517ef58bd17c846c6741c93919"}, "D02_catalog_status": "mirror_licence_unverified"},
        "raw_d01_audit": {
            "tier1_structural_math": {"rows_passing": 1941, "rows_checked": 1941},
            "strict_physical_raw": {"rows_passing": 1939, "rows_checked": 1941},
            "curated_strict_physical": {"rows_passing": 1939, "rows_checked": 1939, "total_rules": 38, "rule_results": rules},
            "real_raw_anomalies": [{"row_index": 518, "pixels_areas": 5357, "sum_of_luminosity": 764, "minimum_of_Luminosity": 123, "minimum_of_luminosity": 123, "expected_min_sum": 658911, "mean_luminosity": 0.14}],
            "duplicates": {"exact_row_duplicates": 0},
            "near_duplicates": {"near_duplicate_rows": 2, "y_minimum_collisions": 2},
            "n_mill_campaign_groups": 239,
            "luminosity_audit_points": [{"min_lum": 100.0, "mean_lum": 125.0, "valid": True}, {"row_index": 518, "min_lum": 123.0, "mean_lum": 0.14, "valid": False}],
        },
        "unit_verification": {
            "passed": True,
            "max_unit_round_trip_abs_err": 1e-14,
            "max_composition_round_trip_abs_err": 1e-14,
            "round_trips": [{"original": 40.0, "from_unit": "mm", "to_unit": "in", "converted": 1.5748, "abs_error": 0.0, "passed": True}],
        },
        "corrupt_copy_rejection": {
            "rejected": True,
            "rejection_report": {"rows_failing": 250, "rows_checked": 2133, "rules_failed": 18, "error_report": "Status: REJECTED"},
            "category_failure_counts": {"Domain & Nulls": 120, "Geometry Physics": 45, "Label Integrity": 35},
            "thickness_distributions": {"raw_corrupted_values": [2.0, 80.0, 50000.0], "normalized_mm_values": [50.8, 80.0, 50.0]},
        },
        "campaign_leakage_cv": {
            "n_campaign_groups": 239,
            "stratified_kfold": {"accuracy": {"mean": 0.80, "std": 0.01}, "macro_f1": {"mean": 0.81, "std": 0.01}},
            "campaign_grouped_kfold": {"accuracy": {"mean": 0.68, "std": 0.03}, "macro_f1": {"mean": 0.69, "std": 0.03}},
            "accuracy_inflation_gap": 0.12,
        },
        "before_after_benchmark": {
            "class_names": classes,
            "models": {
                "extra_trees": {
                    "leaky_internal_qa": {"accuracy": {"mean": 0.82, "std": 0.01}, "macro_f1": {"mean": 0.80, "std": 0.01}, "exact_train_test_overlap": {"mean": 30.0}},
                    "before_gate_uncleaned": {"accuracy": {"mean": 0.65, "std": 0.02}, "macro_f1": {"mean": 0.62, "std": 0.02}, "per_class_f1": per_cls_b},
                    "after_gate_cleaned": {"accuracy": {"mean": 0.78, "std": 0.01}, "macro_f1": {"mean": 0.79, "std": 0.01}, "per_class_f1": per_cls_a, "zero_train_test_leakage_all_seeds": True, "confusion_matrix_normalized": cm},
                    "clean_oracle_reference": {"accuracy": {"mean": 0.80, "std": 0.01}, "macro_f1": {"mean": 0.81, "std": 0.01}},
                    "delta_after_minus_before": {"accuracy_gain_pp": 13.0, "macro_f1_gain_pp": 17.0},
                },
                "random_forest": {
                    "leaky_internal_qa": {"accuracy": {"mean": 0.80, "std": 0.01}, "macro_f1": {"mean": 0.78, "std": 0.01}},
                    "before_gate_uncleaned": {"accuracy": {"mean": 0.63, "std": 0.02}, "macro_f1": {"mean": 0.60, "std": 0.02}},
                    "after_gate_cleaned": {"accuracy": {"mean": 0.76, "std": 0.01}, "macro_f1": {"mean": 0.77, "std": 0.01}},
                    "clean_oracle_reference": {"accuracy": {"mean": 0.78, "std": 0.01}, "macro_f1": {"mean": 0.79, "std": 0.01}},
                    "delta_after_minus_before": {"accuracy_gain_pp": 13.0, "macro_f1_gain_pp": 17.0},
                },
            },
        },
    }
    provenance = {"code_fingerprint": "a" * 64}
    return metrics, provenance


def test_svg_and_html_generators_produce_valid_deterministic_outputs():
    metrics, prov = _sample_metrics_and_provenance()
    s1 = build_scorecard_svg(metrics)
    s2 = build_before_after_svg(metrics)
    s3 = build_diagnostics_svg(metrics)
    html = build_html_quality_report(metrics, prov)

    for svg in (s1, s2, s3):
        assert svg.startswith("<svg ")
        assert svg.endswith("</svg>")
        assert "synthetic" in svg.lower()
    assert "<!DOCTYPE html>" in html
    assert "synthetic" in html.lower()
    assert build_scorecard_svg(metrics) == s1


def test_cli_info_validate_and_report(tmp_path, capsys):
    assert main(["info"]) == 0
    assert "steelgate" in capsys.readouterr().out

    csv_path = tmp_path / "valid.csv"
    make_valid_uci_frame(6).to_csv(csv_path, index=False)
    assert main(["validate", "--input", str(csv_path), "--strict-physical"]) == 0

    bad_df = make_valid_uci_frame(6)
    bad_df.loc[0, "X_Maximum"] = 1
    bad_csv = tmp_path / "bad.csv"
    bad_df.to_csv(bad_csv, index=False)
    assert main(["validate", "--input", str(bad_csv), "--strict-physical"]) == 1

    metrics, prov = _sample_metrics_and_provenance()
    m_path = tmp_path / "metrics.json"
    p_path = tmp_path / "provenance.json"
    m_path.write_text(json.dumps(metrics), encoding="utf-8")
    p_path.write_text(json.dumps(prov), encoding="utf-8")
    assert main(["report", "--metrics", str(m_path), "--provenance", str(p_path), "--figures-dir", str(tmp_path / "figs"), "--html-output", str(tmp_path / "report.html")]) == 0
    assert (tmp_path / "figs" / "c02_scorecard.svg").is_file()
    assert (tmp_path / "report.html").is_file()
