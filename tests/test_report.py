"""Tests for Case 02 v0.7.0 multi-database SVG figures, HTML report, and CLI."""

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
    per_cls_a = {c: {"mean": 0.79, "std": 0.02, "values": [0.79]} for c in classes}
    cm = [[0.80 if i == j else 0.033 for j in range(7)] for i in range(7)]

    def _rules(prefix: str, count: int) -> list[dict]:
        return [
            {
                "rule_id": f"{prefix}{i:02d}",
                "tier": "Tier 1",
                "category": "Domain & Nulls",
                "name": f"Rule {prefix}{i}",
                "description": f"Invariant {prefix}{i}",
                "n_failing": 0,
                "pass_rate": 1.0,
                "passed": True,
                "sample_failing_rows": [],
            }
            for i in range(1, count + 1)
        ]

    ablation_stages = [
        "Before Gate (ET)",
        "+S1 Pint Units",
        "+S2 Repair & 38 Rules",
        "+S3 Dedup",
        "+S4 Confident Learning",
        "+S5 ET+HGB Ensemble",
        "Clean Oracle (ET)",
        "Clean Oracle (ET+HGB)",
    ]
    ablation = [
        {"stage": s, "accuracy": {"mean": 0.74 + 0.008 * i, "std": 0.01}, "macro_f1": {"mean": 0.76 + 0.007 * i, "std": 0.01}}
        for i, s in enumerate(ablation_stages)
    ]

    metrics = {
        "case": "C02",
        "version": "0.7.0",
        "determinism_check": True,
        "total_real_records": 37293,
        "total_schema_rules": 64,
        "data": {"D01": {"sha256": "cb8e"}, "D04": {"sha256": "473b"}, "D09": {"sha256": "d82d"}, "D02_catalog_status": "mirror_licence_unverified"},
        "raw_d01_audit": {
            "tier1_structural_math": {"rows_passing": 1941, "rows_checked": 1941},
            "strict_physical_raw": {"rows_passing": 1939, "rows_checked": 1941},
            "curated_strict_physical": {"rows_passing": 1939, "rows_checked": 1939, "total_rules": 38, "rule_results": _rules("R", 38)},
            "real_raw_anomalies": [{"row_index": 518, "pixels_areas": 5357, "sum_of_luminosity": 764, "minimum_of_luminosity": 123, "expected_min_sum": 658911, "mean_luminosity": 0.14}],
            "luminosity_audit_points": [{"min_lum": 100.0, "mean_lum": 125.0, "valid": True}, {"row_index": 518, "min_lum": 123.0, "mean_lum": 0.14, "valid": False}],
        },
        "unit_verification": {
            "passed": True,
            "n_unit_round_trips": 13,
            "max_unit_round_trip_abs_err": 1e-13,
            "max_composition_round_trip_abs_err": 1e-14,
            "round_trips": [{"original": 40.0, "from_unit": "mm", "to_unit": "in", "converted": 1.5748, "abs_error": 0.0, "passed": True}],
        },
        "corrupt_copy_rejection": {
            "rejected": True,
            "rejection_report": {"rows_failing": 2133, "rows_checked": 2133, "rules_failed": 21, "error_report": "Status: REJECTED"},
        },
        "campaign_leakage_cv": {
            "n_campaign_groups": 239,
            "stratified_kfold": {"accuracy": {"mean": 0.78, "std": 0.01}, "macro_f1": {"mean": 0.80, "std": 0.01}},
            "campaign_grouped_kfold": {"accuracy": {"mean": 0.73, "std": 0.05}, "macro_f1": {"mean": 0.71, "std": 0.07}},
            "accuracy_inflation_gap": 0.05,
        },
        "before_after_benchmark": {
            "class_names": classes,
            "ablation_ladder": ablation,
            "models": {
                "extra_trees": {
                    "before_gate_uncleaned": {"accuracy": {"mean": 0.747, "std": 0.01}, "macro_f1": {"mean": 0.769, "std": 0.02}, "per_class_f1": per_cls_b},
                    "after_gate_cleaned": {"accuracy": {"mean": 0.771, "std": 0.01}, "macro_f1": {"mean": 0.784, "std": 0.02}, "per_class_f1": per_cls_a, "zero_train_test_leakage_all_seeds": True, "confusion_matrix_normalized": cm},
                },
                "voting_ensemble": {
                    "before_gate_uncleaned": {"accuracy": {"mean": 0.772, "std": 0.01}, "macro_f1": {"mean": 0.783, "std": 0.01}, "per_class_f1": per_cls_b},
                    "after_gate_cleaned": {"accuracy": {"mean": 0.786, "std": 0.01}, "macro_f1": {"mean": 0.792, "std": 0.01}, "per_class_f1": per_cls_a, "zero_train_test_leakage_all_seeds": True, "confusion_matrix_normalized": cm},
                },
            },
        },
        "d04_benchmark": {
            "strict_m01_m12": {"rule_results": _rules("M", 12)},
            "real_raw_stoich_anomalies": [{"row_index": 21}],
            "raw_formula_sums": [{"row_index": 0, "raw_fraction_sum": 1.0, "flagged_m12": False}, {"row_index": 21, "raw_fraction_sum": 1.0011, "flagged_m12": True}],
            "five_fold_benchmark": {
                "before_gate": {"mae_mpa": {"mean": 462.55, "std": 41.93}},
                "after_gate": {"mae_mpa": {"mean": 81.90, "std": 12.35}},
                "clean_oracle": {"mae_mpa": {"mean": 79.09, "std": 8.54}},
                "mae_reduction_pct": 82.3,
            },
        },
        "d09_benchmark": {
            "strict_e01_e14": {"rule_results": _rules("E", 14)},
            "temporal_leakage": {"random_80_20_mae_kwh": 2.1, "chronological_80_20_mae_kwh": 3.4, "lookahead_mae_inflation_kwh": 1.3},
            "chronological_before_after": {
                "before_gate": {"mae_kwh": 120.0},
                "after_gate": {"mae_kwh": 3.5},
                "clean_oracle": {"mae_kwh": 3.4},
                "mae_reduction_pct": 97.1,
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
    assert "37,293" in html
    assert build_scorecard_svg(metrics) == s1


def test_cli_info_validate_and_report(tmp_path, capsys):
    assert main(["info"]) == 0
    assert "steelgate 0.7.0" in capsys.readouterr().out

    csv_path = tmp_path / "valid.csv"
    make_valid_uci_frame(6).to_csv(csv_path, index=False)
    assert main(["validate", "--input", str(csv_path), "--dataset", "d01", "--strict-physical"]) == 0

    bad_df = make_valid_uci_frame(6)
    bad_df.loc[0, "X_Maximum"] = 1
    bad_csv = tmp_path / "bad.csv"
    bad_df.to_csv(bad_csv, index=False)
    assert main(["validate", "--input", str(bad_csv), "--dataset", "d01", "--strict-physical"]) == 1

    metrics, prov = _sample_metrics_and_provenance()
    m_path = tmp_path / "metrics.json"
    p_path = tmp_path / "provenance.json"
    m_path.write_text(json.dumps(metrics), encoding="utf-8")
    p_path.write_text(json.dumps(prov), encoding="utf-8")
    assert main(["report", "--metrics", str(m_path), "--provenance", str(p_path), "--figures-dir", str(tmp_path / "figs"), "--html-output", str(tmp_path / "report.html")]) == 0
    assert (tmp_path / "figs" / "c02_scorecard.svg").is_file()
    assert (tmp_path / "report.html").is_file()
