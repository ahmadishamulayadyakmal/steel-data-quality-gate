"""End-to-end synthetic gate tests and 3-database (D01, D04, D09) integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from steelgate.benchmark import reproduce
from steelgate.corruption import (
    apply_d04_quality_gate,
    apply_d09_quality_gate,
    apply_quality_gate,
    corrupt_d04_dataset,
    corrupt_d09_dataset,
    corrupt_mill_dataset,
)
from steelgate.loaders import (
    D04_SPLITS_META,
    D04_STEELS_META,
    D09_ENERGY_META,
    UCI_D01_META,
    load_matbench_steels,
    load_steel_energy,
    load_uci_faults,
    verify_dataset_file,
)
from steelgate.schemas import (
    QualityGateRejectionError,
    validate_d04_frame,
    validate_d04_or_reject,
    validate_d09_or_reject,
    validate_or_reject,
    validate_uci_frame,
)
from test_schemas import make_valid_d04_frame, make_valid_d09_frame, make_valid_uci_frame

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "raw"
CATALOG_PATH = ROOT_DIR / "data" / "catalog.yml"
D01_ZIP = DATA_DIR / UCI_D01_META["file"]
D04_GZ = DATA_DIR / D04_STEELS_META["file"]
D04_SPLITS = DATA_DIR / D04_SPLITS_META["file"]
D09_ZIP = DATA_DIR / D09_ENERGY_META["file"]

ALL_RAW_READY = all(p.is_file() for p in (D01_ZIP, D04_GZ, D04_SPLITS, D09_ZIP))


def test_synthetic_d01_d04_d09_corruptions_are_rejected_and_cleaned():
    # D01
    c_d01, m1 = corrupt_mill_dataset(make_valid_uci_frame(80), seed=42)
    assert m1["synthetic"] is True
    with pytest.raises(QualityGateRejectionError):
        validate_or_reject(c_d01, strict_physical=True)
    g1, a1 = apply_quality_gate(c_d01, strict_physical=True, clean_label_noise=True, seed=42)
    assert a1["post_gate_validation"]["passed"] is True
    assert len(g1) > 0

    # D04
    c_d04, m4 = corrupt_d04_dataset(make_valid_d04_frame(40), seed=42)
    assert m4["synthetic"] is True
    with pytest.raises(QualityGateRejectionError):
        validate_d04_or_reject(c_d04, strict_physical=True)
    g4, a4 = apply_d04_quality_gate(c_d04, clean_outliers=True, seed=42)
    assert a4["post_gate_validation"]["passed"] is True
    assert len(g4) > 0

    # D09
    c_d09, m9 = corrupt_d09_dataset(make_valid_d09_frame(40), seed=42)
    assert m9["synthetic"] is True
    with pytest.raises(QualityGateRejectionError):
        validate_d09_or_reject(c_d09, strict_physical=True)
    g9, a9 = apply_d09_quality_gate(c_d09, strict_physical=True)
    assert a9["post_gate_validation"]["passed"] is True
    assert len(g9) > 0


@pytest.mark.skipif(not ALL_RAW_READY, reason="D01/D04/D09 raw archives not present in data/raw/")
def test_raw_three_databases_integrity_and_real_anomalies():
    assert verify_dataset_file(D01_ZIP, UCI_D01_META)["sha256"] == UCI_D01_META["sha256"]
    assert verify_dataset_file(D04_GZ, D04_STEELS_META)["sha256"] == D04_STEELS_META["sha256"]
    assert verify_dataset_file(D09_ZIP, D09_ENERGY_META)["sha256"] == D09_ENERGY_META["sha256"]

    raw_d01 = load_uci_faults(D01_ZIP, verify_checksum=True)
    assert validate_uci_frame(raw_d01, strict_physical=False)["rows_passing"] == 1941
    strict_d01 = validate_uci_frame(raw_d01, strict_physical=True)
    assert strict_d01["failing_row_indices"] == [518, 591]

    raw_d04 = load_matbench_steels(D04_GZ, verify_checksum=True)
    assert validate_d04_frame(raw_d04, strict_physical=False)["rows_passing"] == 312
    strict_d04 = validate_d04_frame(raw_d04, strict_physical=True)
    assert strict_d04["failing_row_indices"] == [21, 28, 36, 90, 91, 103]

    raw_d09 = load_steel_energy(D09_ZIP, verify_checksum=True)
    strict_d09 = validate_d09_or_reject(raw_d09, strict_physical=True)
    assert strict_d09["rows_passing"] == 35040


@pytest.mark.skipif(not ALL_RAW_READY, reason="D01/D04/D09 raw archives not present in data/raw/")
def test_reproduce_three_database_pipeline_is_deterministic_and_recovers_all_tasks(tmp_path):
    payload = reproduce(DATA_DIR, CATALOG_PATH, tmp_path, seed=42)
    assert payload["determinism_check"] is True
    assert payload["total_real_records"] == 37293
    assert payload["total_schema_rules"] == 64

    ens = payload["before_after_benchmark"]["models"]["voting_ensemble"]
    et = payload["before_after_benchmark"]["models"]["extra_trees"]
    assert ens["after_gate_cleaned"]["zero_train_test_leakage_all_seeds"] is True
    assert ens["after_gate_cleaned"]["accuracy"]["mean"] > et["after_gate_cleaned"]["accuracy"]["mean"]
    assert et["delta_after_minus_before"]["accuracy_gain"] > 0.0

    d04_b = payload["d04_benchmark"]["five_fold_benchmark"]
    assert d04_b["zero_fold_id_leakage"] is True
    assert d04_b["mae_reduction_pct"] > 70.0

    d09_b = payload["d09_benchmark"]["chronological_before_after"]
    assert d09_b["zero_timestamp_leakage"] is True
    assert d09_b["mae_reduction_pct"] > 50.0
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "provenance.json").is_file()
