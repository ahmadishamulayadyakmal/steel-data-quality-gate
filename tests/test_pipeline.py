"""End-to-end synthetic quality-gate tests and pinned UCI D01 integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from steelgate.benchmark import reproduce
from steelgate.corruption import apply_quality_gate, corrupt_mill_dataset
from steelgate.loaders import UCI_D01_META, load_uci_faults, verify_d01_archive
from steelgate.schemas import QualityGateRejectionError, validate_or_reject, validate_uci_frame
from test_schemas import make_valid_uci_frame

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "raw"
CATALOG_PATH = ROOT_DIR / "data" / "catalog.yml"
D01_ZIP = DATA_DIR / UCI_D01_META["file"]
RAW_READY = D01_ZIP.is_file()


def test_synthetic_corruption_is_rejected_then_cleaned_to_100_percent():
    clean = make_valid_uci_frame(80)
    corrupted, manifest = corrupt_mill_dataset(clean, seed=42)
    assert manifest["synthetic"] is True

    with pytest.raises(QualityGateRejectionError):
        validate_or_reject(corrupted, strict_physical=True)

    gated, audit = apply_quality_gate(corrupted, strict_physical=True, clean_label_noise=True, seed=42)
    assert audit["post_gate_validation"]["passed"] is True
    assert audit["post_gate_validation"]["pass_rate"] == 1.0
    assert audit["n_quarantined_schema"] > 0
    assert len(gated) > 0


@pytest.mark.skipif(not RAW_READY, reason="D01 steel_plates_faults.zip not present in data/raw/")
def test_raw_d01_audit_catches_exact_luminosity_anomalies_and_curated_passes():
    verified = verify_d01_archive(D01_ZIP)
    assert verified["sha256"] == UCI_D01_META["sha256"]

    raw_df = load_uci_faults(D01_ZIP, verify_checksum=True)
    tier1 = validate_uci_frame(raw_df, strict_physical=False)
    assert tier1["passed"] is True
    assert tier1["rows_passing"] == 1941

    strict = validate_uci_frame(raw_df, strict_physical=True)
    assert strict["passed"] is False
    assert strict["failing_row_indices"] == [518, 591]

    curated = raw_df.loc[strict["_pass_mask"]].copy().reset_index(drop=True)
    curated_report = validate_or_reject(curated, strict_physical=True)
    assert curated_report["passed"] is True
    assert curated_report["rows_passing"] == 1939


@pytest.mark.skipif(not RAW_READY, reason="D01 steel_plates_faults.zip not present in data/raw/")
def test_reproduce_pipeline_is_deterministic_and_improves_accuracy(tmp_path):
    payload = reproduce(DATA_DIR, CATALOG_PATH, tmp_path, seed=42)
    assert payload["determinism_check"] is True
    assert payload["corrupt_copy_rejection"]["rejected"] is True
    assert payload["unit_verification"]["passed"] is True

    et = payload["before_after_benchmark"]["models"]["extra_trees"]
    assert et["after_gate_cleaned"]["zero_train_test_leakage_all_seeds"] is True
    assert et["delta_after_minus_before"]["accuracy_gain"] > 0.0
    assert et["delta_after_minus_before"]["macro_f1_gain"] > 0.0
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "provenance.json").is_file()
