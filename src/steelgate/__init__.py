"""Case 02: Schema-enforced data quality gate for open steel inspection datasets."""

from __future__ import annotations

__version__ = "0.4.0"

from steelgate.leakage import (
    assign_mill_campaign_groups,
    audit_split_leakage,
    detect_duplicates,
    detect_near_duplicates,
)
from steelgate.loaders import (
    UCI_ALL_COLUMNS,
    UCI_D01_META,
    UCI_FAULT_COLUMNS,
    UCI_FEATURE_COLUMNS,
    load_catalog,
    load_neu_det_manifest,
    load_uci_faults,
    verify_d01_archive,
)
from steelgate.schemas import (
    QualityGateRejectionError,
    build_neu_det_manifest_schema,
    build_uci_schema,
    validate_neu_det_manifest,
    validate_or_reject,
    validate_uci_frame,
)
from steelgate.units import (
    ATOMIC_WEIGHTS,
    at_pct_to_wt_pct,
    check_composition_mass_balance,
    convert_scalar,
    normalize_thickness_series,
    run_unit_verification_suite,
    verify_round_trip,
    wt_pct_to_at_pct,
)

__all__ = [
    "__version__",
    "ATOMIC_WEIGHTS",
    "QualityGateRejectionError",
    "UCI_ALL_COLUMNS",
    "UCI_D01_META",
    "UCI_FAULT_COLUMNS",
    "UCI_FEATURE_COLUMNS",
    "assign_mill_campaign_groups",
    "at_pct_to_wt_pct",
    "audit_split_leakage",
    "build_neu_det_manifest_schema",
    "build_uci_schema",
    "check_composition_mass_balance",
    "convert_scalar",
    "detect_duplicates",
    "detect_near_duplicates",
    "load_catalog",
    "load_neu_det_manifest",
    "load_uci_faults",
    "normalize_thickness_series",
    "run_unit_verification_suite",
    "validate_neu_det_manifest",
    "validate_or_reject",
    "validate_uci_frame",
    "verify_d01_archive",
    "verify_round_trip",
    "wt_pct_to_at_pct",
]
