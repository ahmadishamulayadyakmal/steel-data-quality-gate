"""Tests for duplicate detection, near-duplicate scans, and train/test leakage audits."""

from __future__ import annotations

import pandas as pd

from steelgate.leakage import (
    assign_mill_campaign_groups,
    audit_split_leakage,
    detect_duplicates,
    detect_near_duplicates,
)
from test_schemas import make_valid_uci_frame


def test_detect_duplicates_and_label_conflicts():
    df = make_valid_uci_frame(6)
    dup_exact = df.iloc[[0]].copy()
    dup_conflict = df.iloc[[1]].copy()
    dup_conflict["Pastry"] = 0
    dup_conflict["Z_Scratch"] = 0
    dup_conflict["K_Scatch"] = 0
    dup_conflict["Stains"] = 0
    dup_conflict["Dirtiness"] = 0
    dup_conflict["Bumps"] = 0
    dup_conflict["Other_Faults"] = 1
    combined = pd.concat([df, dup_exact, dup_conflict], ignore_index=True)

    dups = detect_duplicates(combined)
    assert dups["exact_row_duplicates"] == 1
    assert dups["exact_feature_duplicates"] == 2
    assert dups["label_conflict_rows"] == 2


def test_near_duplicate_and_split_leakage_audit():
    df = make_valid_uci_frame(10)
    train = df.iloc[:7].copy()
    test = df.iloc[7:].copy()

    clean_audit = audit_split_leakage(train, test)
    assert clean_audit["zero_exact_leakage"] is True
    assert clean_audit["exact_feature_overlap_count"] == 0

    leaked_test = pd.concat([test, train.iloc[[0]]], ignore_index=True)
    leaky_audit = audit_split_leakage(train, leaked_test)
    assert leaky_audit["zero_exact_leakage"] is False
    assert leaky_audit["exact_feature_overlap_count"] == 1

    near = detect_near_duplicates(pd.concat([df, df.iloc[[0]]], ignore_index=True))
    assert near["near_duplicate_rows"] >= 2
    groups = assign_mill_campaign_groups(df)
    assert len(groups) == 10
