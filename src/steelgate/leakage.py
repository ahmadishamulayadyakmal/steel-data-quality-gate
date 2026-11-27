"""Duplicate detection, near-duplicate similarity, and train/test leakage audits."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from steelgate.loaders import UCI_ALL_COLUMNS, UCI_FAULT_COLUMNS, UCI_FEATURE_COLUMNS


def _row_hashes(df: pd.DataFrame, cols: tuple[str, ...]) -> list[str]:
    hashes: list[str] = []
    sub = df.loc[:, [c for c in cols if c in df.columns]]
    for row in sub.itertuples(index=False, name=None):
        joined = "|".join(f"{val!r}" for val in row)
        hashes.append(hashlib.sha256(joined.encode("utf-8")).hexdigest())
    return hashes


def detect_duplicates(
    df: pd.DataFrame,
    *,
    feature_cols: tuple[str, ...] = UCI_FEATURE_COLUMNS,
    label_cols: tuple[str, ...] = UCI_FAULT_COLUMNS,
) -> dict[str, Any]:
    """Detect exact row duplicates, feature duplicates, and label conflicts."""
    avail_feat = [c for c in feature_cols if c in df.columns]
    avail_lbl = [c for c in label_cols if c in df.columns]
    avail_all = [c for c in UCI_ALL_COLUMNS if c in df.columns]

    exact_row_mask = df.duplicated(subset=avail_all, keep="first") if avail_all else pd.Series(False, index=df.index)
    exact_feat_mask = df.duplicated(subset=avail_feat, keep="first") if avail_feat else pd.Series(False, index=df.index)

    conflict_indices: list[Any] = []
    if avail_feat and avail_lbl:
        feat_hashes = pd.Series(_row_hashes(df, tuple(avail_feat)), index=df.index)
        lbl_hashes = pd.Series(_row_hashes(df, tuple(avail_lbl)), index=df.index)
        grouped = lbl_hashes.groupby(feat_hashes).nunique()
        conflicting_keys = set(grouped[grouped > 1].index.tolist())
        if conflicting_keys:
            conflict_indices = df.index[feat_hashes.isin(conflicting_keys)].tolist()

    return {
        "n_rows": int(len(df)),
        "exact_row_duplicates": int(exact_row_mask.sum()),
        "exact_feature_duplicates": int(exact_feat_mask.sum()),
        "label_conflict_rows": int(len(conflict_indices)),
        "exact_duplicate_indices": [
            int(i) if isinstance(i, (int, np.integer)) else str(i)
            for i in df.index[exact_row_mask].tolist()[:25]
        ],
        "conflict_row_indices": [
            int(i) if isinstance(i, (int, np.integer)) else str(i)
            for i in conflict_indices[:25]
        ],
    }


def detect_near_duplicates(
    df: pd.DataFrame,
    *,
    feature_cols: tuple[str, ...] = UCI_FEATURE_COLUMNS,
    threshold: float = 0.10,
) -> dict[str, Any]:
    """Scan standardized feature space for near-duplicate rows and Y_Minimum collisions."""
    avail_feat = [c for c in feature_cols if c in df.columns]
    numeric = df.loc[:, avail_feat].apply(pd.to_numeric, errors="coerce")
    valid_mask = np.isfinite(numeric.to_numpy()).all(axis=1)
    clean_num = numeric.loc[valid_mask]

    if len(clean_num) < 2:
        return {
            "n_rows_scanned": int(len(clean_num)),
            "threshold": float(threshold),
            "near_duplicate_rows": 0,
            "min_nn_distance": None,
            "percentiles": {},
            "y_minimum_collisions": 0,
        }

    scaled = StandardScaler().fit_transform(clean_num.to_numpy(dtype="float64"))
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(scaled)
    dists, idxs = nn.kneighbors(scaled)
    nn_dist = dists[:, 1]

    near_mask = nn_dist < threshold
    y_collisions = int(df["Y_Minimum"].duplicated().sum()) if "Y_Minimum" in df.columns else 0

    order = np.argsort(nn_dist)[:5]
    closest_pairs = [
        {
            "row_a": int(clean_num.index[i]),
            "row_b": int(clean_num.index[idxs[i, 1]]),
            "distance": float(nn_dist[i]),
        }
        for i in order
    ]
    pcts = np.percentile(nn_dist, [0, 1, 5, 10, 50])

    return {
        "n_rows_scanned": int(len(clean_num)),
        "threshold": float(threshold),
        "near_duplicate_rows": int(near_mask.sum()),
        "min_nn_distance": float(nn_dist.min()),
        "percentiles": {
            "p0": float(pcts[0]),
            "p1": float(pcts[1]),
            "p5": float(pcts[2]),
            "p10": float(pcts[3]),
            "p50": float(pcts[4]),
        },
        "y_minimum_collisions": y_collisions,
        "closest_pairs": closest_pairs,
    }


def assign_mill_campaign_groups(df: pd.DataFrame) -> pd.Series:
    """Group plates by mill campaign proxy ``(Length_of_Conveyer, Steel_Plate_Thickness, TypeOfSteel_A300)``."""
    required = ("Length_of_Conveyer", "Steel_Plate_Thickness", "TypeOfSteel_A300")
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing column required for campaign grouping: {col}")
    return (
        df["Length_of_Conveyer"].astype(int).astype(str)
        + "_"
        + df["Steel_Plate_Thickness"].round(3).astype(str)
        + "_"
        + df["TypeOfSteel_A300"].astype(int).astype(str)
    )


def audit_split_leakage(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    feature_cols: tuple[str, ...] = UCI_FEATURE_COLUMNS,
    near_dup_threshold: float = 0.10,
) -> dict[str, Any]:
    """Audit train vs. test splits for exact duplicate overlap and campaign-group overlap."""
    avail_feat = tuple(c for c in feature_cols if c in df_train.columns and c in df_test.columns)
    train_hashes = set(_row_hashes(df_train, avail_feat))
    test_hashes = _row_hashes(df_test, avail_feat)
    exact_overlap = sum(1 for h in test_hashes if h in train_hashes)

    # Cross-split near-duplicate distance
    tr_num = df_train.loc[:, list(avail_feat)].apply(pd.to_numeric, errors="coerce")
    te_num = df_test.loc[:, list(avail_feat)].apply(pd.to_numeric, errors="coerce")
    tr_valid = tr_num.loc[np.isfinite(tr_num.to_numpy()).all(axis=1)]
    te_valid = te_num.loc[np.isfinite(te_num.to_numpy()).all(axis=1)]
    cross_near_dups = 0
    if len(tr_valid) >= 1 and len(te_valid) >= 1:
        scaler = StandardScaler().fit(tr_valid.to_numpy(dtype="float64"))
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(
            scaler.transform(tr_valid.to_numpy(dtype="float64"))
        )
        dists, _ = nn.kneighbors(scaler.transform(te_valid.to_numpy(dtype="float64")))
        cross_near_dups = int((dists[:, 0] < near_dup_threshold).sum())

    campaign_overlap_frac = None
    if all(c in df_train.columns and c in df_test.columns for c in ("Length_of_Conveyer", "Steel_Plate_Thickness", "TypeOfSteel_A300")):
        tr_groups = set(assign_mill_campaign_groups(df_train).tolist())
        te_groups = assign_mill_campaign_groups(df_test)
        campaign_overlap_frac = float(te_groups.isin(tr_groups).mean())

    return {
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "exact_feature_overlap_count": int(exact_overlap),
        "cross_split_near_duplicates": int(cross_near_dups),
        "near_dup_threshold": float(near_dup_threshold),
        "campaign_overlap_fraction": campaign_overlap_frac,
        "zero_exact_leakage": exact_overlap == 0,
    }
