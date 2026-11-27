"""Case 02 reproducible benchmark, raw D01 audit, R09 leakage CV, and before/after study."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from steelgate.corruption import apply_quality_gate, corrupt_mill_dataset
from steelgate.leakage import (
    assign_mill_campaign_groups,
    audit_split_leakage,
    detect_duplicates,
    detect_near_duplicates,
)
from steelgate.loaders import (
    UCI_FAULT_COLUMNS,
    UCI_FEATURE_COLUMNS,
    load_catalog,
    load_uci_faults,
    verify_d01_archive,
)
from steelgate.provenance import environment_info, source_fingerprint, write_json
from steelgate.schemas import (
    QualityGateRejectionError,
    serialize_validation_report,
    validate_or_reject,
    validate_uci_frame,
)
from steelgate.units import run_unit_verification_suite

SEEDS: tuple[int, ...] = (42, 7, 11, 19, 23)
CLASS_NAMES: list[str] = list(UCI_FAULT_COLUMNS)


def _stat_summary(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype="float64")
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "values": [float(v) for v in arr],
    }


def _make_classifier(model_name: str, seed: int) -> Any:
    if model_name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=300, random_state=seed, n_jobs=1)
    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1)
    raise ValueError(f"Unsupported classifier: {model_name!r}")


def _extract_naive_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    feat = df[list(UCI_FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    lbl = df[list(UCI_FAULT_COLUMNS)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return feat.to_numpy(dtype="float64"), np.argmax(lbl.to_numpy(dtype="float64"), axis=1)


def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    per_class = f1_score(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))), average=None, zero_division=0
    )
    cm = confusion_matrix(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))), normalize="true"
    )
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": {name: float(val) for name, val in zip(CLASS_NAMES, per_class)},
        "confusion_matrix_normalized": [[float(x) for x in row] for row in cm],
    }


def _error_modality_bins(y_true: np.ndarray, probs: np.ndarray) -> list[dict[str, Any]]:
    """Compute Y2R12-style accuracy vs. predicted confidence bands."""
    pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    correct = pred == y_true
    bins = [(0.0, 0.50), (0.50, 0.70), (0.70, 0.85), (0.85, 1.01)]
    out: list[dict[str, Any]] = []
    for lo, hi in bins:
        mask = (conf >= lo) & (conf < hi)
        n = int(mask.sum())
        out.append(
            {
                "confidence_range": f"[{lo:.2f}, {min(hi, 1.0):.2f}]",
                "n_samples": n,
                "accuracy": float(correct[mask].mean()) if n > 0 else None,
            }
        )
    return out


def run_raw_d01_audit(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit unmodified UCI D01 (1,941 rows) against all 38 rules and duplicate checks."""
    tier1_report = validate_uci_frame(raw_df, strict_physical=False)
    strict_report = validate_uci_frame(raw_df, strict_physical=True)

    anomaly_indices = strict_report["failing_row_indices"]
    anomalies: list[dict[str, Any]] = []
    for idx in anomaly_indices:
        row = raw_df.loc[idx]
        area = int(row["Pixels_Areas"])
        lum_sum = int(row["Sum_of_Luminosity"])
        min_lum = int(row["Minimum_of_Luminosity"])
        max_lum = int(row["Maximum_of_Luminosity"])
        anomalies.append(
            {
                "row_index": int(idx),
                "pixels_areas": area,
                "sum_of_luminosity": lum_sum,
                "minimum_of_luminosity": min_lum,
                "maximum_of_luminosity": max_lum,
                "mean_luminosity": float(lum_sum / area),
                "expected_min_sum": int(area * min_lum),
                "luminosity_index": float(row["Luminosity_Index"]),
            }
        )

    curated_df = raw_df.loc[strict_report["_pass_mask"]].copy().reset_index(drop=True)
    curated_report = validate_or_reject(curated_df, strict_physical=True)

    dups = detect_duplicates(raw_df)
    near_dups = detect_near_duplicates(raw_df, threshold=0.10)
    n_campaigns = int(assign_mill_campaign_groups(raw_df).nunique())

    # Capture luminosity physics pairs for Fig. 2a (sampled background + anomalies)
    mean_lum = (raw_df["Sum_of_Luminosity"] / raw_df["Pixels_Areas"]).to_numpy(dtype="float64")
    min_lum_arr = raw_df["Minimum_of_Luminosity"].to_numpy(dtype="float64")
    lum_points = [
        {"min_lum": float(min_lum_arr[i]), "mean_lum": float(mean_lum[i]), "valid": i not in anomaly_indices}
        for i in range(0, len(raw_df), 4)
    ]
    for idx in anomaly_indices:
        lum_points.append(
            {
                "row_index": int(idx),
                "min_lum": float(min_lum_arr[idx]),
                "mean_lum": float(mean_lum[idx]),
                "valid": False,
            }
        )

    audit = {
        "n_raw_rows": int(len(raw_df)),
        "n_curated_rows": int(len(curated_df)),
        "tier1_structural_math": serialize_validation_report(tier1_report),
        "strict_physical_raw": serialize_validation_report(strict_report),
        "curated_strict_physical": serialize_validation_report(curated_report),
        "real_raw_anomalies": anomalies,
        "duplicates": dups,
        "near_duplicates": near_dups,
        "n_mill_campaign_groups": n_campaigns,
        "luminosity_audit_points": lum_points,
    }
    return curated_df, audit


def run_corrupt_rejection_proof(curated_df: pd.DataFrame, *, seed: int = 42) -> dict[str, Any]:
    """Prove the gate rejects a synthetically corrupted copy of D01 with a readable report."""
    corrupted_df, corruption_manifest = corrupt_mill_dataset(curated_df, seed=seed)

    rejected = False
    rejection_report: dict[str, Any] = {}
    try:
        validate_or_reject(corrupted_df, strict_physical=True)
    except QualityGateRejectionError as exc:
        rejected = True
        rejection_report = serialize_validation_report(exc.report)

    if not rejected:
        raise RuntimeError("Quality gate failed to reject deliberately corrupted D01 copy!")

    # Summarize failing counts by rule category for Fig. 1c
    category_failures: dict[str, int] = {}
    for rule in rejection_report["rule_results"]:
        cat = str(rule["category"])
        category_failures[cat] = category_failures.get(cat, 0) + int(rule["n_failing"])

    # Also measure what happens when running apply_quality_gate on the full corrupted copy
    gated_df, gate_audit = apply_quality_gate(
        corrupted_df, strict_physical=True, clean_label_noise=True, seed=seed
    )

    # Sample raw vs normalized thicknesses for Fig. 2b histogram
    raw_thick = pd.to_numeric(corrupted_df["Steel_Plate_Thickness"], errors="coerce").dropna().to_numpy()
    norm_thick = gated_df["Steel_Plate_Thickness"].to_numpy()

    return {
        "synthetic": True,
        "rejected": rejected,
        "corruption_manifest": corruption_manifest,
        "rejection_report": rejection_report,
        "category_failure_counts": category_failures,
        "full_gate_audit": gate_audit,
        "thickness_distributions": {
            "raw_corrupted_values": [float(v) for v in raw_thick[:300]],
            "raw_corrupted_min": float(raw_thick.min()),
            "raw_corrupted_max": float(raw_thick.max()),
            "normalized_mm_values": [float(v) for v in norm_thick[:300]],
            "normalized_mm_min": float(norm_thick.min()),
            "normalized_mm_max": float(norm_thick.max()),
        },
    }


def run_campaign_leakage_cv(curated_df: pd.DataFrame, *, seed: int = 42) -> dict[str, Any]:
    """Compare Stratified 5-fold CV vs R09 Mill-Campaign StratifiedGroupKFold on clean D01."""
    X = curated_df[list(UCI_FEATURE_COLUMNS)].to_numpy(dtype="float64")
    y = np.argmax(curated_df[list(UCI_FAULT_COLUMNS)].to_numpy(dtype=int), axis=1)
    groups = assign_mill_campaign_groups(curated_df).to_numpy()

    strat_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    group_cv = StratifiedGroupKFold(n_splits=5)

    strat_acc: list[float] = []
    strat_f1: list[float] = []
    group_acc: list[float] = []
    group_f1: list[float] = []

    for tr_idx, va_idx in strat_cv.split(X, y):
        clf = _make_classifier("extra_trees", seed)
        clf.fit(X[tr_idx], y[tr_idx])
        pred = clf.predict(X[va_idx])
        strat_acc.append(float(accuracy_score(y[va_idx], pred)))
        strat_f1.append(float(f1_score(y[va_idx], pred, average="macro", zero_division=0)))

    for tr_idx, va_idx in group_cv.split(X, y, groups):
        clf = _make_classifier("extra_trees", seed)
        clf.fit(X[tr_idx], y[tr_idx])
        pred = clf.predict(X[va_idx])
        group_acc.append(float(accuracy_score(y[va_idx], pred)))
        group_f1.append(float(f1_score(y[va_idx], pred, average="macro", zero_division=0)))

    strat_acc_s = _stat_summary(strat_acc)
    strat_f1_s = _stat_summary(strat_f1)
    group_acc_s = _stat_summary(group_acc)
    group_f1_s = _stat_summary(group_f1)

    return {
        "n_samples": int(len(curated_df)),
        "n_campaign_groups": int(len(np.unique(groups))),
        "stratified_kfold": {"accuracy": strat_acc_s, "macro_f1": strat_f1_s},
        "campaign_grouped_kfold": {"accuracy": group_acc_s, "macro_f1": group_f1_s},
        "accuracy_inflation_gap": float(strat_acc_s["mean"] - group_acc_s["mean"]),
        "macro_f1_inflation_gap": float(strat_f1_s["mean"] - group_f1_s["mean"]),
    }


def run_before_after_benchmark(
    curated_df: pd.DataFrame,
    *,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, Any]:
    """Evaluate leaky QA, uncleaned corrupted, post-gate cleaned, and clean oracle across 5 seeds."""
    y_all = np.argmax(curated_df[list(UCI_FAULT_COLUMNS)].to_numpy(dtype=int), axis=1)

    models_out: dict[str, Any] = {}
    for model_name in ("extra_trees", "random_forest"):
        leaky_acc, leaky_f1, leaky_overlaps = [], [], []
        before_acc, before_f1 = [], []
        after_acc, after_f1 = [], []
        oracle_acc, oracle_f1 = [], []

        before_per_class: dict[str, list[float]] = {c: [] for c in CLASS_NAMES}
        after_per_class: dict[str, list[float]] = {c: [] for c in CLASS_NAMES}
        after_cms: list[np.ndarray] = []
        leakage_checks: list[dict[str, Any]] = []
        gate_row_counts: list[dict[str, int]] = []
        modality_seed42: list[dict[str, Any]] = []

        for seed in seeds:
            # 1. Leaky internal QA: random split AFTER batch duplication on corrupted copy
            corrupted_full, _ = corrupt_mill_dataset(curated_df, seed=seed + 200)
            y_corr_full = np.argmax(
                corrupted_full[list(UCI_FAULT_COLUMNS)].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(),
                axis=1,
            )
            tr_leak, te_leak = train_test_split(
                corrupted_full, test_size=0.20, random_state=seed, stratify=y_corr_full
            )
            leak_audit = audit_split_leakage(tr_leak, te_leak)
            leaky_overlaps.append(int(leak_audit["exact_feature_overlap_count"]))

            X_tr_l, y_tr_l = _extract_naive_xy(tr_leak)
            X_te_l, y_te_l = _extract_naive_xy(te_leak)
            pipe_l = Pipeline([("imp", SimpleImputer(strategy="median")), ("clf", _make_classifier(model_name, seed))])
            pipe_l.fit(X_tr_l, y_tr_l)
            ev_l = _evaluate_predictions(y_te_l, pipe_l.predict(X_te_l))
            leaky_acc.append(ev_l["accuracy"])
            leaky_f1.append(ev_l["macro_f1"])

            # 2. Clean untouched 80/20 split of curated D01
            train_clean, test_clean = train_test_split(
                curated_df, test_size=0.20, random_state=seed, stratify=y_all
            )
            X_te_clean = test_clean[list(UCI_FEATURE_COLUMNS)].to_numpy(dtype="float64")
            y_te_clean = np.argmax(test_clean[list(UCI_FAULT_COLUMNS)].to_numpy(dtype=int), axis=1)

            # Corrupt ONLY training data
            train_corrupted, _ = corrupt_mill_dataset(train_clean, seed=seed + 100)

            # 2a. Before quality gate (naive median imputation, ignoring Thickness_Unit/dups/noise)
            X_tr_b, y_tr_b = _extract_naive_xy(train_corrupted)
            pipe_b = Pipeline([("imp", SimpleImputer(strategy="median")), ("clf", _make_classifier(model_name, seed))])
            pipe_b.fit(X_tr_b, y_tr_b)
            ev_b = _evaluate_predictions(y_te_clean, pipe_b.predict(X_te_clean))
            before_acc.append(ev_b["accuracy"])
            before_f1.append(ev_b["macro_f1"])
            for c in CLASS_NAMES:
                before_per_class[c].append(ev_b["per_class_f1"][c])

            # 2b. After quality gate (Pint units -> Pandera 38 rules -> dedup -> OOF noise filter)
            train_gated, gate_audit = apply_quality_gate(
                train_corrupted, strict_physical=True, clean_label_noise=True, seed=seed
            )
            split_audit = audit_split_leakage(train_gated, test_clean)
            if not split_audit["zero_exact_leakage"]:
                raise RuntimeError(f"Train/test leakage detected after gate on seed {seed}")
            leakage_checks.append(split_audit)
            gate_row_counts.append(
                {
                    "seed": int(seed),
                    "n_corrupted_train": gate_audit["n_input_rows"],
                    "n_post_gate_train": gate_audit["n_output_rows"],
                    "n_schema_quarantined": gate_audit["n_quarantined_schema"],
                    "n_exact_dedup": gate_audit["n_exact_duplicates_dropped"],
                    "n_noise_quarantined": gate_audit["n_label_noise_quarantined"],
                }
            )

            X_tr_a = train_gated[list(UCI_FEATURE_COLUMNS)].to_numpy(dtype="float64")
            y_tr_a = np.argmax(train_gated[list(UCI_FAULT_COLUMNS)].to_numpy(dtype=int), axis=1)
            clf_a = _make_classifier(model_name, seed)
            clf_a.fit(X_tr_a, y_tr_a)
            pred_a = clf_a.predict(X_te_clean)
            ev_a = _evaluate_predictions(y_te_clean, pred_a)
            after_acc.append(ev_a["accuracy"])
            after_f1.append(ev_a["macro_f1"])
            for c in CLASS_NAMES:
                after_per_class[c].append(ev_a["per_class_f1"][c])
            after_cms.append(np.asarray(ev_a["confusion_matrix_normalized"], dtype="float64"))
            if seed == 42 and model_name == "extra_trees":
                modality_seed42 = _error_modality_bins(y_te_clean, clf_a.predict_proba(X_te_clean))

            # 2c. Clean oracle reference
            X_tr_o = train_clean[list(UCI_FEATURE_COLUMNS)].to_numpy(dtype="float64")
            y_tr_o = np.argmax(train_clean[list(UCI_FAULT_COLUMNS)].to_numpy(dtype=int), axis=1)
            clf_o = _make_classifier(model_name, seed)
            clf_o.fit(X_tr_o, y_tr_o)
            ev_o = _evaluate_predictions(y_te_clean, clf_o.predict(X_te_clean))
            oracle_acc.append(ev_o["accuracy"])
            oracle_f1.append(ev_o["macro_f1"])

        before_acc_s = _stat_summary(before_acc)
        after_acc_s = _stat_summary(after_acc)
        before_f1_s = _stat_summary(before_f1)
        after_f1_s = _stat_summary(after_f1)
        mean_cm = np.mean(np.stack(after_cms, axis=0), axis=0)

        models_out[model_name] = {
            "seeds": list(seeds),
            "leaky_internal_qa": {
                "accuracy": _stat_summary(leaky_acc),
                "macro_f1": _stat_summary(leaky_f1),
                "exact_train_test_overlap": _stat_summary([float(x) for x in leaky_overlaps]),
            },
            "before_gate_uncleaned": {
                "accuracy": before_acc_s,
                "macro_f1": before_f1_s,
                "per_class_f1": {c: _stat_summary(before_per_class[c]) for c in CLASS_NAMES},
            },
            "after_gate_cleaned": {
                "accuracy": after_acc_s,
                "macro_f1": after_f1_s,
                "per_class_f1": {c: _stat_summary(after_per_class[c]) for c in CLASS_NAMES},
                "confusion_matrix_normalized": [[float(x) for x in row] for row in mean_cm],
                "zero_train_test_leakage_all_seeds": all(c["zero_exact_leakage"] for c in leakage_checks),
                "gate_row_counts": gate_row_counts,
            },
            "clean_oracle_reference": {
                "accuracy": _stat_summary(oracle_acc),
                "macro_f1": _stat_summary(oracle_f1),
            },
            "delta_after_minus_before": {
                "accuracy_gain": float(after_acc_s["mean"] - before_acc_s["mean"]),
                "macro_f1_gain": float(after_f1_s["mean"] - before_f1_s["mean"]),
                "accuracy_gain_pp": float(100.0 * (after_acc_s["mean"] - before_acc_s["mean"])),
                "macro_f1_gain_pp": float(100.0 * (after_f1_s["mean"] - before_f1_s["mean"])),
            },
        }
        if model_name == "extra_trees":
            models_out[model_name]["y2r12_confidence_error_modality_seed42"] = modality_seed42

    return {
        "synthetic": True,
        "evaluation_target": "Clean untouched 20% test split of curated D01 (1,939 rows); only training data is synthetically corrupted",
        "class_names": CLASS_NAMES,
        "models": models_out,
    }


def reproduce(
    data_dir: str | Path,
    catalog_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the full deterministic Case 02 pipeline and write ``metrics.json`` & ``provenance.json``."""
    data_root = Path(data_dir)
    out_root = Path(output_dir)
    d01_zip = data_root / "steel_plates_faults.zip"

    catalog = load_catalog(catalog_path)
    d01_verified = verify_d01_archive(d01_zip)
    raw_df = load_uci_faults(d01_zip, verify_checksum=True)

    curated_df, raw_audit = run_raw_d01_audit(raw_df)
    unit_suite = run_unit_verification_suite()
    rejection_proof = run_corrupt_rejection_proof(curated_df, seed=seed)
    campaign_cv = run_campaign_leakage_cv(curated_df, seed=seed)
    before_after = run_before_after_benchmark(curated_df, seeds=SEEDS)

    # Determinism check: repeat unit verification + single-seed benchmark and verify exact equality
    unit_repeat = run_unit_verification_suite()
    ba_repeat = run_before_after_benchmark(curated_df, seeds=(seed,))
    et_main_s0 = before_after["models"]["extra_trees"]["after_gate_cleaned"]["accuracy"]["values"][0]
    et_rep_s0 = ba_repeat["models"]["extra_trees"]["after_gate_cleaned"]["accuracy"]["values"][0]
    determinism_ok = (unit_suite == unit_repeat) and (et_main_s0 == et_rep_s0)
    if not determinism_ok:
        raise RuntimeError("Determinism verification failed across repeated Case 02 runs!")

    metrics_payload: dict[str, Any] = {
        "case": "C02",
        "title": "Data-quality failures in a steel plate mill",
        "evidence_level": "A",
        "seed": int(seed),
        "determinism_check": bool(determinism_ok),
        "code_fingerprint": source_fingerprint(Path(__file__).resolve().parent),
        "data": {
            "D01": d01_verified,
            "D02_catalog_status": next(
                (d.get("status") for d in catalog["datasets"] if d.get("id") == "D02"),
                "mirror_licence_unverified",
            ),
        },
        "raw_d01_audit": raw_audit,
        "unit_verification": unit_suite,
        "corrupt_copy_rejection": rejection_proof,
        "campaign_leakage_cv": campaign_cv,
        "before_after_benchmark": before_after,
    }

    provenance_payload: dict[str, Any] = {
        "case": "C02",
        "evidence_level": "A",
        "determinism_check": bool(determinism_ok),
        "code_fingerprint": metrics_payload["code_fingerprint"],
        "environment": environment_info(),
        "datasets": {"D01": d01_verified},
        "catalog_version": catalog.get("version"),
        "synthetic_experiments": [
            "corrupt_copy_rejection (synthetic mill corruption of open D01)",
            "before_after_benchmark (synthetic mill corruption of D01 training splits; evaluated on clean untouched D01 test splits)",
        ],
    }

    write_json(out_root / "metrics.json", metrics_payload)
    write_json(out_root / "provenance.json", provenance_payload)
    return metrics_payload
