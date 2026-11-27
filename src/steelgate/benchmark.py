"""Case 02 reproducible 3-database benchmark (D01, D04, D09), ablation, and leakage audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from steelgate.corruption import (
    apply_d04_quality_gate,
    apply_d09_quality_gate,
    apply_quality_gate,
    corrupt_d04_dataset,
    corrupt_d09_dataset,
    corrupt_mill_dataset,
    repair_uci_derived_indices,
)
from steelgate.leakage import (
    assign_mill_campaign_groups,
    audit_split_leakage,
    detect_duplicates,
    detect_near_duplicates,
)
from steelgate.loaders import (
    D04_SPLITS_META,
    D04_STEELS_META,
    D09_ENERGY_META,
    UCI_ALL_COLUMNS,
    UCI_D01_META,
    UCI_FAULT_COLUMNS,
    UCI_FEATURE_COLUMNS,
    load_catalog,
    load_matbench_splits,
    load_matbench_steels,
    load_steel_energy,
    load_uci_faults,
    verify_dataset_file,
)
from steelgate.provenance import environment_info, source_fingerprint, write_json
from steelgate.schemas import (
    D04_QUALITY_RULES,
    D09_QUALITY_RULES,
    QualityGateRejectionError,
    UCI_QUALITY_RULES,
    serialize_validation_report,
    validate_d04_frame,
    validate_d04_or_reject,
    validate_d09_frame,
    validate_d09_or_reject,
    validate_or_reject,
    validate_uci_frame,
)
from steelgate.units import (
    at_pct_to_wt_pct,
    normalize_series_units,
    parse_formula_to_at_pct,
    run_unit_verification_suite,
    wt_pct_to_at_pct,
)

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
    if model_name == "voting_ensemble":
        return VotingClassifier(
            estimators=[
                ("et", ExtraTreesClassifier(n_estimators=300, random_state=seed, n_jobs=1)),
                (
                    "hgb",
                    HistGradientBoostingClassifier(
                        max_iter=150, learning_rate=0.06, max_depth=6, random_state=seed
                    ),
                ),
            ],
            voting="soft",
        )
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


# ---------------------------------------------------------------------------
# D01 Audits, Corruption Rejection, Campaign Leakage & 5-Stage Ablation
# ---------------------------------------------------------------------------

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

    category_failures: dict[str, int] = {}
    for rule in rejection_report["rule_results"]:
        cat = str(rule["category"])
        category_failures[cat] = category_failures.get(cat, 0) + int(rule["n_failing"])

    gated_df, gate_audit = apply_quality_gate(
        corrupted_df, strict_physical=True, clean_label_noise=True, repair_derived=True, seed=seed
    )

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

    strat_acc, strat_f1, group_acc, group_f1 = [], [], [], []
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

    strat_acc_s, strat_f1_s = _stat_summary(strat_acc), _stat_summary(strat_f1)
    group_acc_s, group_f1_s = _stat_summary(group_acc), _stat_summary(group_f1)

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
    """Evaluate 3 classifiers + 5-stage cumulative ablation across 5 seeds on D01."""
    y_all = np.argmax(curated_df[list(UCI_FAULT_COLUMNS)].to_numpy(dtype=int), axis=1)

    # Precompute splits, corruptions, and gated training sets once per seed
    seed_bundles: list[dict[str, Any]] = []
    ablation_acc: dict[str, list[float]] = {
        "Before Gate (ET)": [],
        "+S1 Pint Units": [],
        "+S2 Repair & 38 Rules": [],
        "+S3 Dedup": [],
        "+S4 Confident Learning": [],
        "+S5 ET+HGB Ensemble": [],
        "Clean Oracle (ET)": [],
        "Clean Oracle (ET+HGB)": [],
    }
    ablation_f1: dict[str, list[float]] = {k: [] for k in ablation_acc}

    for seed in seeds:
        corrupted_full, _ = corrupt_mill_dataset(curated_df, seed=seed + 200)
        y_corr_full = np.argmax(
            corrupted_full[list(UCI_FAULT_COLUMNS)].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(),
            axis=1,
        )
        tr_leak, te_leak = train_test_split(
            corrupted_full, test_size=0.20, random_state=seed, stratify=y_corr_full
        )
        leak_audit = audit_split_leakage(tr_leak, te_leak)

        train_clean, test_clean = train_test_split(
            curated_df, test_size=0.20, random_state=seed, stratify=y_all
        )
        X_te_clean = test_clean[list(UCI_FEATURE_COLUMNS)].to_numpy(dtype="float64")
        y_te_clean = np.argmax(test_clean[list(UCI_FAULT_COLUMNS)].to_numpy(dtype=int), axis=1)

        train_corrupted, _ = corrupt_mill_dataset(train_clean, seed=seed + 100)
        train_gated, gate_audit = apply_quality_gate(
            train_corrupted, strict_physical=True, clean_label_noise=True, repair_derived=True, seed=seed
        )
        split_audit = audit_split_leakage(train_gated, test_clean)
        if not split_audit["zero_exact_leakage"]:
            raise RuntimeError(f"Train/test leakage detected after gate on seed {seed}")

        # Intermediate ablation steps (S1 Units, S2 Repair+Schema, S3 Dedup)
        s1 = train_corrupted.copy()
        s1["Steel_Plate_Thickness"], _ = normalize_series_units(
            s1["Steel_Plate_Thickness"], s1["Thickness_Unit"], target_unit="mm"
        )
        s1 = s1.drop(columns=["Thickness_Unit"])

        s2, _ = repair_uci_derived_indices(s1)
        s2_rep = validate_uci_frame(s2, strict_physical=True)
        s2 = s2.loc[s2_rep["_pass_mask"]].copy().reset_index(drop=True)

        s3 = s2.drop_duplicates(subset=list(UCI_ALL_COLUMNS), keep="first").reset_index(drop=True)

        for stage_key, frame in [
            ("Before Gate (ET)", train_corrupted),
            ("+S1 Pint Units", s1),
            ("+S2 Repair & 38 Rules", s2),
            ("+S3 Dedup", s3),
        ]:
            X_st, y_st = _extract_naive_xy(frame)
            pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("clf", _make_classifier("extra_trees", seed))])
            pipe.fit(X_st, y_st)
            pred = pipe.predict(X_te_clean)
            ablation_acc[stage_key].append(float(accuracy_score(y_te_clean, pred)))
            ablation_f1[stage_key].append(float(f1_score(y_te_clean, pred, average="macro", zero_division=0)))

        seed_bundles.append(
            {
                "seed": seed,
                "tr_leak": tr_leak,
                "te_leak": te_leak,
                "leak_audit": leak_audit,
                "train_clean": train_clean,
                "test_clean": test_clean,
                "X_te_clean": X_te_clean,
                "y_te_clean": y_te_clean,
                "train_corrupted": train_corrupted,
                "train_gated": train_gated,
                "gate_audit": gate_audit,
                "split_audit": split_audit,
            }
        )

    models_out: dict[str, Any] = {}
    for model_name in ("extra_trees", "random_forest", "voting_ensemble"):
        leaky_acc, leaky_f1, leaky_overlaps = [], [], []
        before_acc, before_f1 = [], []
        after_acc, after_f1 = [], []
        oracle_acc, oracle_f1 = [], []
        before_per_class = {c: [] for c in CLASS_NAMES}
        after_per_class = {c: [] for c in CLASS_NAMES}
        after_cms = []
        leakage_checks = []
        gate_row_counts = []
        modality_seed42 = []

        for b in seed_bundles:
            seed = b["seed"]
            leaky_overlaps.append(int(b["leak_audit"]["exact_feature_overlap_count"]))
            leakage_checks.append(b["split_audit"])
            ga = b["gate_audit"]
            gate_row_counts.append(
                {
                    "seed": int(seed),
                    "n_corrupted_train": ga["n_input_rows"],
                    "n_post_gate_train": ga["n_output_rows"],
                    "n_repaired_derived": ga["n_repaired_derived_rows"],
                    "n_schema_quarantined": ga["n_quarantined_schema"],
                    "n_exact_dedup": ga["n_exact_duplicates_dropped"],
                    "n_noise_quarantined": ga["n_label_noise_quarantined"],
                }
            )

            X_tr_l, y_tr_l = _extract_naive_xy(b["tr_leak"])
            X_te_l, y_te_l = _extract_naive_xy(b["te_leak"])
            pipe_l = Pipeline([("imp", SimpleImputer(strategy="median")), ("clf", _make_classifier(model_name, seed))])
            pipe_l.fit(X_tr_l, y_tr_l)
            ev_l = _evaluate_predictions(y_te_l, pipe_l.predict(X_te_l))
            leaky_acc.append(ev_l["accuracy"])
            leaky_f1.append(ev_l["macro_f1"])

            X_te_clean, y_te_clean = b["X_te_clean"], b["y_te_clean"]
            X_tr_b, y_tr_b = _extract_naive_xy(b["train_corrupted"])
            pipe_b = Pipeline([("imp", SimpleImputer(strategy="median")), ("clf", _make_classifier(model_name, seed))])
            pipe_b.fit(X_tr_b, y_tr_b)
            ev_b = _evaluate_predictions(y_te_clean, pipe_b.predict(X_te_clean))
            before_acc.append(ev_b["accuracy"])
            before_f1.append(ev_b["macro_f1"])
            for c in CLASS_NAMES:
                before_per_class[c].append(ev_b["per_class_f1"][c])

            X_tr_a = b["train_gated"][list(UCI_FEATURE_COLUMNS)].to_numpy(dtype="float64")
            y_tr_a = np.argmax(b["train_gated"][list(UCI_FAULT_COLUMNS)].to_numpy(dtype=int), axis=1)
            clf_a = _make_classifier(model_name, seed).fit(X_tr_a, y_tr_a)
            ev_a = _evaluate_predictions(y_te_clean, clf_a.predict(X_te_clean))
            after_acc.append(ev_a["accuracy"])
            after_f1.append(ev_a["macro_f1"])
            for c in CLASS_NAMES:
                after_per_class[c].append(ev_a["per_class_f1"][c])
            after_cms.append(np.asarray(ev_a["confusion_matrix_normalized"], dtype="float64"))
            if seed == 42 and model_name == "voting_ensemble":
                modality_seed42 = _error_modality_bins(y_te_clean, clf_a.predict_proba(X_te_clean))

            X_tr_o = b["train_clean"][list(UCI_FEATURE_COLUMNS)].to_numpy(dtype="float64")
            y_tr_o = np.argmax(b["train_clean"][list(UCI_FAULT_COLUMNS)].to_numpy(dtype=int), axis=1)
            clf_o = _make_classifier(model_name, seed).fit(X_tr_o, y_tr_o)
            ev_o = _evaluate_predictions(y_te_clean, clf_o.predict(X_te_clean))
            oracle_acc.append(ev_o["accuracy"])
            oracle_f1.append(ev_o["macro_f1"])

        if model_name == "extra_trees":
            ablation_acc["+S4 Confident Learning"] = list(after_acc)
            ablation_f1["+S4 Confident Learning"] = list(after_f1)
            ablation_acc["Clean Oracle (ET)"] = list(oracle_acc)
            ablation_f1["Clean Oracle (ET)"] = list(oracle_f1)
        elif model_name == "voting_ensemble":
            ablation_acc["+S5 ET+HGB Ensemble"] = list(after_acc)
            ablation_f1["+S5 ET+HGB Ensemble"] = list(after_f1)
            ablation_acc["Clean Oracle (ET+HGB)"] = list(oracle_acc)
            ablation_f1["Clean Oracle (ET+HGB)"] = list(oracle_f1)

        before_acc_s, after_acc_s = _stat_summary(before_acc), _stat_summary(after_acc)
        before_f1_s, after_f1_s = _stat_summary(before_f1), _stat_summary(after_f1)
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
        if model_name == "voting_ensemble":
            models_out[model_name]["y2r12_confidence_error_modality_seed42"] = modality_seed42

    ablation_ladder = [
        {"stage": stage, "accuracy": _stat_summary(ablation_acc[stage]), "macro_f1": _stat_summary(ablation_f1[stage])}
        for stage in ablation_acc
    ]

    return {
        "synthetic": True,
        "evaluation_target": "Clean untouched 20% test split of curated D01 (1,939 rows); only training data is synthetically corrupted",
        "class_names": CLASS_NAMES,
        "ablation_ladder": ablation_ladder,
        "models": models_out,
    }


# ---------------------------------------------------------------------------
# D04: MatBench Steels Audit & Official 5-Fold Regression Quality Gate
# ---------------------------------------------------------------------------

def _featurize_d04(formulas: pd.Series, vocab: list[str]) -> np.ndarray:
    mat = np.zeros((len(formulas), len(vocab)), dtype="float64")
    for i, f in enumerate(formulas):
        comp, _ = parse_formula_to_at_pct(str(f))
        for j, el in enumerate(vocab):
            mat[i, j] = comp.get(el, 0.0) / 100.0
    return mat


def run_d04_audit_and_benchmark(
    d04_df: pd.DataFrame,
    splits: dict[str, dict[str, list[str]]],
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Audit raw MatBench D04 (312 alloys, M01-M12) and run 5-fold before/after regression."""
    tier1_rep = validate_d04_frame(d04_df, strict_physical=False)
    strict_rep = validate_d04_frame(d04_df, strict_physical=True)

    raw_sums: list[dict[str, Any]] = []
    max_stoich_rt_err = 0.0
    for idx, row in d04_df.iterrows():
        at_comp, rsum = parse_formula_to_at_pct(str(row["composition"]))
        wt_comp = wt_pct_to_at_pct(at_pct_to_wt_pct(at_comp))
        err = max(abs(wt_comp[el] - at_comp[el]) for el in at_comp)
        max_stoich_rt_err = max(max_stoich_rt_err, err)
        raw_sums.append(
            {
                "row_index": int(idx),
                "id": str(row["id"]),
                "composition": str(row["composition"]),
                "raw_fraction_sum": float(rsum),
                "deviation": float(abs(rsum - 1.0)),
                "flagged_m12": bool(abs(rsum - 1.0) > 1e-3),
            }
        )

    real_anomalies = [r for r in raw_sums if r["flagged_m12"]]

    # Corrupt-copy rejection proof
    corrupted_d04, corr_manifest = corrupt_d04_dataset(d04_df, seed=seed)
    rejected = False
    rej_report: dict[str, Any] = {}
    try:
        validate_d04_or_reject(corrupted_d04, strict_physical=True)
    except QualityGateRejectionError as exc:
        rejected = True
        rej_report = serialize_validation_report(exc.report)
    if not rejected:
        raise RuntimeError("D04 quality gate failed to reject synthetic corrupted alloy table!")

    # Official 5-fold MatBench before/after benchmark
    by_id = d04_df.set_index("id")
    b_maes, b_rmses, b_r2s = [], [], []
    a_maes, a_rmses, a_r2s = [], [], []
    o_maes, o_rmses, o_r2s = [], [], []
    zero_leakage_all_folds = True

    for fold_idx in range(5):
        sp = splits[f"fold_{fold_idx}"]
        tr = by_id.loc[sp["train"]].reset_index()
        te = by_id.loc[sp["test"]].reset_index()

        vocab_set: set[str] = set()
        for f in tr["composition"]:
            comp, _ = parse_formula_to_at_pct(str(f))
            vocab_set.update(comp.keys())
        vocab = sorted(vocab_set)

        X_te = _featurize_d04(te["composition"], vocab)
        y_te = te["yield_strength_mpa"].to_numpy(dtype="float64")

        tr_corr, _ = corrupt_d04_dataset(tr, seed=seed + fold_idx)

        # Before gate: naive dropna only
        tr_b = tr_corr.dropna(subset=["yield_strength_mpa"])
        X_b = _featurize_d04(tr_b["composition"], vocab)
        y_b = tr_b["yield_strength_mpa"].to_numpy(dtype="float64")
        pred_b = ExtraTreesRegressor(n_estimators=300, random_state=seed, n_jobs=1).fit(X_b, y_b).predict(X_te)
        b_maes.append(float(mean_absolute_error(y_te, pred_b)))
        b_rmses.append(float(np.sqrt(mean_squared_error(y_te, pred_b))))
        b_r2s.append(float(r2_score(y_te, pred_b)))

        # After gate: Pint MPa norm + M01-M11 + dedup + OOF MAD filter
        tr_g, _ = apply_d04_quality_gate(tr_corr, clean_outliers=True, seed=seed)
        if set(tr_g["id"]).intersection(set(te["id"])):
            zero_leakage_all_folds = False
        X_a = _featurize_d04(tr_g["composition"], vocab)
        y_a = tr_g["yield_strength_mpa"].to_numpy(dtype="float64")
        pred_a = ExtraTreesRegressor(n_estimators=300, random_state=seed, n_jobs=1).fit(X_a, y_a).predict(X_te)
        a_maes.append(float(mean_absolute_error(y_te, pred_a)))
        a_rmses.append(float(np.sqrt(mean_squared_error(y_te, pred_a))))
        a_r2s.append(float(r2_score(y_te, pred_a)))

        # Clean oracle
        X_o = _featurize_d04(tr["composition"], vocab)
        y_o = tr["yield_strength_mpa"].to_numpy(dtype="float64")
        pred_o = ExtraTreesRegressor(n_estimators=300, random_state=seed, n_jobs=1).fit(X_o, y_o).predict(X_te)
        o_maes.append(float(mean_absolute_error(y_te, pred_o)))
        o_rmses.append(float(np.sqrt(mean_squared_error(y_te, pred_o))))
        o_r2s.append(float(r2_score(y_te, pred_o)))

    b_mae_s, a_mae_s = _stat_summary(b_maes), _stat_summary(a_maes)
    mae_red_pct = float(100.0 * (b_mae_s["mean"] - a_mae_s["mean"]) / max(b_mae_s["mean"], 1e-9))

    return {
        "n_raw_rows": int(len(d04_df)),
        "tier1_m01_m11": serialize_validation_report(tier1_rep),
        "strict_m01_m12": serialize_validation_report(strict_rep),
        "max_at_wt_at_round_trip_err": float(max_stoich_rt_err),
        "real_raw_stoich_anomalies": real_anomalies,
        "raw_formula_sums": raw_sums,
        "corrupt_copy_rejection": {
            "synthetic": True,
            "rejected": rejected,
            "manifest": corr_manifest,
            "rejection_report": rej_report,
        },
        "five_fold_benchmark": {
            "synthetic": True,
            "zero_fold_id_leakage": zero_leakage_all_folds,
            "before_gate": {"mae_mpa": b_mae_s, "rmse_mpa": _stat_summary(b_rmses), "r2": _stat_summary(b_r2s)},
            "after_gate": {"mae_mpa": a_mae_s, "rmse_mpa": _stat_summary(a_rmses), "r2": _stat_summary(a_r2s)},
            "clean_oracle": {"mae_mpa": _stat_summary(o_maes), "rmse_mpa": _stat_summary(o_rmses), "r2": _stat_summary(o_r2s)},
            "mae_reduction_mpa": float(b_mae_s["mean"] - a_mae_s["mean"]),
            "mae_reduction_pct": mae_red_pct,
        },
    }


# ---------------------------------------------------------------------------
# D09: UCI Steel Industry Energy (35,040 rows) Audit & Temporal/Gate Benchmark
# ---------------------------------------------------------------------------

def _featurize_d09(df: pd.DataFrame) -> np.ndarray:
    """Extract leakage-safe reactive + schedule features (excludes CO2 and Power Factor)."""
    num = df[
        [
            "Lagging_Current_Reactive.Power_kVarh",
            "Leading_Current_Reactive_Power_kVarh",
            "NSM",
        ]
    ].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
    nsm = np.nan_to_num(num[:, 2], nan=0.0)
    sin_t = np.sin(2.0 * np.pi * nsm / 86400.0).reshape(-1, 1)
    cos_t = np.cos(2.0 * np.pi * nsm / 86400.0).reshape(-1, 1)
    wknd = (df["WeekStatus"].astype(str).str.strip() == "Weekend").astype("float64").to_numpy().reshape(-1, 1)
    lt = df["Load_Type"].astype(str).str.strip()
    load_oh = np.column_stack(
        [
            (lt == "Light_Load").astype("float64").to_numpy(),
            (lt == "Medium_Load").astype("float64").to_numpy(),
            (lt == "Maximum_Load").astype("float64").to_numpy(),
        ]
    )
    return np.hstack([num, sin_t, cos_t, wknd, load_oh])


def run_d09_audit_and_benchmark(d09_df: pd.DataFrame, *, seed: int = 42) -> dict[str, Any]:
    """Audit 35,040 rows of UCI D09 (E01-E14), test temporal leakage, and run energy gate."""
    strict_rep = validate_d09_or_reject(d09_df, strict_physical=True)

    # Sort chronologically (raw CSV stores each day's 00:00 at the end of that day)
    dt = pd.to_datetime(d09_df["date"], format="%d/%m/%Y %H:%M")
    sorted_df = d09_df.assign(_dt=dt).sort_values("_dt").drop(columns=["_dt"]).reset_index(drop=True)

    # Corrupt-copy rejection proof
    corrupted_d09, corr_manifest = corrupt_d09_dataset(sorted_df, seed=seed)
    rejected = False
    rej_report: dict[str, Any] = {}
    try:
        validate_d09_or_reject(corrupted_d09, strict_physical=True)
    except QualityGateRejectionError as exc:
        rejected = True
        rej_report = serialize_validation_report(exc.report)
    if not rejected:
        raise RuntimeError("D09 quality gate failed to reject synthetic corrupted energy table!")

    # Temporal leakage comparison on clean sorted D09: Random 80/20 vs Chronological 80/20
    X_all = _featurize_d09(sorted_df)
    y_all = sorted_df["Usage_kWh"].to_numpy(dtype="float64")

    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_all, y_all, test_size=0.20, random_state=seed)
    pred_rand = HistGradientBoostingRegressor(max_iter=150, max_depth=6, random_state=seed).fit(X_tr_r, y_tr_r).predict(X_te_r)
    rand_mae = float(mean_absolute_error(y_te_r, pred_rand))
    rand_r2 = float(r2_score(y_te_r, pred_rand))

    split_idx = int(round(0.80 * len(sorted_df)))
    tr_chron = sorted_df.iloc[:split_idx].copy().reset_index(drop=True)
    te_chron = sorted_df.iloc[split_idx:].copy().reset_index(drop=True)
    X_tr_c, y_tr_c = _featurize_d09(tr_chron), tr_chron["Usage_kWh"].to_numpy(dtype="float64")
    X_te_c, y_te_c = _featurize_d09(te_chron), te_chron["Usage_kWh"].to_numpy(dtype="float64")

    pred_oracle = HistGradientBoostingRegressor(max_iter=150, max_depth=6, random_state=seed).fit(X_tr_c, y_tr_c).predict(X_te_c)
    oracle_mae = float(mean_absolute_error(y_te_c, pred_oracle))
    oracle_r2 = float(r2_score(y_te_c, pred_oracle))

    # Corrupt ONLY chronological training window and evaluate on untouched clean last 20%
    tr_corr, _ = corrupt_d09_dataset(tr_chron, seed=seed + 10)
    tr_b = tr_corr.dropna(subset=["Usage_kWh"])
    X_b = _featurize_d09(tr_b)
    y_b = tr_b["Usage_kWh"].to_numpy(dtype="float64")
    pred_b = HistGradientBoostingRegressor(max_iter=150, max_depth=6, random_state=seed).fit(X_b, y_b).predict(X_te_c)
    before_mae = float(mean_absolute_error(y_te_c, pred_b))
    before_r2 = float(r2_score(y_te_c, pred_b))

    tr_gated, gate_audit = apply_d09_quality_gate(tr_corr, strict_physical=True)
    zero_ts_overlap = len(set(tr_gated["date"]).intersection(set(te_chron["date"]))) == 0
    X_a = _featurize_d09(tr_gated)
    y_a = tr_gated["Usage_kWh"].to_numpy(dtype="float64")
    pred_a = HistGradientBoostingRegressor(max_iter=150, max_depth=6, random_state=seed).fit(X_a, y_a).predict(X_te_c)
    after_mae = float(mean_absolute_error(y_te_c, pred_a))
    after_r2 = float(r2_score(y_te_c, pred_a))

    return {
        "n_raw_rows": int(len(d09_df)),
        "strict_e01_e14": serialize_validation_report(strict_rep),
        "corrupt_copy_rejection": {
            "synthetic": True,
            "rejected": rejected,
            "manifest": corr_manifest,
            "rejection_report": rej_report,
        },
        "temporal_leakage": {
            "random_80_20_mae_kwh": rand_mae,
            "random_80_20_r2": rand_r2,
            "chronological_80_20_mae_kwh": oracle_mae,
            "chronological_80_20_r2": oracle_r2,
            "lookahead_mae_inflation_kwh": float(oracle_mae - rand_mae),
        },
        "chronological_before_after": {
            "synthetic": True,
            "zero_timestamp_leakage": zero_ts_overlap,
            "gate_audit": {k: v for k, v in gate_audit.items() if k not in ("pre_gate_validation", "post_gate_validation")},
            "before_gate": {"mae_kwh": before_mae, "r2": before_r2},
            "after_gate": {"mae_kwh": after_mae, "r2": after_r2},
            "clean_oracle": {"mae_kwh": oracle_mae, "r2": oracle_r2},
            "mae_reduction_kwh": float(before_mae - after_mae),
            "mae_reduction_pct": float(100.0 * (before_mae - after_mae) / max(before_mae, 1e-9)),
        },
    }


def reproduce(
    data_dir: str | Path,
    catalog_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Run all 3 open-database audits, 64-rule gates, unit suite, and benchmarks."""
    data_root = Path(data_dir)
    out_root = Path(output_dir)

    d01_zip = data_root / UCI_D01_META["file"]
    d04_gz = data_root / D04_STEELS_META["file"]
    d04_splits_json = data_root / D04_SPLITS_META["file"]
    d09_zip = data_root / D09_ENERGY_META["file"]

    catalog = load_catalog(catalog_path)
    d01_verified = verify_dataset_file(d01_zip, UCI_D01_META)
    d04_verified = verify_dataset_file(d04_gz, D04_STEELS_META)
    d04_splits_verified = verify_dataset_file(d04_splits_json, D04_SPLITS_META)
    d09_verified = verify_dataset_file(d09_zip, D09_ENERGY_META)

    raw_d01 = load_uci_faults(d01_zip, verify_checksum=True)
    raw_d04 = load_matbench_steels(d04_gz, verify_checksum=True)
    d04_splits = load_matbench_splits(d04_splits_json, verify_checksum=True)
    raw_d09 = load_steel_energy(d09_zip, verify_checksum=True)

    curated_d01, raw_d01_audit = run_raw_d01_audit(raw_d01)
    unit_suite = run_unit_verification_suite()
    rejection_d01 = run_corrupt_rejection_proof(curated_d01, seed=seed)
    campaign_cv = run_campaign_leakage_cv(curated_d01, seed=seed)
    before_after_d01 = run_before_after_benchmark(curated_d01, seeds=SEEDS)

    d04_results = run_d04_audit_and_benchmark(raw_d04, d04_splits, seed=seed)
    d09_results = run_d09_audit_and_benchmark(raw_d09, seed=seed)

    # Determinism check across unit suite + D01 seed-42 ensemble
    unit_repeat = run_unit_verification_suite()
    ba_repeat = run_before_after_benchmark(curated_d01, seeds=(seed,))
    ens_main = before_after_d01["models"]["voting_ensemble"]["after_gate_cleaned"]["accuracy"]["values"][0]
    ens_rep = ba_repeat["models"]["voting_ensemble"]["after_gate_cleaned"]["accuracy"]["values"][0]
    determinism_ok = (unit_suite == unit_repeat) and (ens_main == ens_rep)
    if not determinism_ok:
        raise RuntimeError("Determinism verification failed across repeated Case 02 runs!")

    total_rules = len(UCI_QUALITY_RULES) + len(D04_QUALITY_RULES) + len(D09_QUALITY_RULES)
    total_records = int(len(raw_d01) + len(raw_d04) + len(raw_d09))

    metrics_payload: dict[str, Any] = {
        "case": "C02",
        "version": "0.7.0",
        "title": "Data-quality failures in a steel plate mill & multi-database quality gate",
        "evidence_level": "A",
        "seed": int(seed),
        "determinism_check": bool(determinism_ok),
        "total_real_records": total_records,
        "total_schema_rules": total_rules,
        "code_fingerprint": source_fingerprint(Path(__file__).resolve().parent),
        "data": {
            "D01": d01_verified,
            "D04": d04_verified,
            "D04-splits": d04_splits_verified,
            "D09": d09_verified,
            "D02_catalog_status": next(
                (d.get("status") for d in catalog["datasets"] if d.get("id") == "D02"),
                "mirror_licence_unverified",
            ),
        },
        "raw_d01_audit": raw_d01_audit,
        "unit_verification": unit_suite,
        "corrupt_copy_rejection": rejection_d01,
        "campaign_leakage_cv": campaign_cv,
        "before_after_benchmark": before_after_d01,
        "d04_benchmark": d04_results,
        "d09_benchmark": d09_results,
    }

    provenance_payload: dict[str, Any] = {
        "case": "C02",
        "version": "0.7.0",
        "evidence_level": "A",
        "determinism_check": bool(determinism_ok),
        "code_fingerprint": metrics_payload["code_fingerprint"],
        "environment": environment_info(),
        "datasets": {
            "D01": d01_verified,
            "D04": d04_verified,
            "D04-splits": d04_splits_verified,
            "D09": d09_verified,
        },
        "catalog_version": catalog.get("version"),
        "synthetic_experiments": [
            "corrupt_copy_rejection (synthetic mill corruption of open D01)",
            "before_after_benchmark (synthetic corruption of D01 training splits; clean untouched D01 test splits)",
            "d04_benchmark (synthetic corruption of D04 official training folds; clean untouched D04 test folds)",
            "d09_benchmark (synthetic corruption of D09 chronological training window; clean untouched D09 test window)",
        ],
    }

    write_json(out_root / "metrics.json", metrics_payload)
    write_json(out_root / "provenance.json", provenance_payload)
    return metrics_payload
