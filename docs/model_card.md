# Model card — Case 02 Multi-Database SteelGate Models

**Version:** `0.7.0` · **Date:** 2026-11-27

## Intended use

Demonstrates how a unified 64-rule schema, Pint unit-normalization, and leakage-control data lake (`steelgate`) protects downstream steel classification and property/energy regression models (`D01` fault classification, `D04` yield-strength regression, `D09` mill energy forecasting). Intended strictly as an open-data data-governance benchmark—**not** for production plant control or alloy certification.

## Data

- **Sources, licences & checksums:**
  - `D01` UCI Steel Plates Faults (CC BY 4.0, DOI `10.24432/C5J88N`, SHA-256 `cb8eb9859198b63f053e443513036b401746fa517ef58bd17c846c6741c93919`, `100,890` bytes; 1,941 raw / 1,939 curated).
  - `D04` MatBench `matbench_steels` (Public benchmark, DOI `10.1038/s41524-020-00406-3`, SHA-256 `473bc4957b2ea5e6465aef84bc29bb48ac34db27d69ea4ec5f508745c6fae252`, `8,836` bytes; 312 alloys + official 5-fold splits `ef3bc6cf09ec08a23a8d0a6f1e04c0e2108296644d5e522abd8900f55e2b1a4f`).
  - `D09` UCI Steel Industry Energy Consumption (CC BY 4.0, SHA-256 `d82d28b33780ff1582507fcf08ae764ff648af459d58234370c551e62aadeaef`, `481,973` bytes; 35,040 15-min rows).
  - `D02` NEU-DET (metadata-only contract `mirror_licence_unverified`).
- **Splits & leakage checks:** 5 stratified seeds `(42, 7, 11, 19, 23)` for `D01`; official 5 MatBench folds for `D04`; strict chronological 80/20 split (sorted timestamps) for `D09`. All post-gate splits pass zero exact train/test overlap checks.

## Features and targets

- **`D01`:** 27 normalized plate/defect inspection features (`Steel_Plate_Thickness` in `mm`) → 7 mutually exclusive fault classes.
- **`D04`:** Normalized elemental atomic fractions from alloy formulas → yield strength (`MPa`).
- **`D09`:** Reactive power (`kVarh`), cyclical `NSM` (`sin`/`cos`), `WeekStatus`, and `Load_Type` (excluding `CO2` and `Power_Factor` to prevent target leakage) → `Usage_kWh` (`kWh`).

## Reproducibility

- **Commands:** `python -m steelgate.cli reproduce`, `python -m steelgate.cli report`, `python -m pytest -v`.
- **Artefacts:** `results/metrics.json`, `results/provenance.json`, `docs/quality_report.html`, `docs/results.md`, `docs/figures/c02_*.svg`.

## Metrics (Before Gate → After Gate, Clean Untouched Test Splits)

| Task | Model | Before Gate | After Gate (`steelgate`) | Clean Oracle |
|---|---|---:|---:|---:|
| `D01` 7-class faults (5 seeds) | `ET + HGB` Soft Voting | 0.7722 ± 0.0152 Acc | **0.7856 ± 0.0110 Acc** (`0.7918` F1) | 0.8103 ± 0.0136 Acc |
| `D01` 7-class faults (5 seeds) | `ExtraTrees` | 0.7474 ± 0.0138 Acc | **0.7711 ± 0.0177 Acc** (`0.7837` F1) | 0.7887 ± 0.0103 Acc |
| `D04` Yield strength (5 folds) | `ExtraTreesRegressor` | 452.20 ± 53.15 MPa MAE | **82.14 ± 10.16 MPa MAE** (`R² = 0.8502`) | 79.09 ± 8.54 MPa MAE |
| `D09` Plant energy (Chron. 80/20) | `HistGradientBoosting` | 2,875.59 kWh MAE | **5.10 kWh MAE** (`R² = 0.9030`) | 5.14 kWh MAE |

## Limitations

- Synthetic corruptions simulate stylized mixed units, outliers, missing values, duplicates, and label flips rather than plant-logged failure streams.
- `D01` `Pastry` and `Other_Faults` remain hardest to separate; `D09` models 2018 load profiles from one facility only.
