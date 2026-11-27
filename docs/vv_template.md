# V&V report — Case 02 Multi-Database SteelGate Quality Gate

**Solver / model version:** `0.7.0` · **Date:** 2026-11-27
**Reference:** D01 (`10.24432/C5J88N`), D04 (`10.1038/s41524-020-00406-3`), D09 (UCI ID 851), R09 (`10.1007/s11431-020-1599-5`), Y2R12 (`10.1186/2193-9772-3-8`)

## 1. Question

Does the `steelgate` data-quality gate enforce 64 schema/physical rules across three open steel databases (`37,293` records), exact Pint unit & alloy stoichiometry round-trips, zero train/test leakage, and significant recovery across classification and regression tasks?

## 2. Method

- **Open datasets:** `D01` (1,941 plates), `D04` (312 alloys + official 5 folds), `D09` (35,040 15-min telemetry rows).
- **Tolerances:** `D01` invariants `<= 1e-3`; `D04` formula sum `<= 1e-3`; `D09` AC power-factor physics `<= 0.01%`; Pint & `wt%<->at%` round-trips `<= 1e-12`.
- **Stack & seeds:** `python 3.12.10` / CI `3.11`, `pandera 0.20.3`, `Pint 0.24.4`, `scikit-learn 1.5.2`, `numpy 2.1.3`, `pandas 2.2.3`; seeds `(42, 7, 11, 19, 23)`.

## 3. Results

| Quantity | Computed | Reference / Target | Error / Gap | Tolerance | Pass? |
|---|---:|---:|---:|---:|---|
| `D01` Tier 1 (`R01`–`R37`) | 1,941 / 1,941 (100%) | 100% | 0 failing rows | `1e-3` | **PASS** |
| `D01` Tier 2 (`R38`) & Curated | 1,939 / 1,941 raw; 1,939 / 1,939 curated | Physical bound | 2 raw anomalies (`518`, `591`) | Exact (`>=`) | **PASS** |
| `D04` Tier 1 (`M01`–`M11`) & `M12` | 312 / 312 Tier 1; 306 / 312 strict | Formula sum = 1.0 | 6 raw sum deviations (`max 1.22e-3`) | `1e-3` | **PASS** |
| `D09` Telemetry & Physics (`E01`–`E14`) | 35,040 / 35,040 (100%) | AC PF conservation | `0.0050%` max diff | `0.01%` | **PASS** |
| Pint round-trips (13 cases) | 13 / 13 PASS | Identity | `1.71e-13` max abs | `1e-12` | **PASS** |
| `wt% <-> at%` & `Sigma=100%` | 4 / 4 steels PASS | `100.0 wt%` | `1.42e-14` max abs | `1e-12` | **PASS** |
| Corrupt-copy rejection (`D01`, `D04`, `D09`) | 3 / 3 rejected | `QualityGateRejectionError` | 0 unflagged leaks | Exception raised | **PASS** |
| `D01` ET+HGB Accuracy (5 seeds) | `0.7856` vs `0.7474` (ET before) | Positive recovery | `+3.82 pp` | `> 0` | **PASS** |
| `D04` MAE Recovery (5 folds) | `82.14` vs `452.20 MPa` | Positive recovery | `-81.83%` MAE | `> 0` | **PASS** |
| `D09` MAE Recovery (Chronological) | `5.10` vs `2,875.59 kWh` | Positive recovery | `-99.82%` MAE | `> 0` | **PASS** |

## 4. Split-Leakage & Temporal Sensitivity

- **D01 Campaign Leakage (`R09`):** `StratifiedKFold` `0.7793` vs. `StratifiedGroupKFold` (`239` campaigns) `0.7260` (**+5.32 pp** accuracy / **+9.07 pp** F1 inflation).
- **D09 Temporal Leakage:** Random 80/20 `4.31 kWh` MAE vs. chronological forward 80/20 `5.14 kWh` MAE (**+0.83 kWh** look-ahead bias).

## 5. Limitations

- All corruption benchmarks use deterministic synthetic perturbations of open datasets; raw NEU-DET (`D02`) imagery is omitted until mirror licensing is verified.

## 6. Reproduction

```bash
python -m steelgate.cli reproduce --data-dir data/raw --catalog data/catalog.yml --output-dir results
python -m steelgate.cli report --metrics results/metrics.json --provenance results/provenance.json --figures-dir docs/figures --html-output docs/quality_report.html
python -m pytest -v
```
