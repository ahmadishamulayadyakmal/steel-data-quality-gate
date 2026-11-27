# Changelog

## [0.7.0] — 2026-11-27

### Added & Improved
- **Multi-Database Open Data Lake (`37,293` real records, `64` Pandera & physics rules):**
  - Expanded from `D01` (`1,941` plates, `38` rules) to include **MatBench `matbench_steels` (`D04`, `312` alloys, `12` rules `M01`–`M12`)** and **UCI Steel Industry Energy Consumption (`D09`, `35,040` 15-min rows, `14` rules `E01`–`E14`)**.
  - Added real raw-data physics discoveries: `D01` luminosity anomalies (`rows 518 & 591`), `D04` stoichiometric sum deviations (`6` alloys), and `D09` 100% AC power-factor conservation (`<= 0.005%`) + out-of-order `00:00` timestamp detection.
- **Upgraded Quality Gate & Predictive Models:**
  - Deterministic derived-index sensor repair (`repair_uci_derived_indices`), **Northcutt class-adaptive Confident Learning** for `D01`, OOF MAD residual outlier filter for `D04`, and Pint energy unit normalization (`Wh`/`MWh` → `kWh`) for `D09`.
  - Added **5-stage cumulative ablation ladder** and `ExtraTrees + HistGradientBoosting` soft-voting ensemble on `D01` (`0.7474` uncleaned ET → **`0.7856 ± 0.0110`** post-gate ensemble), plus `D04` (`452.20 → 82.14 MPa` MAE) and `D09` (`2,875.59 → 5.10 kWh` MAE) recovery benchmarks and temporal-leakage quantification.
- **Multi-Database Visual Reports:** Upgraded Springer-style SVGs (`c02_scorecard.svg`, `c02_before_after.svg`, `c02_diagnostics.svg`) and `docs/quality_report.html`.

## [0.4.0] — 2026-11-27

- Initial `steelgate` implementation for Case 02 (`D01` 38-rule contract, Pint round-trips, leakage detection, and 5-seed benchmark).

## [0.1.0] — 2026-11-24

- Initial import from the workspace skeleton.
