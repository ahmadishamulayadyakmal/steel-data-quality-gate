# V&V report — Case 02 SteelGate Data Quality Gate

**Solver / model version:** `0.4.0` · **Date:** 2026-11-27
**Reference:** D01 (`10.24432/C5J88N`), R09 (`10.1007/s11431-020-1599-5`), Y2R12 (`10.1186/2193-9772-3-8`)

## 1. Question

Does the `steelgate` data-quality gate enforce 100% schema and physical consistency, exact unit round-trips, zero train/test duplicate leakage, and measurable recovery of downstream fault classification accuracy when corrupted inspection records are ingested?

## 2. Method

- **Open dataset & synthetic benchmark:** Unmodified UCI Steel Plates Faults (`D01`, 1,941 rows) for physical/schema auditing and R09 mill-campaign CV; deterministic seeded synthetic corruptions of D01 (`corrupt_mill_dataset`) for corrupt-copy rejection and 5-seed before/after evaluation on clean untouched 20% test splits.
- **Sample points:** 1,941 raw D01 rows; 1,939 curated D01 rows; 2,133 synthetically corrupted rows; 10 Pint unit round-trips; 4 reference steel compositions (`Sigma = 100%`).
- **Tolerances:** Invariant checks `<= 1e-3` (matching 4-decimal UCI rounding); unit & composition round-trips `<= 1e-12`; mass balance `<= 0.05 wt%`.
- **Library versions & seeds:** `python 3.12.10` / CI `3.11`, `pandera 0.20.3`, `Pint 0.24.4`, `scikit-learn 1.5.2`, `numpy 2.1.3`, `pandas 2.2.3`; seeds `(42, 7, 11, 19, 23)`.

## 3. Results

| Quantity | Computed | Reference / Target | Error / Gap | Tolerance | Pass? |
|---|---:|---:|---:|---:|---|
| Tier 1 D01 rules (`R01`–`R37`) | 1,941 / 1,941 (100%) | 1,941 / 1,941 | 0 failing rows | `1e-3` math tol | **PASS** |
| Tier 2 D01 luminosity (`R38`) | 1,939 / 1,941 | Physical bound | 2 real anomalies (`518`, `591`) | Exact (`>=`) | **DETECTED** |
| Curated D01 (`R01`–`R38`) | 1,939 / 1,939 (100%) | 100% schema pass | 0 failing rows | 0 failures | **PASS** |
| Pint unit round-trips (10 cases) | 10 / 10 PASS | Exact identity | `1.71e-13` max abs | `1e-12` | **PASS** |
| `wt% <-> at%` & `Sigma=100%` (4 steels) | 4 / 4 PASS | `100.0 wt%` | `1.42e-14` max abs | `0.05 wt%` / `1e-12` | **PASS** |
| Corrupt-copy rejection (`validate_or_reject`) | Rejected (21 rules) | Must reject | 0 unflagged leaks | Rejection error | **PASS** |
| Post-gate train/test exact leakage | 0 / 5 seeds | 0 overlap | 0 leaked rows | 0 rows | **PASS** |
| ExtraTrees Accuracy (After − Before) | `0.7696` vs `0.7474` | Positive recovery | `+2.22 pp` (`+1.32 pp` F1) | `> 0` | **PASS** |

## 4. Convergence & Split Sensitivity

- **R09 Mill-Campaign Leakage Check:** 5-fold `StratifiedKFold` yields `0.7793 ± 0.0109` accuracy (`0.7984 ± 0.0093` macro F1) vs. 5-fold `StratifiedGroupKFold` across 239 campaigns at `0.7260 ± 0.0536` accuracy (`0.7077 ± 0.0740` macro F1) — an optimistic inflation of **+5.32 pp** accuracy and **+9.07 pp** macro F1 when processing-campaign proxies overlap between train and validation folds.

## 5. Limitations

- **Where the comparison is weak:** UCI D01 is static; synthetic corruption modes (18% mixed thickness units, 4% NaNs, 3% physics errors, 2% label-schema defects, 12% label noise, 10% duplicates) model stylized rather than plant-logged failure rates.
- **What was not tested:** Live streaming ingestion, and raw NEU-DET (`D02`) image files (metadata schema only until a mirror licence is verified).

## 6. Reproduction

```bash
python -m steelgate.cli reproduce --data-dir data/raw --catalog data/catalog.yml --output-dir results
python -m steelgate.cli report --metrics results/metrics.json --provenance results/provenance.json --figures-dir docs/figures --html-output docs/quality_report.html
python -m pytest -v
```
