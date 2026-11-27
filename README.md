# Data-quality failures in a steel plate mill & multi-database quality gate (C02)

> **What this project does:** Implements **`steelgate` (v0.7.0)**, a deterministic, schema-enforced data quality gate governing **37,293 open metallurgy records across three databases** (`D01`, `D04`, `D09`) with **64 Pandera & physics rules**, **Pint unit normalization** (length, stress, temperature, energy), **$\Sigma = 100\%$ alloy stoichiometry verification**, **Northcutt Confident Learning**, and **leakage-safe before/after recovery benchmarks**.

Part of the [Computational Metallurgy Portfolio](https://github.com/ahmadishamulayadyakmal/metallurgy-project).

---

## Industrial problem

Steel inspection, alloy composition, and plant energy databases combine categorical fault labels, chemical formulas, and 15-minute meter telemetry—frequently corrupted by mixed engineering units (`mm`/`in`/`cm`/`um`, `MPa`/`ksi`/`GPa`, `kWh`/`Wh`/`MWh`), missing sensors, duplicate batches, label noise, and temporal/campaign leakage. A model trained on uncurated records passes naive internal QA while failing in deployment.

**Why it happens.** No schema or physics contract at ingestion, no unit normalisation, no SHA-256 provenance manifest, and no duplicate, processing-campaign, or temporal leakage checks.

---

## Data (no plant data)

Open data only (**37,293 real records** across three pinned public archives + one metadata-only contract):
- **UCI Steel Plates Faults (`D01`)**: CC BY 4.0, DOI `10.24432/C5J88N`, SHA-256 `cb8eb9859198b63f053e443513036b401746fa517ef58bd17c846c6741c93919` (`1,941` plates × `27` features × `7` fault classes).
- **MatBench `matbench_steels` (`D04`)**: Public benchmark, DOI `10.1038/s41524-020-00406-3`, SHA-256 `473bc4957b2ea5e6465aef84bc29bb48ac34db27d69ea4ec5f508745c6fae252` (`312` steel alloys + official 5-fold splits).
- **UCI Steel Industry Energy Consumption (`D09`)**: CC BY 4.0, UCI ID 851, SHA-256 `d82d28b33780ff1582507fcf08ae764ff648af459d58234370c551e62aadeaef` (`35,040` 15-min plant records).
- **NEU-DET (`D02`)**: Metadata-only camera contract (`mirror_licence_unverified`; no raw imagery bundled).

All corrupt-copy rejection and before/after perturbation experiments use **explicitly synthetic corruptions of `D01`, `D04`, and `D09`** evaluated strictly on **clean, untouched test splits**.

> Only open datasets and published literature are used. No plant or proprietary data are held here; plant numbers quoted from papers are external literature, not independently reproduced results.

---

## Tool chain

`pandera` (64 named rules across `D01`/`D04`/`D09` + `neu_det_manifest_v1`), `pint` unit normalization & $\Sigma = 100\%$ alloy stoichiometry, `scikit-learn` (`ExtraTrees + HistGradientBoosting` soft-voting ensemble, Northcutt Confident Learning, `StratifiedGroupKFold`), SHA-256 checksums, `data/SOURCES.md`, `data/catalog.yml`, `steelgate` CLI, Springer-style SVGs, and self-contained HTML quality report (`docs/quality_report.html`).

---

## Headline Results (v0.7.0)

| Database / Component | Scope | Measured Result (Our Computation) | Verdict |
|---|---|---|---:|
| **`D01` Plates Faults (38 rules)** | 1,941 raw / 1,939 curated plates | Tier 1: `1,941/1,941`; `R38` physics catches raw rows **518 & 591**; Curated: `1,939/1,939` (100%) | **PASS** |
| **`D01` 5-Stage Gate + `ET+HGB`** | 5 seeds, clean untouched test splits | Acc **0.7474** (uncleaned ET) → **0.7711** (gated ET) → **0.7856 ± 0.0110** (`ET+HGB`, **+3.82 pp**); **0** leaked rows | **IMPROVED** |
| **`D04` MatBench Steels (12 rules)** | 312 alloys, official 5-fold CV | `M12` catches **6 raw formula deviations**; MAE **452.20 ± 53.15 → 82.14 ± 10.16 MPa** (**-81.8%**, $R^2 = 0.8502$) | **PASS** |
| **`D09` Steel Energy (14 rules)** | 35,040 15-min rows (2018) | `35,040/35,040` pass `E01`–`E14` (AC PF $\le 0.01\%$); Chronological MAE **2,875.59 → 5.10 kWh** (**-99.8%**) | **PASS** |
| **Pint Units & Alloy Stoichiometry** | 13 round-trips + 4 steels | **13/13 PASS** (`1.71e-13` max err); `wt% <-> at%` err `1.42e-14`; `103.5 wt%` alloy rejected | **PASS** |
| **Corrupt-Copy Rejections** | Synthetic `D01`, `D04`, `D09` copies | All 3 gates raise `QualityGateRejectionError` (`D01`: 21 rules, `D04`: 6 rules, `D09`: 5 rules) | **REJECTED** |
| **R09 & Temporal Leakage Audits** | `D01` (239 campaigns) & `D09` (80/20) | `D01` Campaign gap **+5.32 pp** Acc (`0.7793 → 0.7260`); `D09` Random vs. Forward gap **+0.83 kWh** | **MEASURED** |

![Table 1: Case 02 multi-database quantitative scorecard](docs/figures/c02_scorecard.svg)

![Fig. 1: Multi-database quality-gate recovery (D01, D04, D09) and leakage gaps](docs/figures/c02_before_after.svg)

![Fig. 2: Real raw-data physics anomalies in D01 & D04, plus post-gate ET+HGB 7x7 confusion matrix](docs/figures/c02_diagnostics.svg)

Full ablation tables and diagnostics are in [`docs/results.md`](docs/results.md) and [`docs/quality_report.html`](docs/quality_report.html).

---

## Quick Start & Reproduction

```bash
micromamba create -f environment.yml && micromamba activate steel-data-quality-gate
python -m pip install --no-deps -e .
# Place pinned open archives (D01, D04, D04-splits, D09) into data/raw/ (see data/SOURCES.md)
python -m steelgate.cli validate --input data/raw/steel_plates_faults.zip --dataset d01 --strict-physical
python -m steelgate.cli validate --input data/raw/steel_industry_energy_consumption.zip --dataset d09 --strict-physical
python -m steelgate.cli reproduce --data-dir data/raw --catalog data/catalog.yml --output-dir results
python -m steelgate.cli report --metrics results/metrics.json --provenance results/provenance.json --figures-dir docs/figures --html-output docs/quality_report.html
python -m pytest -v
```

---

## Journal references

- **R07** ([DOI](https://doi.org/10.1016/j.commatsci.2018.05.018)) — Ward et al. (2018), *Computational Materials Science*, 152, 60–69 — composition featurization conventions and descriptor hygiene.
- **R08** ([DOI](https://doi.org/10.1038/s41524-020-00406-3)) — Dunn et al. (2020), *npj Computational Materials*, 6, 138 — official MatBench `matbench_steels` splits (`D04`).
- **R09** ([DOI](https://doi.org/10.1007/s11431-020-1599-5)) — Xiong et al. (2020), *Science China Technological Sciences* — steel feature families and processing-parameter leakage risk (`StratifiedGroupKFold` mill-campaign test).
- **Y2R12** ([DOI](https://doi.org/10.1186/2193-9772-3-8)) — Agrawal et al. (2014), *Integrating Materials and Manufacturing Innovation*, 3, 8 — error-modality confidence stratification motivating quality gates.
- **D01** ([source](https://archive.ics.uci.edu/dataset/198/steel+plates+faults)) — Buscema et al. (2010), UCI Steel Plates Faults (CC BY 4.0, DOI `10.24432/C5J88N`).
- **D02** ([source](https://scholar.google.com/)) — NEU-DET surface defect database provenance and mirror-licence verification contract.
- **D09** ([source](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption)) — Sathishkumar et al. (2021), UCI Steel Industry Energy Consumption (CC BY 4.0).

---

## Validation & claims

- **100% post-gate schema compliance:** Curated `D01` (1,939 rows, 38 rules), `D04` (312 rows, 11 structural/physical rules + `M12` audit), and `D09` (35,040 rows, 14 rules) pass at 100.0% after gating.
- **Exact unit & stoichiometry round-trips:** All 13 physical unit conversions and 4 steel `wt% <-> at%` compositions round-trip within `<= 1.71e-13` absolute error.
- **Zero train/test leakage:** Exact feature/ID/timestamp overlap is `0` across all 5 `D01` seeds, all 5 `D04` MatBench folds, and the `D09` chronological test split.
- **Explicit corrupt-copy rejection:** All three dataset gates raise `QualityGateRejectionError` on corrupted copies.

---

## Limitations

- `D01`, `D04`, and `D09` are static public archives, not live plant streams; synthetic corruptions model stylized rather than plant-logged failure rates.
- Raw NEU-DET (`D02`) imagery is **not** bundled because mirror licences vary and must be verified (`mirror_licence_verified == True`) prior to ingestion.
- Models evaluated here are data-quality benchmarks on public data only and must not be used for production release or structural design without plant validation.

---

## Repository layout

- `src/steelgate/` — multi-database loaders (`D01`/`D04`/`D09`), Pint unit normalizer, 64-rule Pandera contracts, leakage auditor, repair & Confident Learning gate, benchmark suite, SVG/HTML reporter, and CLI.
- `tests/` — 26 unit and 3-database integration tests (`pytest`).
- `data/SOURCES.md` & `data/catalog.yml` — pinned SHA-256 checksums, licences, and canonical units.
- `results/` — deterministic `metrics.json` and `provenance.json`.
- `docs/` — [`results.md`](docs/results.md), [`quality_report.html`](docs/quality_report.html), [`model_card.md`](docs/model_card.md), [`vv_template.md`](docs/vv_template.md), and 3 Springer-style SVGs.
- `.github/workflows/ci.yml` — README hygiene checks and cross-OS (`ubuntu-latest`, `windows-latest`) `pytest` matrix.

Only open data and published literature are used. No plant or proprietary data
are included. All artefacts are written in English.
