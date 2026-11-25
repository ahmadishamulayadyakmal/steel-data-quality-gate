# Data-quality failures in a steel plate mill (C02)

## Industrial problem

A plate mill's inspection database mixes seven fault classes with inconsistent units, missing values, duplicates and mislabelled records. A classifier trained on it learns label noise yet passes internal QA.

**Why it happens.** No schema contract at ingestion, no unit normalisation, no lineage, no drift or leakage check.

## Data (no plant data)

Open data only: UCI Steel Plates Faults (D01: 1,941 plates, 27 features, 7 fault types) and NEU-DET (D02) as an open stand-in for mill camera data. No plant database.

> Only open datasets and published literature are used. No plant or proprietary data are held here; plant numbers quoted from papers are external literature, not independently reproduced results.

## Tool chain

`pandera` schemas, `pint` unit conversion, pandas, SHA-256 checksums, `data/SOURCES.md`, `data/catalog.yml`, `metalkit` typed loaders, HTML quality report.

## Journal references

- **R07** ([DOI](https://doi.org/10.1016/j.commatsci.2018.05.018)) — matminer composition featurization for the extension.
- **R09** ([DOI](https://doi.org/10.1007/s11431-020-1599-5)) — steel feature families and the leakage risk from processing parameters.
- **Y2R12** ([DOI](https://doi.org/10.1186/2193-9772-3-8)) — error-modality warning that motivates the quality gate.
- **D01** ([source](https://archive.ics.uci.edu/dataset/198/steel+plates+faults)) — UCI Steel Plates Faults, licence CC BY 4.0 and checksum.
- **D02** ([source](https://scholar.google.com/)) — NEU-DET provenance and licence.

## Deliverables

20+ schema rules; unit round-trip tests; duplicate/leakage report; before/after model score showing the effect of cleaning; `data/catalog.yml`; `data/SOURCES.md`; HTML quality report.

## Validation & claims

100% of rows pass the schema; unit conversions round-trip; no train/test leakage; all dataset licences and checksums recorded.

## Limitations

UCI is a static research dataset, not a live plant stream. NEU-DET licences vary by mirror and must be verified before use.

## Repository layout

- `README.md` — case summary, data policy, method, results and limitations.
- `CONTRIBUTING.md` — how to propose changes.
- `docs/index.md` — documentation map.
- `docs/model_card.md` and `docs/vv_template.md` — reporting templates.
- `.github/workflows/ci.yml` — automated checks.

Only open data and published literature are used. No plant or proprietary data
are included. All artefacts are written in English.
