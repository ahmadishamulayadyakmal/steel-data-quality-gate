# Data-quality failures in a steel plate mill (C02)

> Workspace folder: `e02-data-lake-quality-gate` · Suggested GitHub repository: `metallurgy-data-lake`
> Epic: E02 · Weeks: W05–W08 · Evidence level: **A**
> Master specification: [`CASES.md`](https://github.com/ahmadishamulayadyakmal/metallurgy-project/blob/main/CASES.md) §4 · Download the papers: [`JOURNALS.md`](https://github.com/ahmadishamulayadyakmal/metallurgy-project/blob/main/JOURNALS.md)

## Industrial problem

A plate mill's inspection database mixes seven fault classes with inconsistent units, missing values, duplicates and mislabelled records. A classifier trained on it learns label noise yet passes internal QA.

**Why it happens.** No schema contract at ingestion, no unit normalisation, no lineage, no drift or leakage check.

## Data (no plant data)

Open data only: UCI Steel Plates Faults (D01: 1,941 plates, 27 features, 7 fault types) and NEU-DET (D02) as an open stand-in for mill camera data. No plant database. Evidence level A.

> The author cannot supply factory data. This project uses **only open datasets and
> credible peer-reviewed journals**. If this is a B-level case, the case report
> must contain the exact disclaimer from `CASES.md` §1 and must **never** claim
> plant validation.

## Tool chain

`pandera` schemas, `pint` unit conversion, pandas, SHA-256 checksums, `data/SOURCES.md`, `data/catalog.yml`, `metalkit` typed loaders, HTML quality report.

## Journal references

- **R07** — matminer composition featurization for the extension.
- **R09** — steel feature families and the leakage risk from processing parameters.
- **Y2R12** — error-modality warning that motivates the quality gate.
- **D01** — UCI Steel Plates Faults, licence CC BY 4.0 and checksum.
- **D02** — NEU-DET provenance and licence.

Each reference above is listed with its DOI or stable URL, access status and
manual download route in [`JOURNALS.md`](https://github.com/ahmadishamulayadyakmal/metallurgy-project/blob/main/JOURNALS.md) §4, and in the shared
BibTeX library `papers/references.bib`. Do not cite a paper you have not been
able to obtain; if it is inaccessible, record that in `JOURNALS.md` §8.

## Deliverables

20+ schema rules; unit round-trip tests; duplicate/leakage report; before/after model score showing the effect of cleaning; `data/catalog.yml`; `data/SOURCES.md`; HTML quality report.

## Validation & claims

100% of rows pass the schema; unit conversions round-trip; no train/test leakage; all dataset licences and checksums recorded.

## Limitations

UCI is a static research dataset, not a live plant stream. NEU-DET licences vary by mirror and must be verified before use.

## Definition of Done

1 merged PR · green tests · 1 new artefact · 1 README paragraph · 1 sprint note
`docs/sprints/wNN.md` · tag `wNN`. Before the case tag `c02` is created,
this README must contain all six sections: Industrial problem, Data (no plant
data), Tool chain, Journal references, Validation & claims, Limitations.

## Repository skeleton

Shared files: `CONTRIBUTING.md` (sprint contract), `.github/` (issue, PR and
CI templates), `docs/sprints/wNN.md`, `docs/vv_template.md` and
`docs/model_card.md`. `CASES.md`, `JOURNALS.md` and `REPOSITORIES.md` are in a
separate planning repository; their links above remain valid after this folder
is pushed as its own repository.

The real industrial case dossier is in `case/`: `README.md` (full case card),
`evidence.md`, `sources.md` and `week_plan.md`.

## Push this folder to GitHub

This folder is prepared but is **not** a Git repository yet:

```bash
cd e02-data-lake-quality-gate
git init
git branch -M main
git remote add origin https://github.com/<your-user>/metallurgy-data-lake.git
git add .
git commit -m "Initial import: C02 · W05–W08"
git push -u origin main
```

Use weekly tags `wNN` and one case tag `c02` inside this repository.
Never commit API keys, raw downloaded datasets or publisher PDFs.

## Language

All artefacts in this repository are written in **English**.
