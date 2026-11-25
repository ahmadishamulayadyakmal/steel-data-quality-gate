# C02 · Data-quality failures in a steel plate mill — W05–W08

> **Real industrial case dossier.** Copied from the authoritative case card in
> `CASES.md` §4 so this repository stays self-contained after it is pushed to
> GitHub. Planning links: [`CASES.md`](https://github.com/ahmadishamulayadyakmal/metallurgy-project/blob/main/CASES.md) ·
> [`JOURNALS.md`](https://github.com/ahmadishamulayadyakmal/metallurgy-project/blob/main/JOURNALS.md) ·
> [`WEEKLY_PLAN.md`](https://github.com/ahmadishamulayadyakmal/metallurgy-project/blob/main/WEEKLY_PLAN.md).
> **Evidence level: A** · No plant data is held in this repository.

**Repository:** `e02-data-lake-quality-gate` · **Evidence: A** ·
**References:** R07, R09, Y2R12, D01, D02.

**Industrial problem.** A plate mill's inspection database mixes seven fault
classes with inconsistent units, missing values, duplicate plates and mislabelled
records. A classifier trained on it learns label noise and passes QA while
misclassifying in production.

**Why it happens.** No schema contract at ingestion, no unit normalisation, no
lineage, no drift check.

**Data (open, no plant data).** UCI Steel Plates Faults (D01: 1,941 plates, 27
features, 7 fault types) and NEU-DET surface-defect images (D02) as an open
stand-in for mill camera data. No plant database is used.

**Tool chain.** `pandera` schemas → `pint` unit conversion → duplicate/leakage
detection → checksum manifest → HTML quality report → `data/SOURCES.md` →
`metalkit` typed loaders.

**What each reference validates.**
- R09 (Xiong 2020) — steel feature families and the danger of leakage from
  processing parameters.
- R07 — matminer composition featurization for the extension.
- Y2R12 — error-modality warning that motivates the quality gate.
- D01, D02 — provenance and licensing of the open datasets.

**Deliverables.** 20+ schema rules; unit round-trip tests; anomaly report;
before/after model score showing the effect of cleaning; `data/catalog.yml`.

**Validation.** 100% of rows pass schema; unit conversions round-trip; no train/
test leakage; all dataset licences recorded.

**Acceptance.** A deliberately corrupted copy of D01 is rejected by the gate with
a readable error report.

## Claim boundary

See [`evidence.md`](evidence.md) for the evidence level and the exact wording
that must be used when quoting literature numbers. All datasets and references
are listed in [`sources.md`](sources.md); the four-week execution plan is in
[`week_plan.md`](week_plan.md).
