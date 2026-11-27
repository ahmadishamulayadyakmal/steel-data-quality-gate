# Data sources — Case 02

Every dataset used in this repository must appear in this file **before** it
is referenced in a figure, quality report, model card, or benchmark table.
Only public open datasets, published literature, and explicitly labelled
synthetic corruptions derived from open data are used. **No plant or
proprietary data are held or used here.**

| Dataset | Source URL | License / terms | Access date | Citation | Checksum (SHA-256) | Bytes | Used for |
|---|---|---|---|---|---|---:|---|
| UCI Steel Plates Faults (`D01`, `steel_plates_faults.zip`) | https://archive.ics.uci.edu/static/public/198/steel+plates+faults.zip (catalog: https://archive.ics.uci.edu/dataset/198/steel+plates+faults) | CC BY 4.0 | 2026-11-27 | Buscema, M., Terzi, S., & Tastle, W. (2010). *Steel Plates Faults* [Dataset]. UCI ML Repository. DOI: `10.24432/C5J88N` | `cb8eb9859198b63f053e443513036b401746fa517ef58bd17c846c6741c93919` | 100,890 | Raw D01 physical/schema audit (1,941 plates × 27 features + 7 fault classes), R09 mill-campaign leakage CV, and clean benchmark reference |
| NEU-DET Surface Defect Database (`D02`, metadata schema only) | https://scholar.google.com/ (search: `"NEU-DET surface defect database"`) | Mirror-dependent; **verify licence before mirroring** (`mirror_licence_unverified` — raw imagery NOT bundled) | 2026-11-27 | Song, K., & Yan, Y. (2013). *A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects*. Applied Surface Science, 285, 858–864 | — (no raw imagery committed) | — | Pandera camera-inspection manifest schema (`neu_det_manifest_v1`) enforcing 200×200 px bounds, 6 defect classes, SHA-256 hex, and `mirror_licence_verified == True` |
| Synthetic Mill-Corruption Benchmark (derived from `D01`) | Generated deterministically in-memory from `D01` via `steelgate.corruption.corrupt_mill_dataset` (seeded PRNG) | Synthetic derivative of `D01` (CC BY 4.0) | 2026-11-27 | Explicitly **synthetic** — simulates mixed thickness units (`in`/`cm`/`um`), missing values, physical/geometric violations, label-schema defects, 10% duplicate batches, and 12% label noise | Derived from `D01` SHA-256 above | — | Corrupt-copy rejection proof (`QualityGateRejectionError`) and 5-seed before/after quality-gate classifier benchmark |

## Integrity & Provenance Rules

- Raw datasets in `data/raw/` are gitignored; `steelgate reproduce` verifies the exact SHA-256 digest (`cb8eb9859198b63f053e443513036b401746fa517ef58bd17c846c6741c93919`) and byte count (`100890`) of `steel_plates_faults.zip` before execution and aborts on any mismatch.
- Every synthetic dataset or synthetic corruption experiment is explicitly labelled **synthetic** in this manifest, in `results/metrics.json`, in `docs/quality_report.html`, and in all figure captions.
- API keys, `.env` secrets, raw image mirrors, and publisher PDFs are never committed.
