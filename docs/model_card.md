# Model card — Case 02 SteelGate Plate-Fault Classifier

**Version:** `0.4.0` · **Date:** 2026-11-27

## Intended use

Demonstrates how a schema-enforced data quality gate (`steelgate`: Pint unit normalization, 38-rule Pandera contract, duplicate/leakage removal, and training-only OOF confident-learning noise quarantine) protects a 7-class steel plate fault classifier from ingestion failures. Intended strictly as an open-data reproducibility and data-governance benchmark—**not** as a production mill release model.

## Data

- **Source, licence and checksum:** UCI Steel Plates Faults (`D01`, CC BY 4.0, DOI `10.24432/C5J88N`, SHA-256 `cb8eb9859198b63f053e443513036b401746fa517ef58bd17c846c6741c93919`, `100,890` bytes). NEU-DET (`D02`) is registered as metadata-only (`mirror_licence_unverified`).
- **Number of samples and splits:** 1,941 raw UCI D01 rows; 1,939 curated rows after quarantining real physical luminosity anomalies at rows `518` and `591`. Stratified 80/20 splits across 5 deterministic seeds `(42, 7, 11, 19, 23)`.
- **Synthetic perturbation scope:** Only training splits receive synthetic mill corruptions (`corrupt_mill_dataset`); all evaluation metrics are measured on untouched clean 20% test splits.
- **Leakage checks:** `audit_split_leakage` confirms `0` exact SHA-256 feature overlaps across train/test after the gate on all 5 seeds (compared to `62.6 ± 7.5` leaked duplicates in naive internal QA). Mill-campaign leakage (`StratifiedGroupKFold` across 239 campaigns) is measured separately.

## Features and target

- **Features:** 27 geometric, luminosity, conveyer, steel-grade (`A300`/`A400`), and normalized thickness (`mm` via `pint`) features from UCI D01.
- **Target:** 7 mutually exclusive fault classes (`Pastry`, `Z_Scratch`, `K_Scatch`, `Stains`, `Dirtiness`, `Bumps`, `Other_Faults`).
- **Domain justification:** Physical luminosity bounds (`R38`), bounding-box consistency (`R19`–`R21`, `R25`), and mathematical index invariants (`R27`–`R37`) prevent corrupt inspection records from distorting tree splits.

## Reproducibility

- **Seed, library versions, commands:** Seeds `(42, 7, 11, 19, 23)`, `numpy==2.1.3`, `pandas==2.2.3`, `scikit-learn==1.5.2`, `pandera==0.20.3`, `Pint==0.24.4`. Commands: `python -m steelgate.cli reproduce`, `python -m steelgate.cli report`, `python -m pytest -v`.
- **Artefacts recorded:** `results/metrics.json`, `results/provenance.json`, `docs/quality_report.html`, `docs/results.md`, and 3 SVGs in `docs/figures/`.

## Metrics

| Metric (5 seeds, clean untouched test) | Before Gate (Uncleaned) | After Gate (`steelgate`) | Clean Oracle Reference | Gain (After − Before) |
|---|---:|---:|---:|---:|
| ExtraTrees Accuracy | 0.7474 ± 0.0138 | **0.7696 ± 0.0145** | 0.7887 ± 0.0103 | **+2.22 pp** |
| ExtraTrees Macro F1 | 0.7693 ± 0.0187 | **0.7825 ± 0.0159** | 0.8047 ± 0.0197 | **+1.32 pp** |
| RandomForest Accuracy | 0.7680 ± 0.0094 | **0.7737 ± 0.0152** | 0.7794 ± 0.0226 | **+0.57 pp** |
| RandomForest Macro F1 | 0.7824 ± 0.0192 | **0.7894 ± 0.0220** | 0.7933 ± 0.0233 | **+0.71 pp** |

## Limitations

- **Out-of-distribution behaviour:** Trained and evaluated on static UCI D01 plates; schema thresholds (e.g., `Length_of_Conveyer in [500, 5000]`) reflect this mill line only.
- **Known failure modes:** Minority fault `Pastry` (`0.5596` F1) and `Other_Faults` (`0.7211` F1) remain the hardest classes; generalizing across unseen mill campaigns drops accuracy by `5.32 pp`.
- **What the model must not be used for:** Must **not** be used for automated production plate disposition or safety-critical certification without plant-specific calibration and domain verification.
