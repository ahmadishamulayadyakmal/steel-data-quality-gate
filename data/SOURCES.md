# Data sources

Every dataset used in this repository must appear in this file **before** it
is referenced in a figure, model card or benchmark table.

| Dataset | Source URL | License / terms | Access date | Citation | Checksum | Used for |
|---|---|---|---|---|---|---|
| _No datasets ingested yet_ | — | — | — | — | — | — |

## Required entry format

| Field | Rule |
|---|---|
| Dataset | Exact name as published by the provider; include version/revision if available |
| Source URL | Direct, stable download or catalogue page; no shortened links |
| License / terms | SPDX identifier where possible; note "academic use only" restrictions explicitly |
| Access date | `YYYY-MM-DD` |
| Citation | The provider's required citation, or DOI if no formal citation exists |
| Checksum | SHA-256 of the raw file as downloaded; never of a transformed file |
| Used for | The experiment or result that consumes the dataset |

## Candidate open sources

| Dataset | Known constraint |
|---|---|
| UCI Steel Plates Faults | Check UCI terms before redistribution |
| MatBench `matbench_steels` and other tasks | Respect official folds; record task version |
| NEU-DET surface defects | Verify licence of the chosen mirror |
| Kaggle Severstal Steel Defect Detection | Competition terms; do not redistribute raw images |
| NIST AM-Bench | Cite NIST; record measurement IDs |

> API keys and tokens are never committed. Store them in the local environment
> or a secret manager; `.env` files are ignored by Git.
