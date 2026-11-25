# Data sources

Every dataset used by the portfolio must appear in this file **before** it is
referenced in a figure, model card, or benchmark table. This is enforced during
the weekly Definition of Done review.

| Dataset | Source URL | License / terms | Access date | Citation | Checksum | Used in |
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
| Used in | Sprint ID (`W05`) and epic (`E02`) |

## Planned sources

| Sprint | Dataset | Known constraint |
|---|---|---|
| W05 | UCI Steel Plates Faults | Check UCI terms before redistribution |
| W06 | MatBench `matbench_steels`, NIMS steel samples | Respect benchmark split; record official task version |
| W14–W17 | MatBench tasks | Use official 5-fold splits only |
| W23–W26 | Kaggle Severstal Steel Defect Detection | Competition terms; do not redistribute raw images |
| W40–W44 | NIST AM-Bench | Cite NIST; record measurement IDs |

> API keys and tokens are never committed. Store them in the local environment
> or a secret manager; `.env` files are ignored by Git.
