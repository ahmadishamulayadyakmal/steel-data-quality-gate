# Changelog

## [0.4.0] — 2026-11-27

### Added
- `steelgate` Python package (`src/steelgate/`) with typed loaders, SHA-256 provenance verification, and `data/catalog.yml` validation.
- **38-rule Pandera schema & physical-invariant contract** (`R01`–`R38`) for UCI Steel Plates Faults (`D01`), plus `neu_det_manifest_v1` camera-inspection contract for `D02`.
- **Pint unit-normalization & stoichiometry module** (`units.py`) with 10 verified physical round-trips (`<= 1.71e-13` error), `Sigma wt% = 100%` alloy checks, and `wt% <-> at%` conversions.
- **Leakage & duplicate auditor** (`leakage.py`) detecting exact duplicates, label conflicts, near-duplicates, and R09 mill-campaign overlap (`239` campaign groups).
- **Synthetic mill-corruption generator & 4-stage quality gate** (`corruption.py`) and 5-seed before/after benchmark (`benchmark.py`).
- **Automated reporting & CI:** Deterministic Springer-style SVG figures (`c02_scorecard.svg`, `c02_before_after.svg`, `c02_diagnostics.svg`), self-contained `docs/quality_report.html`, `docs/results.md`, populated model card & V&V report, 27 `pytest` tests, and cross-OS GitHub Actions CI.

## [0.1.0] — 2026-11-24

- Initial import from the workspace skeleton.
