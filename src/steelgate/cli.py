"""Command-line interface for Case 02 (steelgate)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from steelgate import __version__
from steelgate.benchmark import reproduce
from steelgate.loaders import load_matbench_steels, load_steel_energy, load_uci_faults
from steelgate.provenance import environment_info
from steelgate.report import render_all_reports
from steelgate.schemas import (
    QualityGateRejectionError,
    validate_d04_or_reject,
    validate_d09_or_reject,
    validate_or_reject,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="steelgate",
        description="Multi-database schema-enforced data quality gate and unit normalization (Case 02).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="Print steelgate version and active Python environment.")

    val = sub.add_parser("validate", help="Validate D01, D04, or D09 dataset archive/table.")
    val.add_argument("--input", required=True, help="Path to dataset zip/json.gz/csv.")
    val.add_argument("--dataset", choices=["d01", "d04", "d09"], default="d01", help="Target schema contract.")
    val.add_argument(
        "--strict-physical",
        action="store_true",
        default=False,
        help="Enforce Tier 2 physical conservation rules (R38 / M12 / E14).",
    )

    rep = sub.add_parser("reproduce", help="Run all 3-database audits, unit suite, and benchmarks.")
    rep.add_argument("--data-dir", default="data/raw", help="Directory containing raw datasets.")
    rep.add_argument("--catalog", default="data/catalog.yml", help="Path to data/catalog.yml.")
    rep.add_argument("--output-dir", default="results", help="Directory for metrics.json and provenance.json.")
    rep.add_argument("--seed", type=int, default=42, help="Base deterministic seed.")

    rpt = sub.add_parser("report", help="Render Case 02 SVGs and HTML quality report from metrics.json.")
    rpt.add_argument("--metrics", default="results/metrics.json", help="Path to metrics.json.")
    rpt.add_argument("--provenance", default="results/provenance.json", help="Path to provenance.json.")
    rpt.add_argument("--figures-dir", default="docs/figures", help="Output directory for SVG figures.")
    rpt.add_argument("--html-output", default="docs/quality_report.html", help="Output path for HTML quality report.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "info":
        env = environment_info()
        print(f"steelgate {__version__} — Multi-database data quality gate (Case 02)")
        print(f"Python {env['python']} on {env['platform']}")
        return 0

    if args.command == "validate":
        target = Path(args.input)
        strict = bool(args.strict_physical)
        try:
            if args.dataset == "d01":
                frame = load_uci_faults(target, verify_checksum=target.suffix.lower() == ".zip")
                report = validate_or_reject(frame, strict_physical=strict)
            elif args.dataset == "d04":
                frame = load_matbench_steels(target, verify_checksum=target.name.endswith(".json.gz"))
                report = validate_d04_or_reject(frame, strict_physical=strict)
            else:
                frame = load_steel_energy(target, verify_checksum=target.suffix.lower() == ".zip")
                report = validate_d09_or_reject(frame, strict_physical=strict)
        except QualityGateRejectionError as exc:
            print(exc.report["error_report"])
            return 1
        print(report["error_report"])
        return 0

    if args.command == "reproduce":
        payload = reproduce(
            data_dir=args.data_dir,
            catalog_path=args.catalog,
            output_dir=args.output_dir,
            seed=int(args.seed),
        )
        ens = payload["before_after_benchmark"]["models"]["voting_ensemble"]
        d04_b = payload["d04_benchmark"]["five_fold_benchmark"]
        d09_b = payload["d09_benchmark"]["chronological_before_after"]
        print(
            f"Reproduced Case 02 v0.7.0 (determinism={payload['determinism_check']}, "
            f"{payload['total_real_records']} rows, {payload['total_schema_rules']} rules): "
            f"D01 ET+HGB {ens['before_gate_uncleaned']['accuracy']['mean']:.4f}->{ens['after_gate_cleaned']['accuracy']['mean']:.4f} | "
            f"D04 MAE {d04_b['before_gate']['mae_mpa']['mean']:.2f}->{d04_b['after_gate']['mae_mpa']['mean']:.2f} MPa | "
            f"D09 MAE {d09_b['before_gate']['mae_kwh']:.2f}->{d09_b['after_gate']['mae_kwh']:.2f} kWh"
        )
        return 0

    if args.command == "report":
        outputs = render_all_reports(
            metrics_path=args.metrics,
            provenance_path=args.provenance,
            figures_dir=args.figures_dir,
            html_output=args.html_output,
        )
        for name, path in outputs.items():
            print(f"Wrote {name}: {path}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
