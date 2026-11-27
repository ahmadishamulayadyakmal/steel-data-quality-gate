"""Command-line interface for Case 02 (steelgate)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from steelgate import __version__
from steelgate.benchmark import reproduce
from steelgate.loaders import load_uci_faults
from steelgate.provenance import environment_info
from steelgate.report import render_all_reports
from steelgate.schemas import QualityGateRejectionError, validate_or_reject


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="steelgate",
        description="Schema-enforced data quality gate and unit normalization for Case 02.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="Print steelgate version and active Python environment.")

    val = sub.add_parser("validate", help="Validate a UCI D01 zip or CSV inspection table.")
    val.add_argument("--input", required=True, help="Path to steel_plates_faults.zip or CSV file.")
    val.add_argument(
        "--strict-physical",
        action="store_true",
        default=False,
        help="Enforce Tier 2 physical luminosity conservation rule R38.",
    )

    rep = sub.add_parser("reproduce", help="Run all Case 02 audits, unit tests, and 5-seed benchmarks.")
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
        print(f"steelgate {__version__} — Schema-enforced data quality gate (Case 02)")
        print(f"Python {env['python']} on {env['platform']}")
        return 0

    if args.command == "validate":
        target = Path(args.input)
        frame = load_uci_faults(target, verify_checksum=target.suffix.lower() == ".zip")
        try:
            report = validate_or_reject(frame, strict_physical=bool(args.strict_physical))
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
        et = payload["before_after_benchmark"]["models"]["extra_trees"]
        print(
            f"Reproduced Case 02 (determinism={payload['determinism_check']}): "
            f"ExtraTrees Accuracy {et['before_gate_uncleaned']['accuracy']['mean']:.4f} -> "
            f"{et['after_gate_cleaned']['accuracy']['mean']:.4f}"
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
