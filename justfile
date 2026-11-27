# Task runner: https://github.com/casey/just

default:
    @just --list

test:
    python -m pytest

info:
    python -m steelgate.cli info

validate:
    python -m steelgate.cli validate --input data/raw/steel_plates_faults.zip --strict-physical

reproduce:
    python -m steelgate.cli reproduce --data-dir data/raw --catalog data/catalog.yml --output-dir results

report:
    python -m steelgate.cli report --metrics results/metrics.json --provenance results/provenance.json --figures-dir docs/figures --html-output docs/quality_report.html

all: test reproduce report
    @echo "All Case 02 checks passed."
