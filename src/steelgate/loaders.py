"""Typed dataset loaders, catalog reader, and SHA-256 integrity checks for Case 02."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from steelgate.provenance import sha256_file

UCI_FEATURE_COLUMNS: tuple[str, ...] = (
    "X_Minimum",
    "X_Maximum",
    "Y_Minimum",
    "Y_Maximum",
    "Pixels_Areas",
    "X_Perimeter",
    "Y_Perimeter",
    "Sum_of_Luminosity",
    "Minimum_of_Luminosity",
    "Maximum_of_Luminosity",
    "Length_of_Conveyer",
    "TypeOfSteel_A300",
    "TypeOfSteel_A400",
    "Steel_Plate_Thickness",
    "Edges_Index",
    "Empty_Index",
    "Square_Index",
    "Outside_X_Index",
    "Edges_X_Index",
    "Edges_Y_Index",
    "Outside_Global_Index",
    "LogOfAreas",
    "Log_X_Index",
    "Log_Y_Index",
    "Orientation_Index",
    "Luminosity_Index",
    "SigmoidOfAreas",
)

UCI_FAULT_COLUMNS: tuple[str, ...] = (
    "Pastry",
    "Z_Scratch",
    "K_Scatch",
    "Stains",
    "Dirtiness",
    "Bumps",
    "Other_Faults",
)

UCI_ALL_COLUMNS: tuple[str, ...] = UCI_FEATURE_COLUMNS + UCI_FAULT_COLUMNS

UCI_D01_META: dict[str, Any] = {
    "id": "D01",
    "file": "steel_plates_faults.zip",
    "sha256": "cb8eb9859198b63f053e443513036b401746fa517ef58bd17c846c6741c93919",
    "bytes": 100890,
    "license": "CC BY 4.0",
    "doi": "10.24432/C5J88N",
    "url": "https://archive.ics.uci.edu/static/public/198/steel+plates+faults.zip",
}


def load_catalog(catalog_path: str | Path) -> dict[str, Any]:
    """Load and minimally validate ``data/catalog.yml``."""
    path = Path(catalog_path)
    if not path.is_file():
        raise FileNotFoundError(f"Catalog file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "datasets" not in payload:
        raise ValueError(f"Invalid catalog structure in {path}: missing 'datasets' key")
    if not isinstance(payload["datasets"], list) or not payload["datasets"]:
        raise ValueError(f"Catalog in {path} must declare a non-empty 'datasets' list")
    return payload


def verify_d01_archive(path: str | Path) -> dict[str, Any]:
    """Verify the SHA-256 digest and byte size of the pinned UCI D01 zip archive."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(
            f"Missing {target.name} at {target}. Download UCI D01 per data/SOURCES.md."
        )
    actual_hash = sha256_file(target)
    actual_bytes = target.stat().st_size
    if actual_hash != UCI_D01_META["sha256"] or actual_bytes != UCI_D01_META["bytes"]:
        raise ValueError(
            f"Checksum/size mismatch for {target.name}: expected "
            f"{UCI_D01_META['sha256']} / {UCI_D01_META['bytes']} bytes, got "
            f"{actual_hash} / {actual_bytes} bytes"
        )
    return {
        "id": UCI_D01_META["id"],
        "file": target.name,
        "sha256": actual_hash,
        "bytes": actual_bytes,
        "license": UCI_D01_META["license"],
        "doi": UCI_D01_META["doi"],
        "url": UCI_D01_META["url"],
    }


def load_uci_faults(path: str | Path, *, verify_checksum: bool = True) -> pd.DataFrame:
    """Load UCI Steel Plates Faults (D01) from zip or CSV into a typed DataFrame."""
    target = Path(path)
    if target.suffix.lower() == ".csv":
        return pd.read_csv(target)

    if verify_checksum:
        verify_d01_archive(target)

    with zipfile.ZipFile(target) as archive:
        names = [
            line.strip()
            for line in archive.read("Faults27x7_var").decode("latin-1").splitlines()
            if line.strip()
        ]
        payload = io.StringIO(archive.read("Faults.NNA").decode("latin-1"))
        frame = pd.read_csv(payload, sep="\t", header=None, names=names)

    if tuple(frame.columns) != UCI_ALL_COLUMNS:
        raise ValueError(f"Unexpected UCI D01 columns: {list(frame.columns)}")
    if frame.shape != (1941, 34):
        raise ValueError(f"Unexpected UCI D01 shape: {frame.shape}")
    return frame


def load_neu_det_manifest(path: str | Path) -> pd.DataFrame:
    """Load a NEU-DET (D02) camera-inspection metadata manifest from CSV or JSON."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"NEU-DET manifest file not found: {target}")
    if target.suffix.lower() == ".json":
        data = json.loads(target.read_text(encoding="utf-8"))
        return pd.DataFrame(data)
    return pd.read_csv(target)
