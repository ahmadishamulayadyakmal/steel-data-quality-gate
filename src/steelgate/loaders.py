"""Typed dataset loaders, catalog reader, and SHA-256 integrity checks for Case 02."""

from __future__ import annotations

import gzip
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

D09_ENERGY_COLUMNS: tuple[str, ...] = (
    "date",
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
    "WeekStatus",
    "Day_of_week",
    "Load_Type",
)

UCI_D01_META: dict[str, Any] = {
    "id": "D01",
    "file": "steel_plates_faults.zip",
    "sha256": "cb8eb9859198b63f053e443513036b401746fa517ef58bd17c846c6741c93919",
    "bytes": 100890,
    "license": "CC BY 4.0",
    "doi": "10.24432/C5J88N",
    "url": "https://archive.ics.uci.edu/static/public/198/steel+plates+faults.zip",
}

D04_STEELS_META: dict[str, Any] = {
    "id": "D04",
    "file": "matbench_steels.json.gz",
    "sha256": "473bc4957b2ea5e6465aef84bc29bb48ac34db27d69ea4ec5f508745c6fae252",
    "bytes": 8836,
    "license": "Public benchmark (Dunn et al., 2020)",
    "doi": "10.1038/s41524-020-00406-3",
    "url": "https://github.com/materialsproject/matbench/blob/main/benchmarks/matbench_v0.1_Atomic_Affine_QRF/inputs/matbench_steels.json.gz",
}

D04_SPLITS_META: dict[str, Any] = {
    "id": "D04-splits",
    "file": "steels_splits.json",
    "sha256": "ef3bc6cf09ec08a23a8d0a6f1e04c0e2108296644d5e522abd8900f55e2b1a4f",
    "bytes": 42676,
    "license": "Public benchmark (Dunn et al., 2020)",
    "doi": "10.1038/s41524-020-00406-3",
    "url": "https://github.com/materialsproject/matbench/blob/main/benchmarks/matbench_v0.1_Atomic_Affine_QRF/inputs/steels_splits.json",
}

D09_ENERGY_META: dict[str, Any] = {
    "id": "D09",
    "file": "steel_industry_energy_consumption.zip",
    "sha256": "d82d28b33780ff1582507fcf08ae764ff648af459d58234370c551e62aadeaef",
    "bytes": 481973,
    "license": "CC BY 4.0",
    "url": "https://archive.ics.uci.edu/static/public/851/steel+industry+energy+consumption.zip",
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


def verify_dataset_file(path: str | Path, meta: dict[str, Any]) -> dict[str, Any]:
    """Verify SHA-256 digest and byte size of a pinned raw file against ``meta``."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(
            f"Missing {target.name} at {target}. Download {meta['id']} per data/SOURCES.md."
        )
    actual_hash = sha256_file(target)
    actual_bytes = target.stat().st_size
    if actual_hash != meta["sha256"] or actual_bytes != meta["bytes"]:
        raise ValueError(
            f"Checksum/size mismatch for {target.name}: expected "
            f"{meta['sha256']} / {meta['bytes']} bytes, got "
            f"{actual_hash} / {actual_bytes} bytes"
        )
    out = {
        "id": meta["id"],
        "file": target.name,
        "sha256": actual_hash,
        "bytes": actual_bytes,
        "license": meta["license"],
        "url": meta["url"],
    }
    if "doi" in meta:
        out["doi"] = meta["doi"]
    return out


def verify_d01_archive(path: str | Path) -> dict[str, Any]:
    """Verify the SHA-256 digest and byte size of the pinned UCI D01 zip archive."""
    return verify_dataset_file(path, UCI_D01_META)


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


def load_matbench_steels(path: str | Path, *, verify_checksum: bool = True) -> pd.DataFrame:
    """Load MatBench ``matbench_steels`` (D04) with stable ``mb-steels-NNN`` IDs."""
    target = Path(path)
    if verify_checksum:
        verify_dataset_file(target, D04_STEELS_META)
    with gzip.open(target, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)

    frame = pd.DataFrame(payload["data"], columns=payload["columns"])
    if list(frame.columns) != ["composition", "yield strength"]:
        raise ValueError(f"Unexpected MatBench D04 columns: {list(frame.columns)}")
    frame.insert(0, "id", [f"mb-steels-{i:03d}" for i in range(1, len(frame) + 1)])
    frame = frame.rename(columns={"yield strength": "yield_strength_mpa"})
    if len(frame) != 312:
        raise ValueError(f"Unexpected MatBench D04 row count: {len(frame)}")
    return frame


def load_matbench_splits(path: str | Path, *, verify_checksum: bool = True) -> dict[str, dict[str, list[str]]]:
    """Load official 5-fold cross-validation splits for ``matbench_steels`` (D04)."""
    target = Path(path)
    if verify_checksum:
        verify_dataset_file(target, D04_SPLITS_META)
    payload = json.loads(target.read_text(encoding="utf-8"))
    metadata = payload["metadata"]
    if metadata != {"n_splits": 5, "random_state": 18012019, "shuffle": True}:
        raise ValueError(f"Unexpected MatBench split metadata: {metadata}")
    return payload["splits"]["matbench_steels"]


def load_steel_energy(path: str | Path, *, verify_checksum: bool = True) -> pd.DataFrame:
    """Load UCI Steel Industry Energy Consumption (D09) from zip or CSV."""
    target = Path(path)
    if target.suffix.lower() == ".csv":
        return pd.read_csv(target)

    if verify_checksum:
        verify_dataset_file(target, D09_ENERGY_META)

    with zipfile.ZipFile(target) as archive:
        csv_names = [n for n in archive.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV found inside {target}")
        frame = pd.read_csv(io.BytesIO(archive.read(csv_names[0])))

    if tuple(frame.columns) != D09_ENERGY_COLUMNS:
        raise ValueError(f"Unexpected UCI D09 columns: {list(frame.columns)}")
    if len(frame) != 35040:
        raise ValueError(f"Unexpected UCI D09 row count: {len(frame)}")
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
