"""Deterministic provenance, SHA-256 hashing, and environment capture for Case 02."""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "pandera",
    "pint",
    "PyYAML",
    "joblib",
    "threadpoolctl",
    "pytest",
)


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 hex digest of a file streamed in 1 MiB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(source_dir: str | Path) -> str:
    """Hash all shipped Python modules in ``source_dir`` deterministically."""
    digest = hashlib.sha256()
    for path in sorted(Path(source_dir).glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def environment_info() -> dict[str, Any]:
    """Return the active Python version, OS platform, and tracked package versions."""
    packages: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write deterministic, sorted JSON with a trailing newline."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target
