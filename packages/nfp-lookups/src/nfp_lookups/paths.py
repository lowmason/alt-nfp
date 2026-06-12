"""Canonical data directory layout for the NFP project.

All path constants are derived from a single BASE_DIR (the repository root).
Other packages import these paths rather than constructing their own.
"""

from __future__ import annotations

from pathlib import Path


def _find_base_dir() -> Path:
    """Walk up from this file to find the monorepo root.

    Heuristic: the root contains a ``pyproject.toml`` with a ``data/`` directory.
    """
    candidate = Path(__file__).resolve()
    for parent in candidate.parents:
        if (parent / "data").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    # Fallback: four levels above packages/nfp-lookups/src/nfp_lookups/
    return Path(__file__).resolve().parents[4]


BASE_DIR: Path = _find_base_dir()
DATA_DIR: Path = BASE_DIR / "data"
STORE_DIR: Path = DATA_DIR / "store"
DOWNLOADS_DIR: Path = DATA_DIR / "downloads"
INTERMEDIATE_DIR: Path = DATA_DIR / "intermediate"
INDICATORS_DIR: Path = DATA_DIR / "indicators"
OUTPUT_DIR: Path = BASE_DIR / "output"
