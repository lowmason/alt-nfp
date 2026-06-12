"""Canonical data directory layout for the NFP project.

All path constants are derived from a single BASE_DIR (the repository root).
Other packages import these paths rather than constructing their own.

Set the ``NFP_BASE_DIR`` environment variable (before first import) to point
the layout at a different root, e.g. a snapshot or fixture directory.
"""

from __future__ import annotations

import os
from pathlib import Path


def _find_base_dir() -> Path:
    """Resolve the workspace root.

    Precedence: the ``NFP_BASE_DIR`` environment variable, then walking up
    from this file to the first directory containing both ``packages/`` and
    ``pyproject.toml`` (committed markers — ``data/`` is gitignored, so it
    cannot be relied on after a fresh clone), then a fixed-depth fallback
    for editable installs.
    """
    env = os.environ.get("NFP_BASE_DIR")
    if env:
        return Path(env).expanduser().resolve()

    candidate = Path(__file__).resolve()
    for parent in candidate.parents:
        if (parent / "packages").is_dir() and (parent / "pyproject.toml").is_file():
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

# Vintage store root (alias of STORE_DIR; named for the artifact it holds)
VINTAGE_STORE_PATH: Path = STORE_DIR

# Release-dates pipeline artifacts (scraped BLS schedules → parquet)
RELEASES_DIR: Path = DOWNLOADS_DIR / "releases"
RELEASE_DATES_PATH: Path = INTERMEDIATE_DIR / "release_dates.parquet"
VINTAGE_DATES_PATH: Path = INTERMEDIATE_DIR / "vintage_dates.parquet"
