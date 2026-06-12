"""Cyclical indicator store: download from FRED and read from parquet.

Indicators are stored as individual parquet files under
``data/indicators/<name>.parquet`` with a uniform ``(ref_date, value)``
schema.  The :func:`download_indicators` function fetches all series
defined in :data:`~alt_nfp.config.CYCLICAL_INDICATORS` from the FRED API,
and :func:`read_indicator` provides a convenience reader.
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
import polars as pl
from nfp_download.fred import fetch_fred_series
from nfp_lookups.paths import INDICATORS_DIR

logger = logging.getLogger(__name__)

INDICATOR_SCHEMA = {"ref_date": pl.Date, "value": pl.Float64}


def indicator_path(name: str, store_dir: Path = INDICATORS_DIR) -> Path:
    """Return the parquet path for a named indicator."""
    return store_dir / f"{name}.parquet"


def read_indicator(
    name: str,
    store_dir: Path = INDICATORS_DIR,
) -> pl.DataFrame | None:
    """Read an indicator parquet file, or return ``None`` if absent.

    Parameters
    ----------
    name : str
        Indicator name matching a ``CYCLICAL_INDICATORS`` entry
        (e.g. ``'claims'``, ``'jolts'``).
    store_dir : Path
        Root directory for indicator parquet files.

    Returns
    -------
    pl.DataFrame or None
        DataFrame with ``(ref_date, value)`` or ``None``.
    """
    fpath = indicator_path(name, store_dir)
    if not fpath.exists():
        return None
    try:
        return pl.read_parquet(fpath)
    except (OSError, pl.exceptions.ComputeError) as e:
        logger.warning("Failed to read indicator %s from %s: %s", name, fpath, e)
        return None


def download_indicators(
    *,
    indicators: list[dict] | None = None,
    start_date: str = "2000-01-01",
    api_key: str | None = None,
    store_dir: Path = INDICATORS_DIR,
) -> dict[str, int]:
    """Download all cyclical indicators from FRED and write to parquet.

    Parameters
    ----------
    indicators : list[dict], optional
        Indicator specs (each with ``name`` and ``fred_id``).
        Defaults to ``CYCLICAL_INDICATORS``.
    start_date : str
        Observation start date (``YYYY-MM-DD``).
    api_key : str or None
        FRED API key.  Falls back to ``FRED_API_KEY`` env var.
    store_dir : Path
        Target directory for parquet files.

    Returns
    -------
    dict[str, int]
        Mapping of indicator name to row count written.
    """
    if indicators is None:
        from nfp_lookups.provider_config import CYCLICAL_INDICATORS_DEFAULT
        indicators = [
            {"name": ci.name, "fred_id": ci.fred_id, "freq": ci.freq}
            for ci in CYCLICAL_INDICATORS_DEFAULT
        ]
    store_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, int] = {}

    for spec in indicators:
        name = spec["name"]
        fred_id = spec["fred_id"]
        logger.info("Downloading %s (%s) from FRED...", name, fred_id)
        print(f"  Fetching {name} ({fred_id})...")

        try:
            df = fetch_fred_series(
                fred_id, start_date=start_date, api_key=api_key,
            )
        except (httpx.HTTPError, ValueError) as e:
            logger.exception("Failed to download %s (%s): %s", name, fred_id, e)
            print(f"  FAILED: {name}")
            results[name] = 0
            continue

        out_path = indicator_path(name, store_dir)
        df.write_parquet(out_path)
        results[name] = len(df)
        logger.info("Wrote %d rows to %s", len(df), out_path)
        print(f"  Wrote {len(df)} rows → {out_path}")

    return results
