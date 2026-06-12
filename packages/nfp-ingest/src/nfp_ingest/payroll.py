"""Payroll provider data ingestion.

Loads payroll provider parquet files with a standard multi-dimensional schema::

    ref_date, geographic_type, geographic_code,
    industry_type, industry_code, employment, [birth_rate]

Filters to the slice specified by :class:`~nfp_lookups.provider_config.ProviderConfig`
and produces flat ``(ref_date, employment[, birth_rate])`` time series for
the model pipeline, as well as PANEL_SCHEMA rows for the observation panel.

Cell-level provider parquets (``geographic_type='region'``) are routed
through :mod:`~nfp_ingest.compositing` for QCEW-weighted national
compositing before entering the standard pipeline.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
from nfp_lookups.paths import DATA_DIR, VINTAGE_STORE_PATH
from nfp_lookups.provider_config import MIN_PSEUDO_ESTABS_PER_CELL, ProviderConfig
from nfp_lookups.schemas import PANEL_SCHEMA, empty_panel

logger = logging.getLogger(__name__)


def _is_cell_level(raw: pl.DataFrame) -> bool:
    """Return True if the provider DataFrame contains cell-level data."""
    if "geographic_type" not in raw.columns:
        return False
    return "region" in raw["geographic_type"].unique().to_list()


def _join_birth_file(series_df: pl.DataFrame, birth_path: Path) -> pl.DataFrame:
    """Left-join a separate birth-rate parquet onto the provider series."""
    if not birth_path.exists():
        logger.warning("Birth-rate file not found: %s", birth_path)
        return series_df
    try:
        bdf = pl.read_parquet(str(birth_path))
    except Exception as exc:
        logger.warning("Failed to read birth-rate file %s: %s", birth_path, exc)
        return series_df
    if "birth_rate" not in bdf.columns or "ref_date" not in bdf.columns:
        logger.warning("birth_rate or ref_date column missing in %s", birth_path)
        return series_df
    births = bdf.select(["ref_date", "birth_rate"]).drop_nulls()
    return series_df.join(births, on="ref_date", how="left")


def read_provider_table(fpath: Path) -> pl.DataFrame | None:
    """Read a provider parquet file; return sorted by ref_date, or None on failure."""
    if not fpath.exists():
        return None
    try:
        raw = pl.read_parquet(str(fpath))
    except Exception as e:
        logger.warning("Failed to read provider file %s: %s", fpath, e)
        return None
    if "ref_date" not in raw.columns:
        logger.warning("Column ref_date not found in %s", fpath)
        return None
    if raw["ref_date"].dtype != pl.Date:
        raw = raw.with_columns(pl.col("ref_date").cast(pl.Date))
    return raw.sort("ref_date")


def load_provider_series(
    config: ProviderConfig,
    *,
    data_dir: Path | None = None,
    store_dir: Path | None = None,
    min_pseudo_estabs: int | None = None,
) -> pl.DataFrame | None:
    """Load and filter a provider file to a flat time series.

    Reads the file at ``data_dir / config.file``, filters to the
    geography/industry slice specified in *config*, and returns a
    DataFrame with columns ``ref_date``, ``employment``, and optionally
    ``birth_rate`` (if present in the source file).

    For cell-level providers (``geographic_type='region'``), QCEW-weighted
    compositing produces a synthetic national employment level.

    Returns ``None`` if the file is missing or the filtered result is empty.
    """
    _data_dir = data_dir if data_dir is not None else DATA_DIR
    _store_dir = store_dir if store_dir is not None else VINTAGE_STORE_PATH
    _min_estabs = min_pseudo_estabs if min_pseudo_estabs is not None else MIN_PSEUDO_ESTABS_PER_CELL

    fpath = _data_dir / config.file
    raw = read_provider_table(fpath)
    if raw is None:
        logger.warning("Provider file not found or unreadable: %s", fpath)
        return None

    if "employment" not in raw.columns:
        logger.warning("Column 'employment' not found in %s. Available: %s", fpath, raw.columns)
        return None

    if _is_cell_level(raw):
        from .compositing import compute_provider_composite

        composite_df, _ = compute_provider_composite(
            raw, _store_dir, _min_estabs,
        )
        if composite_df.is_empty():
            logger.warning("Compositing produced no rows for %s", fpath)
            return None
        result = composite_df.select(["ref_date", "employment"])
    else:
        filter_cols = {
            "geographic_type": config.geography_type,
            "geographic_code": config.geography_code,
            "industry_type": config.industry_type,
            "industry_code": config.industry_code,
        }
        mask = pl.lit(True)
        for col, val in filter_cols.items():
            if col in raw.columns:
                mask = mask & (pl.col(col) == val)

        filtered = raw.filter(mask).sort("ref_date")
        if len(filtered) == 0:
            logger.warning(
                "No rows after filtering %s with %s", fpath, filter_cols
            )
            return None

        keep = ["ref_date", "employment"]
        if "birth_rate" in filtered.columns:
            keep.append("birth_rate")
        result = filtered.select(keep).drop_nulls(subset=["ref_date", "employment"])

    # Join birth rates from a separate file if configured.
    if config.birth_file is not None and "birth_rate" not in result.columns:
        result = _join_birth_file(result, _data_dir / config.birth_file)

    return result


def ingest_provider(
    config: ProviderConfig,
    raw_dir: Path | None = None,
    *,
    data_dir: Path | None = None,
    store_dir: Path | None = None,
    min_pseudo_estabs: int | None = None,
) -> pl.DataFrame:
    """Load a single payroll provider's data into PANEL_SCHEMA format.

    Parameters
    ----------
    config : ProviderConfig
        Provider configuration from nfp_lookups.provider_config.
    raw_dir : Path, optional
        Directory containing raw provider files. Falls back to *data_dir*
        if not provided or if file not found in raw_dir.
    data_dir : Path, optional
        Root data directory. Defaults to ``DATA_DIR``.
    store_dir : Path, optional
        Vintage store directory (for compositing). Defaults to ``VINTAGE_STORE_PATH``.
    min_pseudo_estabs : int, optional
        Minimum pseudo-establishments per cell. Defaults to config constant.

    Returns
    -------
    pl.DataFrame
        Observation panel rows conforming to PANEL_SCHEMA.
    """
    _data_dir = data_dir if data_dir is not None else DATA_DIR
    _store_dir = store_dir if store_dir is not None else VINTAGE_STORE_PATH
    _min_estabs = min_pseudo_estabs if min_pseudo_estabs is not None else MIN_PSEUDO_ESTABS_PER_CELL

    fpath = None
    if raw_dir is not None:
        candidate = raw_dir / config.file
        if candidate.exists():
            fpath = candidate

    if fpath is None:
        fpath = _data_dir / config.file
    raw = read_provider_table(fpath)
    if raw is None:
        logger.warning("Provider file not found or unreadable: %s", fpath)
        return empty_panel()

    if "employment" not in raw.columns:
        logger.warning(
            "Column 'employment' not found in %s. Available: %s", fpath, raw.columns
        )
        return empty_panel()

    if _is_cell_level(raw):
        from .compositing import compute_provider_composite

        composite_df, _ = compute_provider_composite(
            raw, _store_dir, _min_estabs,
        )
        if composite_df.is_empty():
            return empty_panel()
        raw = composite_df.sort("ref_date")
    else:
        filter_cols = {
            "geographic_type": config.geography_type,
            "geographic_code": config.geography_code,
            "industry_type": config.industry_type,
            "industry_code": config.industry_code,
        }
        mask = pl.lit(True)
        for col, val in filter_cols.items():
            if col in raw.columns:
                mask = mask & (pl.col(col) == val)
        raw = raw.filter(mask).sort("ref_date")

    raw = raw.select(["ref_date", "employment"]).drop_nulls()

    if len(raw) < 2:
        return empty_panel()

    ref_dates = raw["ref_date"].to_list()
    levels = raw["employment"].to_numpy().astype(float)

    rows: list[dict] = []
    source_name = config.name.lower()

    for i in range(1, len(levels)):
        if (
            levels[i] > 0
            and levels[i - 1] > 0
            and np.isfinite(levels[i])
            and np.isfinite(levels[i - 1])
        ):
            growth = float(np.log(levels[i]) - np.log(levels[i - 1]))

            ref_dt = ref_dates[i]
            vintage_dt = ref_dt + timedelta(weeks=3)
            lag = (
                (vintage_dt.year - ref_dt.year) * 12
                + vintage_dt.month
                - ref_dt.month
            )

            rows.append(
                {
                    "period": ref_dt,
                    "geographic_type": config.geography_type,
                    "geographic_code": config.geography_code,
                    "industry_code": config.industry_code,
                    "industry_level": config.industry_type,
                    "source": source_name,
                    "source_type": "payroll",
                    "growth": growth,
                    "employment_level": float(levels[i]),
                    "is_seasonally_adjusted": False,
                    "vintage_date": vintage_dt,
                    "revision_number": 0,
                    "is_final": True,
                    "publication_lag_months": lag,
                    "coverage_ratio": None,
                }
            )

    if not rows:
        return empty_panel()

    return pl.DataFrame(rows, schema=PANEL_SCHEMA)


