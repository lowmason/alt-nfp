"""CES National data ingestion: BLS download → PANEL_SCHEMA.

CES (Current Employment Statistics) provides monthly employment estimates
from a survey of ~670,000 establishments, with a 3-revision cycle
(first print, second, third) plus annual benchmark revision.

This module replaces the CES-related logic previously in ``ces.py``,
using the new ``bls/`` download layer instead of eco-stats.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from nfp_download.bls import BLSHttpClient
from nfp_download.bls import fetch_ces_national as bls_fetch_ces_national
from nfp_lookups.schemas import CES_VINTAGE_SCHEMA, PANEL_SCHEMA, empty_panel

logger = logging.getLogger(__name__)


def fetch_ces_current(
    start_year: int,
    end_year: int,
    client: BLSHttpClient | None = None,
) -> pl.DataFrame:
    """Fetch current CES national data from BLS via the bls/ download layer.

    Parameters
    ----------
    start_year : int
        First year to fetch.
    end_year : int
        Last year to fetch (inclusive).
    client : BLSHttpClient or None
        Optional HTTP client.

    Returns
    -------
    pl.DataFrame
        Observation panel rows conforming to PANEL_SCHEMA.
    """
    try:
        raw = bls_fetch_ces_national(
            start_year=start_year,
            end_year=end_year,
            client=client,
        )
    except Exception as e:
        logger.warning(f'Failed to fetch CES national data: {e}')
        return empty_panel()

    if len(raw) == 0:
        return empty_panel()

    today = date.today()
    rows: list[dict] = []

    # Group by (supersector_code, is_seasonally_adjusted) to compute growth
    for (ss_code, is_sa), group in raw.group_by(
        ['supersector_code', 'is_seasonally_adjusted'],
        maintain_order=True,
    ):
        group = group.sort('date').drop_nulls(subset=['date', 'value'])
        if len(group) < 2:
            continue

        dates = group['date'].to_list()
        values = group['value'].to_numpy().astype(float)

        source = 'ces_sa' if is_sa else 'ces_nsa'
        source_type = 'official_sa' if is_sa else 'official_nsa'

        for i in range(1, len(values)):
            if values[i] > 0 and values[i - 1] > 0:
                growth = float(np.log(values[i]) - np.log(values[i - 1]))
                ref_dt = dates[i]

                months_ago = (
                    (today.year - ref_dt.year) * 12
                    + today.month - ref_dt.month
                )
                if months_ago <= 1:
                    rev_num = 0
                elif months_ago <= 2:
                    rev_num = 1
                elif months_ago <= 3:
                    rev_num = 2
                else:
                    rev_num = -1  # benchmark

                rows.append(
                    {
                        'period': ref_dt,
                        'geographic_type': 'national',
                        'geographic_code': 'US',
                        'industry_code': str(ss_code),
                        'industry_level': 'supersector',
                        'source': source,
                        'source_type': source_type,
                        'growth': growth,
                        'employment_level': float(values[i]),
                        'is_seasonally_adjusted': bool(is_sa),
                        'vintage_date': today,
                        'revision_number': rev_num,
                        'is_final': months_ago > 13,
                        'publication_lag_months': months_ago,
                        'coverage_ratio': None,
                    }
                )

    if not rows:
        return empty_panel()

    return pl.DataFrame(rows, schema=PANEL_SCHEMA)


def load_ces_vintages(path: Path) -> pl.DataFrame:
    """Load historical CES revision data from a parquet file.

    Parameters
    ----------
    path : Path
        Path to ces_vintages.parquet conforming to CES_VINTAGE_SCHEMA.

    Returns
    -------
    pl.DataFrame
        Observation panel rows conforming to PANEL_SCHEMA.
    """
    if not path.exists():
        logger.info(f'CES vintage file not found: {path}')
        return empty_panel()

    raw = pl.read_parquet(path)

    for col in CES_VINTAGE_SCHEMA:
        if col not in raw.columns:
            raise ValueError(f'CES vintage parquet missing column: {col}')

    rows: list[dict] = []

    for (ss_code, sa_flag, rev_num), group in raw.group_by(
        ['supersector_code', 'seasonal_adjustment', 'revision_number'],
        maintain_order=True,
    ):
        group = group.sort('ref_date')
        emp = group['employment'].to_numpy().astype(float)
        ref_dates = group['ref_date'].to_list()
        vintage_dates = group['vintage_date'].to_list()

        is_sa = sa_flag == 'S'
        source = 'ces_sa' if is_sa else 'ces_nsa'
        source_type = 'official_sa' if is_sa else 'official_nsa'
        is_final = rev_num == -1

        for i in range(1, len(emp)):
            if emp[i] > 0 and emp[i - 1] > 0:
                growth = float(np.log(emp[i]) - np.log(emp[i - 1]))
                vdate = vintage_dates[i]
                lag = (
                    (vdate.year - ref_dates[i].year) * 12
                    + vdate.month
                    - ref_dates[i].month
                )

                rows.append(
                    {
                        'period': ref_dates[i],
                        'geographic_type': 'national',
                        'geographic_code': 'US',
                        'industry_code': str(ss_code),
                        'industry_level': 'supersector',
                        'source': source,
                        'source_type': source_type,
                        'growth': growth,
                        'employment_level': float(emp[i]),
                        'is_seasonally_adjusted': is_sa,
                        'vintage_date': vdate,
                        'revision_number': int(rev_num),
                        'is_final': is_final,
                        'publication_lag_months': lag,
                        'coverage_ratio': None,
                    }
                )

    if not rows:
        return empty_panel()

    return pl.DataFrame(rows, schema=PANEL_SCHEMA)


def ingest_ces_national(
    vintage_dir: Path | None = None,
    start_year: int = 2003,
    end_year: int | None = None,
    client: BLSHttpClient | None = None,
) -> pl.DataFrame:
    """Orchestrate CES national data ingestion from BLS and historical vintages.

    Parameters
    ----------
    vintage_dir : Path, optional
        Directory containing ces_vintages.parquet.
    start_year : int
        First year for API fetch (default 2003).
    end_year : int, optional
        Last year for API fetch. Defaults to current year.
    client : BLSHttpClient or None
        Optional HTTP client for BLS downloads.

    Returns
    -------
    pl.DataFrame
        Combined and deduplicated observation panel.
    """
    if end_year is None:
        end_year = date.today().year

    parts: list[pl.DataFrame] = []

    # Load historical vintages if available
    if vintage_dir is not None:
        vintage_path = vintage_dir / 'ces_vintages.parquet'
        vintage_df = load_ces_vintages(vintage_path)
        if len(vintage_df) > 0:
            parts.append(vintage_df)

    # Fetch current data via bls/ download layer
    try:
        current_df = fetch_ces_current(start_year, end_year, client=client)
        if len(current_df) > 0:
            parts.append(current_df)
    except Exception as e:
        logger.warning(f'Failed to fetch current CES data: {e}')

    if not parts:
        return empty_panel()

    combined = pl.concat(parts)

    # Deduplicate: prefer vintage data (explicit revision tracking) over API data
    combined = (
        combined.sort('revision_number', descending=True)
        .unique(
            subset=[
                'period', 'geographic_type', 'geographic_code',
                'source', 'industry_code', 'revision_number',
            ],
            keep='first',
        )
        .sort('period', 'industry_code', 'source', 'revision_number')
    )

    return combined


# Keep backward-compatible alias
ingest_ces = ingest_ces_national


