'''
CES National (CE) data download from BLS flat files and JSON API.

Downloads monthly national employment estimates by supersector.  The primary
path uses flat files (``ce.data.0.AllCESSeries``) for full history; the JSON
API is available as a fallback for the most recent months.

This module handles the download/parsing layer only — transformation into
``PANEL_SCHEMA`` is done by ``ingest/ces_national.py``.
'''

from __future__ import annotations

import logging

import polars as pl

from ._http import BLSHttpClient, _reference_day
from ._programs import build_series_id

logger = logging.getLogger(__name__)

# Supersector codes used in CES National.
# '00' = total nonfarm, '05' = total private, then the 10 supersectors.
_SUPERSECTOR_CODES = [
    '00',  # Total nonfarm
    '05',  # Total private
    '10',  # Mining and logging
    '20',  # Construction
    '30',  # Manufacturing
    '40',  # Trade, transportation, and utilities
    '50',  # Information
    '55',  # Financial activities
    '60',  # Professional and business services
    '65',  # Education and health services
    '70',  # Leisure and hospitality
    '80',  # Other services
]

# Build series IDs programmatically: SA and NSA employment (data_type='01')
CES_SERIES_MAP: dict[str, dict[str, str]] = {}
for _code in _SUPERSECTOR_CODES:
    sa_id = build_series_id(
        'CE', seasonal='S', supersector=_code,
        industry='000000', data_type='01',
    )
    nsa_id = build_series_id(
        'CE', seasonal='U', supersector=_code,
        industry='000000', data_type='01',
    )
    CES_SERIES_MAP[_code] = {'sa': sa_id, 'nsa': nsa_id}


def fetch_ces_national(
    start_year: int | None = None,
    end_year: int | None = None,
    supersectors: list[str] | None = None,
    include_nsa: bool = True,
    client: BLSHttpClient | None = None,
) -> pl.DataFrame:
    '''
    Download CES national employment data via flat files.

    Uses ``ce.data.0.AllCESSeries`` as the primary data source, which
    provides full history without rate limits.

    Parameters
    ----------
    start_year : int or None
        First year to include. If ``None``, no lower bound.
    end_year : int or None
        Last year to include. If ``None``, no upper bound.
    supersectors : list[str] or None
        Supersector codes to fetch. Defaults to all supersectors.
    include_nsa : bool
        If ``True`` (default), include not-seasonally-adjusted series.
    client : BLSHttpClient or None
        Optional HTTP client. Creates a default if ``None``.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns: ``series_id``, ``date``, ``year``,
        ``period``, ``value``, ``supersector_code``,
        ``is_seasonally_adjusted``.
    '''
    if supersectors is None:
        supersectors = _SUPERSECTOR_CODES

    # Build target series ID set
    target_ids: set[str] = set()
    for code in supersectors:
        if code not in CES_SERIES_MAP:
            logger.warning(f'Unknown CES supersector code: {code}')
            continue
        target_ids.add(CES_SERIES_MAP[code]['sa'])
        if include_nsa:
            target_ids.add(CES_SERIES_MAP[code]['nsa'])

    own_client = client is None
    if own_client:
        client = BLSHttpClient()

    try:
        rows = client.get_data('CE', '0.AllCESSeries')
        if not rows:
            logger.warning('No data returned from CE flat file')
            return pl.DataFrame()

        df = pl.DataFrame(rows)

        # Cast types
        if 'year' in df.columns:
            df = df.with_columns(pl.col('year').cast(pl.Int64, strict=False))
        if 'value' in df.columns:
            df = df.with_columns(pl.col('value').cast(pl.Float64, strict=False))

        # Filter to target series
        df = df.filter(pl.col('series_id').is_in(list(target_ids)))

        # Filter by year range
        if start_year is not None:
            df = df.filter(pl.col('year') >= start_year)
        if end_year is not None:
            df = df.filter(pl.col('year') <= end_year)

        # Derive date
        day = _reference_day('CE')
        df = BLSHttpClient._add_date_column(df, day=day)

        # Filter out non-monthly periods (M13 = annual average)
        df = df.filter(
            pl.col('period').str.starts_with('M')
            & (pl.col('period') != 'M13')
        )

        # Add derived columns
        df = df.with_columns([
            pl.col('series_id').str.slice(3, 2).alias('supersector_code'),
            pl.when(pl.col('series_id').str.slice(2, 1) == 'S')
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias('is_seasonally_adjusted'),
        ])

        # Reorder and sort
        col_order = [
            'series_id', 'date', 'year', 'period', 'value',
            'supersector_code', 'is_seasonally_adjusted',
        ]
        extra = [c for c in df.columns if c not in col_order]
        df = df.select([c for c in col_order if c in df.columns] + extra)
        df = df.sort('series_id', 'date')

        return df
    finally:
        if own_client:
            client.close()


def fetch_ces_national_via_api(
    series_ids: list[str],
    start_year: int,
    end_year: int,
    client: BLSHttpClient | None = None,
) -> pl.DataFrame:
    '''
    Download CES National data via the JSON API.

    Fallback for cases where flat files are insufficient (e.g., current
    month data not yet in flat files). Handles the 50-series-per-request
    and year-window constraints by batching.

    Parameters
    ----------
    series_ids : list[str]
        BLS series IDs to fetch.
    start_year : int
        First year.
    end_year : int
        Last year (inclusive).
    client : BLSHttpClient or None
        Optional HTTP client.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns: ``series_id``, ``date``, ``year``,
        ``period``, ``period_name``, ``value``.
    '''
    own_client = client is None
    if own_client:
        client = BLSHttpClient()

    try:
        # Determine year window size based on API version
        window = 20 if client.api_key else 10
        batch_size = 50

        frames: list[pl.DataFrame] = []

        # Split into year windows
        for win_start in range(start_year, end_year + 1, window):
            win_end = min(win_start + window - 1, end_year)

            # Split into series batches
            for batch_start in range(0, len(series_ids), batch_size):
                batch = series_ids[batch_start : batch_start + batch_size]

                try:
                    df = client.get_series(
                        batch,
                        start_year=str(win_start),
                        end_year=str(win_end),
                    )
                    if len(df) > 0:
                        frames.append(df)
                except Exception as e:
                    logger.warning(
                        f'CES API request failed for '
                        f'{win_start}-{win_end}, batch {batch_start}: {e}'
                    )

        if not frames:
            return pl.DataFrame()

        return pl.concat(frames).sort('series_id', 'date')
    finally:
        if own_client:
            client.close()
