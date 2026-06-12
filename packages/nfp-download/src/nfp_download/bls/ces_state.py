'''
CES State and Area (SM) data download from BLS flat files and JSON API.

Downloads monthly state-level employment estimates by supersector. Uses
flat files (``sm.data.0.Current``) as the primary source.

This module handles the download/parsing layer only — transformation into
``PANEL_SCHEMA`` is done by ``ingest/ces_state.py``.

CES State estimates differ materially from national estimates: they use
top-down estimation with small area models, undergo full QCEW replacement
during benchmarking, and release two weeks after national.
'''

from __future__ import annotations

import logging

import polars as pl

from ._http import BLSHttpClient, _reference_day
from ._programs import build_series_id

logger = logging.getLogger(__name__)

# All 50 states + DC FIPS codes
_ALL_STATE_FIPS = [
    '01', '02', '04', '05', '06', '08', '09', '10', '11', '12',
    '13', '15', '16', '17', '18', '19', '20', '21', '22', '23',
    '24', '25', '26', '27', '28', '29', '30', '31', '32', '33',
    '34', '35', '36', '37', '38', '39', '40', '41', '42', '44',
    '45', '46', '47', '48', '49', '50', '51', '53', '54', '55',
    '56',
]

# SM supersector/industry codes (8-digit).  Total nonfarm and the 10
# supersectors use the same codes as CE but zero-padded to 8 digits.
_SM_SUPERSECTOR_CODES = [
    '00000000',  # Total nonfarm
    '05000000',  # Total private
    '10000000',  # Mining and logging
    '20000000',  # Construction
    '30000000',  # Manufacturing
    '40000000',  # Trade, transportation, and utilities
    '50000000',  # Information
    '55000000',  # Financial activities
    '60000000',  # Professional and business services
    '65000000',  # Education and health services
    '70000000',  # Leisure and hospitality
    '80000000',  # Other services
]


def build_state_series_ids(
    states: list[str],
    supersectors: list[str] | None = None,
    seasonal: str = 'S',
    data_type: str = '01',
) -> list[str]:
    '''
    Build CES State series IDs for the given states and supersectors.

    Parameters
    ----------
    states : list[str]
        2-digit state FIPS codes.
    supersectors : list[str] or None
        8-digit supersector/industry codes. Defaults to all standard
        supersectors.
    seasonal : str
        ``'S'`` for SA, ``'U'`` for NSA.
    data_type : str
        Data type code. ``'01'`` = all employees.

    Returns
    -------
    list[str]
        List of 20-character SM series IDs.
    '''
    if supersectors is None:
        supersectors = _SM_SUPERSECTOR_CODES

    ids: list[str] = []
    for state in states:
        for ss in supersectors:
            sid = build_series_id(
                'SM',
                seasonal=seasonal,
                state=state,
                area='00000',
                supersector_industry=ss,
                data_type=data_type,
            )
            ids.append(sid)
    return ids


def fetch_ces_state(
    states: list[str] | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    supersectors: list[str] | None = None,
    include_nsa: bool = True,
    client: BLSHttpClient | None = None,
) -> pl.DataFrame:
    '''
    Download CES State employment data via flat files.

    Uses ``sm.data.0.Current`` as the primary data source.

    Parameters
    ----------
    states : list[str] or None
        2-digit state FIPS codes. Defaults to all 50 states + DC.
    start_year : int or None
        First year to include.
    end_year : int or None
        Last year to include.
    supersectors : list[str] or None
        8-digit supersector/industry codes. Defaults to all standard
        supersectors.
    include_nsa : bool
        If ``True`` (default), include NSA series.
    client : BLSHttpClient or None
        Optional HTTP client.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns: ``series_id``, ``date``, ``year``,
        ``period``, ``value``, ``state_fips``, ``area_code``,
        ``supersector_code``, ``is_seasonally_adjusted``.
    '''
    if states is None:
        states = _ALL_STATE_FIPS
    if supersectors is None:
        supersectors = _SM_SUPERSECTOR_CODES

    # Build target series ID set
    target_ids: set[str] = set()
    sa_ids = build_state_series_ids(states, supersectors, seasonal='S')
    target_ids.update(sa_ids)
    if include_nsa:
        nsa_ids = build_state_series_ids(states, supersectors, seasonal='U')
        target_ids.update(nsa_ids)

    own_client = client is None
    if own_client:
        client = BLSHttpClient()

    try:
        rows = client.get_data('SM', '0.Current')
        if not rows:
            logger.warning('No data returned from SM flat file')
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
        day = _reference_day('SM')
        df = BLSHttpClient._add_date_column(df, day=day)

        # Filter out non-monthly periods
        df = df.filter(
            pl.col('period').str.starts_with('M')
            & (pl.col('period') != 'M13')
        )

        # Add derived columns from series ID (SM = 20 chars)
        # positions: prefix(1-2), seasonal(3), state(4-5), area(6-10),
        #            supersector_industry(11-18), data_type(19-20)
        df = df.with_columns([
            pl.col('series_id').str.slice(3, 2).alias('state_fips'),
            pl.col('series_id').str.slice(5, 5).alias('area_code'),
            pl.col('series_id').str.slice(10, 8).alias('supersector_code'),
            pl.when(pl.col('series_id').str.slice(2, 1) == 'S')
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias('is_seasonally_adjusted'),
        ])

        # Reorder and sort
        col_order = [
            'series_id', 'date', 'year', 'period', 'value',
            'state_fips', 'area_code', 'supersector_code',
            'is_seasonally_adjusted',
        ]
        extra = [c for c in df.columns if c not in col_order]
        df = df.select([c for c in col_order if c in df.columns] + extra)
        df = df.sort('series_id', 'date')

        return df
    finally:
        if own_client:
            client.close()


def fetch_state_mapping(
    client: BLSHttpClient | None = None,
) -> pl.DataFrame:
    '''
    Download the SM state mapping file.

    Parameters
    ----------
    client : BLSHttpClient or None
        Optional HTTP client.

    Returns
    -------
    pl.DataFrame
        DataFrame with state FIPS codes and names.
    '''
    own_client = client is None
    if own_client:
        client = BLSHttpClient()

    try:
        rows = client.get_mapping('SM', 'state')
        if not rows:
            return pl.DataFrame()
        return pl.DataFrame(rows)
    finally:
        if own_client:
            client.close()


def fetch_area_mapping(
    client: BLSHttpClient | None = None,
) -> pl.DataFrame:
    '''
    Download the SM area mapping file for MSA codes.

    Parameters
    ----------
    client : BLSHttpClient or None
        Optional HTTP client.

    Returns
    -------
    pl.DataFrame
        DataFrame with area codes and names.
    '''
    own_client = client is None
    if own_client:
        client = BLSHttpClient()

    try:
        rows = client.get_mapping('SM', 'area')
        if not rows:
            return pl.DataFrame()
        return pl.DataFrame(rows)
    finally:
        if own_client:
            client.close()
