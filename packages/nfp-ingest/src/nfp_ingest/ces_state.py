"""CES State data ingestion: BLS download → PANEL_SCHEMA.

CES State estimates use the same underlying sample as national but employ
a top-down estimation structure at the Estimation Super Sector level, with
over 55% of series relying on small area models rather than direct sampling.
State estimates undergo full QCEW replacement during benchmarking and
release two weeks after national.

This module uses the ``bls/`` download layer to fetch state-level CES data
and transforms it into the unified observation panel format.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from nfp_download.bls import BLSHttpClient
from nfp_download.bls import fetch_ces_state as bls_fetch_ces_state
from nfp_lookups.schemas import PANEL_SCHEMA, empty_panel

logger = logging.getLogger(__name__)


def fetch_ces_state_current(
    start_year: int,
    end_year: int,
    states: list[str] | None = None,
    client: BLSHttpClient | None = None,
) -> pl.DataFrame:
    """Fetch current CES state data from BLS via the bls/ download layer.

    Parameters
    ----------
    start_year : int
        First year to fetch.
    end_year : int
        Last year to fetch (inclusive).
    states : list[str] or None
        2-digit state FIPS codes. Defaults to all states.
    client : BLSHttpClient or None
        Optional HTTP client.

    Returns
    -------
    pl.DataFrame
        Observation panel rows conforming to PANEL_SCHEMA.
    """
    try:
        raw = bls_fetch_ces_state(
            states=states,
            start_year=start_year,
            end_year=end_year,
            client=client,
        )
    except Exception as e:
        logger.warning(f'Failed to fetch CES state data: {e}')
        return empty_panel()

    if len(raw) == 0:
        return empty_panel()

    today = date.today()
    rows: list[dict] = []

    # Group by (state_fips, supersector_code, is_seasonally_adjusted)
    for (state_fips, ss_code, is_sa), group in raw.group_by(
        ['state_fips', 'supersector_code', 'is_seasonally_adjusted'],
        maintain_order=True,
    ):
        group = group.sort('date').drop_nulls(subset=['date', 'value'])
        if len(group) < 2:
            continue

        dates = group['date'].to_list()
        values = group['value'].to_numpy().astype(float)

        source = 'ces_state_sa' if is_sa else 'ces_state_nsa'
        source_type = 'official_sa' if is_sa else 'official_nsa'

        # SM supersector_code is 8 digits; extract the 2-digit code
        ss_2digit = str(ss_code)[:2]

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
                    rev_num = -1

                rows.append(
                    {
                        'period': ref_dt,
                        'geographic_type': 'state',
                        'geographic_code': str(state_fips),
                        'industry_code': ss_2digit,
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


def ingest_ces_state(
    vintage_dir: Path | None = None,
    start_year: int = 2003,
    end_year: int | None = None,
    states: list[str] | None = None,
    client: BLSHttpClient | None = None,
) -> pl.DataFrame:
    """Orchestrate CES state data ingestion.

    Parameters
    ----------
    vintage_dir : Path, optional
        Directory containing state vintage parquets (format TBD).
    start_year : int
        First year for data fetch (default 2003).
    end_year : int, optional
        Last year for data fetch. Defaults to current year.
    states : list[str] or None
        2-digit state FIPS codes. Defaults to all states.
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

    # TODO: load state vintage parquets when available
    # State release schedule differs from national (5th Friday after 12th)

    # Fetch current data via bls/ download layer
    try:
        current_df = fetch_ces_state_current(
            start_year, end_year, states=states, client=client,
        )
        if len(current_df) > 0:
            parts.append(current_df)
    except Exception as e:
        logger.warning(f'Failed to fetch current CES state data: {e}')

    if not parts:
        return empty_panel()

    combined = pl.concat(parts)

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
