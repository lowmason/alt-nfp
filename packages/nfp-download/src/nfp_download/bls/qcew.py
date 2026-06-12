'''
QCEW data download from the BLS QCEW CSV API.

Downloads quarterly employment and wages data by industry. This module
handles the download/parsing layer only — transformation into
``PANEL_SCHEMA`` is done by ``ingest/qcew.py``.

The QCEW CSV API at ``data.bls.gov/cew/data/api/`` returns quarterly
CSV files with monthly employment levels, wages, and establishment
counts by area, ownership, and industry.

.. note::
   The CSV API only has data from **2014** onward. Requests for
   earlier years (e.g. 2003–2013) return 404. For historical data
   use the downloadable data files at bls.gov/cew.
'''

from __future__ import annotations

import logging
import re

import polars as pl

from ._http import BLSHttpClient

logger = logging.getLogger(__name__)

_STATE_AREA_RE = re.compile(r'^\d{2}000$')

# QCEW CSV API industry codes.  The API uses its own code scheme:
#   '10'   = Total, all industries
#   '1011' = NAICS 11 (Agriculture)
#   '1012' = NAICS 21 (Mining)
#   '1013' = NAICS 22 (Utilities)
#   '1021' = NAICS 23 (Construction)
#   '1022' = NAICS 31-33 (Manufacturing)
#   '1023' = NAICS 42 (Wholesale Trade)
#   '1024' = NAICS 44-45 (Retail Trade)
#   '1025' = NAICS 48-49 (Transportation and Warehousing)
#   '1026' = NAICS 51 (Information)
#   '1027' = NAICS 52 (Finance and Insurance)
#   '1028' = NAICS 53 (Real Estate)
#   '1029' = NAICS 54 (Professional and Technical Services)
#   '102A' = NAICS 55 (Management of Companies)
#   '102B' = NAICS 56 (Administrative and Waste Services)
#   '102C' = NAICS 61 (Educational Services)
#   '102D' = NAICS 62 (Health Care and Social Assistance)
#   '102E' = NAICS 71 (Arts, Entertainment, and Recreation)
#   '102F' = NAICS 72 (Accommodation and Food Services)
#   '102G' = NAICS 81 (Other Services)

QCEW_INDUSTRY_CODES: dict[str, str] = {
    '10': 'Total, all industries',
    '1011': 'NAICS 11 - Agriculture',
    '1012': 'NAICS 21 - Mining',
    '1013': 'NAICS 22 - Utilities',
    '1021': 'NAICS 23 - Construction',
    '1022': 'NAICS 31-33 - Manufacturing',
    '1023': 'NAICS 42 - Wholesale Trade',
    '1024': 'NAICS 44-45 - Retail Trade',
    '1025': 'NAICS 48-49 - Transportation and Warehousing',
    '1026': 'NAICS 51 - Information',
    '1027': 'NAICS 52 - Finance and Insurance',
    '1028': 'NAICS 53 - Real Estate',
    '1029': 'NAICS 54 - Professional and Technical Services',
    '102A': 'NAICS 55 - Management of Companies',
    '102B': 'NAICS 56 - Administrative and Waste Services',
    '102C': 'NAICS 61 - Educational Services',
    '102D': 'NAICS 62 - Health Care and Social Assistance',
    '102E': 'NAICS 71 - Arts, Entertainment, and Recreation',
    '102F': 'NAICS 72 - Accommodation and Food Services',
    '102G': 'NAICS 81 - Other Services',
}

# Default industries: total + 2-digit private-sector NAICS needed for the model.
_DEFAULT_INDUSTRIES = [
    '10',
    '1012', '1013', '1021', '1022', '1023', '1024', '1025',
    '1026', '1027', '1028', '1029', '102A', '102B', '102C',
    '102D', '102E', '102F', '102G',
]

# Government ownership codes → CES sector labels.
_GOVT_OWNERSHIP_CODES = ['1', '2', '3']


def fetch_qcew(
    years: list[int],
    quarters: list[int] | None = None,
    industries: list[str] | None = None,
    area_fips: str = 'US000',
    ownership_code: str = '5',
    client: BLSHttpClient | None = None,
) -> pl.DataFrame:
    '''
    Download QCEW data for the specified year/quarter/industry combinations.

    Parameters
    ----------
    years : list[int]
        Reference years (e.g., ``[2020, 2021, 2022]``).
    quarters : list[int] or None
        Quarters (1-4). Defaults to ``[1, 2, 3, 4]``.
    industries : list[str] or None
        QCEW API industry codes. If ``None``, defaults to all 2-digit
        private-sector NAICS codes.
    area_fips : str
        Area FIPS code. Defaults to ``'US000'`` (national).
    ownership_code : str
        Ownership filter. Defaults to ``'5'`` (private).
    client : BLSHttpClient or None
        Optional HTTP client. Creates a default if ``None``.

    Returns
    -------
    pl.DataFrame
        Raw QCEW data with columns including ``area_fips``, ``own_code``,
        ``industry_code``, ``year``, ``qtr``, ``month1_emplvl``,
        ``month2_emplvl``, ``month3_emplvl``, and others.
    '''
    if quarters is None:
        quarters = [1, 2, 3, 4]
    if industries is None:
        industries = _DEFAULT_INDUSTRIES

    own_client = client is None
    if own_client:
        client = BLSHttpClient()

    try:
        frames: list[pl.DataFrame] = []
        first_failure: str | None = None
        for year in years:
            for qtr in quarters:
                for industry in industries:
                    try:
                        df = client.get_qcew_csv(year, qtr, industry)
                    except Exception as e:
                        if first_failure is None:
                            first_failure = f'{year}Q{qtr} industry={industry}: {e}'
                        logger.debug(
                            f'QCEW fetch failed: {year}Q{qtr} '
                            f'industry={industry}: {e}'
                        )
                        continue

                    if len(df) == 0:
                        continue

                    # Cast own_code to string for filtering
                    if 'own_code' in df.columns:
                        df = df.with_columns(
                            pl.col('own_code').cast(pl.Utf8)
                        )
                    if 'area_fips' in df.columns:
                        df = df.with_columns(
                            pl.col('area_fips').cast(pl.Utf8)
                        )

                    # Filter by area and ownership
                    df = df.filter(
                        (pl.col('area_fips') == area_fips)
                        & (pl.col('own_code') == ownership_code)
                    )

                    if len(df) > 0:
                        frames.append(df)
                        logger.debug(
                            f'QCEW fetched: {year}Q{qtr} '
                            f'industry={industry} ({len(df)} rows)'
                        )

        if not frames:
            logger.warning(
                'No QCEW data fetched for the requested parameters. '
                'The BLS QCEW CSV API at data.bls.gov only has data from 2014 '
                'onward; for 2003-2013 use downloadable data files from bls.gov/cew.'
            )
            if first_failure:
                logger.warning(f'First failure: {first_failure}')
            return pl.DataFrame()

        return pl.concat(frames, how='diagonal_relaxed')
    finally:
        if own_client:
            client.close()


def fetch_qcew_with_geography(
    years: list[int],
    quarters: list[int] | None = None,
    industries: list[str] | None = None,
    ownership_codes: list[str] | None = None,
    include_national: bool = True,
    include_states: bool = True,
    state_fips_list: list[str] | None = None,
    client: BLSHttpClient | None = None,
) -> pl.DataFrame:
    '''
    Download QCEW data for national and/or state-level geographies.

    Parameters
    ----------
    years : list[int]
        Reference years.
    quarters : list[int] or None
        Quarters (1-4). Defaults to all four.
    industries : list[str] or None
        QCEW API industry codes. Defaults to all private-sector + total.
    ownership_codes : list[str] or None
        Ownership codes to include. Defaults to ``['5']`` (private only).
        Use ``['5', '1', '2', '3']`` to include government sectors.
    include_national : bool
        Include national (US000) rows. Defaults to ``True``.
    include_states : bool
        Include state-level rows. Defaults to ``True``.
    state_fips_list : list[str] or None
        Specific state FIPS to include. If ``None``, uses all from
        :data:`~alt_nfp.lookups.geography.STATES`.
    client : BLSHttpClient or None
        Optional HTTP client.

    Returns
    -------
    pl.DataFrame
        Raw QCEW data with added ``geographic_type``, ``geographic_code``,
        and ``own_code`` columns. National rows have
        ``geographic_type='national'`` and ``geographic_code='00'``;
        state rows have ``geographic_type='state'`` and
        ``geographic_code`` set to the 2-digit state FIPS.
    '''
    if quarters is None:
        quarters = [1, 2, 3, 4]
    if industries is None:
        industries = _DEFAULT_INDUSTRIES
    if ownership_codes is None:
        ownership_codes = ['5']
    if state_fips_list is None:
        from alt_nfp.lookups.geography import STATES
        state_fips_list = STATES

    ownership_set = set(ownership_codes)
    state_areas = {f'{fips}000' for fips in state_fips_list}

    own_client = client is None
    if own_client:
        client = BLSHttpClient()

    try:
        frames: list[pl.DataFrame] = []
        first_failure: str | None = None
        for year in years:
            for qtr in quarters:
                for industry in industries:
                    try:
                        df = client.get_qcew_csv(year, qtr, industry)
                    except Exception as e:
                        if first_failure is None:
                            first_failure = f'{year}Q{qtr} industry={industry}: {e}'
                        logger.debug(
                            f'QCEW fetch failed: {year}Q{qtr} '
                            f'industry={industry}: {e}'
                        )
                        continue

                    if len(df) == 0:
                        continue

                    if 'own_code' in df.columns:
                        df = df.with_columns(pl.col('own_code').cast(pl.Utf8))
                    if 'area_fips' in df.columns:
                        df = df.with_columns(pl.col('area_fips').cast(pl.Utf8))

                    df = df.filter(
                        pl.col('own_code').is_in(list(ownership_set))
                    )

                    if len(df) == 0:
                        continue

                    parts: list[pl.DataFrame] = []

                    if include_national:
                        nat = df.filter(pl.col('area_fips') == 'US000')
                        if len(nat) > 0:
                            nat = nat.with_columns(
                                pl.lit('national').alias('geographic_type'),
                                pl.lit('00').alias('geographic_code'),
                            )
                            parts.append(nat)

                    if include_states:
                        state_df = df.filter(
                            pl.col('area_fips').is_in(list(state_areas))
                        )
                        if len(state_df) > 0:
                            state_df = state_df.with_columns(
                                pl.lit('state').alias('geographic_type'),
                                pl.col('area_fips').str.slice(0, 2).alias(
                                    'geographic_code'
                                ),
                            )
                            parts.append(state_df)

                    if parts:
                        combined = pl.concat(parts, how='diagonal_relaxed')
                        frames.append(combined)

        if not frames:
            logger.warning(
                'No QCEW data fetched for the requested parameters. '
                'The BLS QCEW CSV API at data.bls.gov only has data from 2014 '
                'onward; for 2003-2013 use downloadable data files from bls.gov/cew.'
            )
            if first_failure:
                logger.warning(f'First failure: {first_failure}')
            return pl.DataFrame()

        return pl.concat(frames, how='diagonal_relaxed')
    finally:
        if own_client:
            client.close()


def fetch_qcew_annual_files(
    year: int,
    client: BLSHttpClient | None = None,
) -> pl.DataFrame:
    '''
    Download full annual QCEW data files.

    .. note::
        Not yet implemented. Placeholder for bulk historical downloads
        from ``bls.gov/cew/downloadable-data-files.htm``.

    Parameters
    ----------
    year : int
        Reference year.
    client : BLSHttpClient or None
        Optional HTTP client.

    Returns
    -------
    pl.DataFrame
        Empty DataFrame (not yet implemented).
    '''
    # TODO: implement bulk annual file download
    raise NotImplementedError(
        'fetch_qcew_annual_files is not yet implemented. '
        'Use fetch_qcew() for quarterly API downloads.'
    )
