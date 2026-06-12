"""QCEW data ingestion: BLS download → PANEL_SCHEMA.

QCEW (Quarterly Census of Employment and Wages) provides near-census
employment counts with a 5-month publication lag and asymmetric revision
schedule across quarters.

This module uses the ``bls/`` download layer instead of direct urllib or
eco-stats calls.  It downloads sector-level QCEW data (private + government),
then aggregates to supersector and domain levels to produce all 35 CES
industry codes.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from nfp_download.bls import BLSHttpClient
from nfp_download.bls import fetch_qcew as bls_fetch_qcew
from nfp_download.bls.qcew import QCEW_INDUSTRY_CODES, fetch_qcew_with_geography
from nfp_lookups.industry import (
    GOVT_OWNERSHIP_TO_SECTOR,
    get_domain_supersectors,
    get_supersector_components,
    qcew_to_sector,
)
from nfp_lookups.revision_schedules import QCEW_REVISIONS
from nfp_lookups.schemas import PANEL_SCHEMA, QCEW_VINTAGE_SCHEMA, empty_panel

logger = logging.getLogger(__name__)

# Canonical mapping: QCEW API code / raw NAICS code -> NAICS-based sector code.
# Derived from lookups/industry.py; also includes range-notation NAICS codes that
# may appear in raw API responses.
_QCEW_SECTOR_MAP: dict[str, str] = {
    **qcew_to_sector(),
    '31-33': '31',
    '44-45': '44',
    '48-49': '48',
}


def _map_industry_code(code: str) -> str | None:
    """Map a QCEW industry code to our NAICS-based sector code."""
    return _QCEW_SECTOR_MAP.get(code)


def _extract_sector_employment(
    raw: pl.DataFrame,
    geographic_type: str = 'national',
    geographic_code: str = 'US',
) -> pl.DataFrame:
    """Parse raw QCEW data into sector-level monthly employment rows.

    Returns a DataFrame with columns: ``industry_code``, ``ref_date``,
    ``employment``, ``qtr``, ``geographic_type``, ``geographic_code``.
    """
    required = ['industry_code', 'year', 'qtr',
                 'month1_emplvl', 'month2_emplvl', 'month3_emplvl']
    for col in required:
        if col not in raw.columns:
            logger.warning(f'QCEW response missing column: {col}')
            return pl.DataFrame()

    rows: list[dict] = []
    for row in raw.iter_rows(named=True):
        ind_code = str(row['industry_code'])
        sector = _map_industry_code(ind_code)
        if sector is None:
            continue

        year = int(row['year'])
        qtr = int(row['qtr'])
        geo_t = str(row.get('geographic_type', geographic_type))
        geo_c = str(row.get('geographic_code', geographic_code))

        for m_idx, m_col in enumerate(
            ['month1_emplvl', 'month2_emplvl', 'month3_emplvl'],
            start=1,
        ):
            if row.get(m_col) is None:
                continue
            try:
                emp = int(row[m_col])
            except (ValueError, TypeError):
                continue
            if emp <= 0:
                continue

            month_num = (qtr - 1) * 3 + m_idx
            ref_date = date(year, month_num, 1)
            rows.append({
                'industry_code': sector,
                'ref_date': ref_date,
                'employment': emp,
                'qtr': qtr,
                'geographic_type': geo_t,
                'geographic_code': geo_c,
            })

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)


def _extract_government_employment(raw: pl.DataFrame) -> pl.DataFrame:
    """Extract government employment from QCEW data with ownership codes 1/2/3.

    Government employment is identified by ``own_code`` (1=Federal, 2=State,
    3=Local) on the ``industry_code='10'`` (Total) rows.  Returns a DataFrame
    matching the output of :func:`_extract_sector_employment` with
    ``industry_code`` set to the CES government sector code.
    """
    required = ['own_code', 'industry_code', 'year', 'qtr',
                 'month1_emplvl', 'month2_emplvl', 'month3_emplvl']
    for col in required:
        if col not in raw.columns:
            return pl.DataFrame()

    rows: list[dict] = []
    for row in raw.iter_rows(named=True):
        own = str(row['own_code'])
        govt_sector = GOVT_OWNERSHIP_TO_SECTOR.get(own)
        if govt_sector is None:
            continue
        if str(row['industry_code']) != '10':
            continue

        year = int(row['year'])
        qtr = int(row['qtr'])
        geo_t = str(row.get('geographic_type', 'national'))
        geo_c = str(row.get('geographic_code', 'US'))

        for m_idx, m_col in enumerate(
            ['month1_emplvl', 'month2_emplvl', 'month3_emplvl'],
            start=1,
        ):
            if row.get(m_col) is None:
                continue
            try:
                emp = int(row[m_col])
            except (ValueError, TypeError):
                continue
            if emp <= 0:
                continue

            month_num = (qtr - 1) * 3 + m_idx
            ref_date = date(year, month_num, 1)
            rows.append({
                'industry_code': govt_sector,
                'ref_date': ref_date,
                'employment': emp,
                'qtr': qtr,
                'geographic_type': geo_t,
                'geographic_code': geo_c,
            })

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)


def aggregate_to_hierarchy(sector_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate sector employment into supersector and domain totals.

    Parameters
    ----------
    sector_df : pl.DataFrame
        Sector-level employment with columns ``industry_code``,
        ``ref_date``, ``employment``, ``qtr``, ``geographic_type``,
        ``geographic_code``.

    Returns
    -------
    pl.DataFrame
        Combined DataFrame with sector, supersector, and domain rows.
        Each row has an added ``industry_level`` column.
    """
    if len(sector_df) == 0:
        return pl.DataFrame()

    # Tag sector rows
    result_parts = [
        sector_df.with_columns(pl.lit('sector').alias('industry_level'))
    ]

    group_cols = ['geographic_type', 'geographic_code', 'ref_date', 'qtr']

    # --- Supersectors: sum component sectors ---
    ss_components = get_supersector_components()
    for ss_code, component_sectors in ss_components.items():
        ss_df = (
            sector_df
            .filter(pl.col('industry_code').is_in(component_sectors))
            .group_by(group_cols)
            .agg(pl.col('employment').sum())
            .with_columns(
                pl.lit(ss_code).alias('industry_code'),
                pl.lit('supersector').alias('industry_level'),
            )
        )
        if len(ss_df) > 0:
            result_parts.append(ss_df)

    # --- Domains: sum component supersectors ---
    # First, collect the supersector rows we just built
    ss_rows = [p for p in result_parts if 'industry_level' in p.columns]
    if len(ss_rows) > 1:
        all_ss = pl.concat(
            [p.filter(pl.col('industry_level') == 'supersector') for p in ss_rows],
            how='diagonal_relaxed',
        )
    else:
        all_ss = pl.DataFrame()

    if len(all_ss) > 0:
        for domain_code in ['00', '05', '06', '07', '08']:
            component_ss = get_domain_supersectors(domain_code)
            dom_df = (
                all_ss
                .filter(pl.col('industry_code').is_in(component_ss))
                .group_by(group_cols)
                .agg(pl.col('employment').sum())
                .with_columns(
                    pl.lit(domain_code).alias('industry_code'),
                    pl.lit('domain').alias('industry_level'),
                )
            )
            if len(dom_df) > 0:
                result_parts.append(dom_df)

    if not result_parts:
        return pl.DataFrame()

    return pl.concat(result_parts, how='diagonal_relaxed')


def _build_growth_panel(
    emp_df: pl.DataFrame,
) -> pl.DataFrame:
    """Compute growth rates from employment levels and build PANEL_SCHEMA rows.

    Parameters
    ----------
    emp_df : pl.DataFrame
        Employment data with columns ``industry_code``, ``industry_level``,
        ``ref_date``, ``employment``, ``qtr``, ``geographic_type``,
        ``geographic_code``.

    Returns
    -------
    pl.DataFrame
        Panel rows conforming to PANEL_SCHEMA.
    """
    if len(emp_df) == 0:
        return empty_panel()

    rows: list[dict] = []
    for keys, grp in emp_df.group_by(
        ['geographic_type', 'geographic_code', 'industry_code', 'industry_level'],
        maintain_order=True,
    ):
        geo_type, geo_code, ind_code, ind_level = keys
        grp = grp.sort('ref_date')
        emp = grp['employment'].to_numpy().astype(float)
        ref_dates = grp['ref_date'].to_list()
        qtrs = grp['qtr'].to_list()

        for i in range(1, len(emp)):
            if emp[i] > 0 and emp[i - 1] > 0:
                growth = float(np.log(emp[i]) - np.log(emp[i - 1]))
                q_label = f'Q{qtrs[i]}'
                pub_lag = QCEW_REVISIONS[q_label][0].lag_months
                ref_month = ref_dates[i].month
                vdate_month = ref_month + pub_lag
                vdate_year = ref_dates[i].year + (vdate_month - 1) // 12
                vdate_month = ((vdate_month - 1) % 12) + 1
                vintage_dt = date(vdate_year, vdate_month, 1)

                lag_months = (
                    (vintage_dt.year - ref_dates[i].year) * 12
                    + vintage_dt.month
                    - ref_dates[i].month
                )

                rows.append({
                    'period': ref_dates[i],
                    'geographic_type': str(geo_type),
                    'geographic_code': str(geo_code),
                    'industry_code': str(ind_code),
                    'industry_level': str(ind_level),
                    'source': 'qcew',
                    'source_type': 'census',
                    'growth': growth,
                    'employment_level': float(emp[i]),
                    'is_seasonally_adjusted': False,
                    'vintage_date': vintage_dt,
                    'revision_number': 0,
                    'is_final': False,
                    'publication_lag_months': lag_months,
                    'coverage_ratio': None,
                })

    if not rows:
        return empty_panel()

    return pl.DataFrame(rows, schema=PANEL_SCHEMA)


def fetch_qcew_current(
    start_year: int,
    end_year: int,
    client: BLSHttpClient | None = None,
) -> pl.DataFrame:
    """Fetch current QCEW data from the BLS QCEW CSV API via bls/ layer.

    Downloads sector-level data (private + government), then aggregates
    to produce supersector and domain totals.

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
        Observation panel rows conforming to PANEL_SCHEMA with all
        industry levels (sector, supersector, domain).
    """
    years = list(range(start_year, end_year + 1))

    try:
        raw = bls_fetch_qcew(
            years=years,
            industries=['10'] + list(QCEW_INDUSTRY_CODES.keys()),
            ownership_code='5',
            client=client,
        )
    except Exception as e:
        logger.warning(f'Failed to fetch QCEW data via bls layer: {e}')
        return empty_panel()

    if len(raw) == 0:
        return empty_panel()

    # Extract private-sector employment
    sector_emp = _extract_sector_employment(raw)

    # Fetch government data (ownership 1/2/3 on total industry)
    try:
        for own_code in ['1', '2', '3']:
            govt_raw = bls_fetch_qcew(
                years=years, industries=['10'],
                ownership_code=own_code, client=client,
            )
            if len(govt_raw) > 0:
                govt_emp = _extract_government_employment(govt_raw)
                if len(govt_emp) > 0:
                    sector_emp = (
                        pl.concat([sector_emp, govt_emp], how='diagonal_relaxed')
                        if len(sector_emp) > 0
                        else govt_emp
                    )
    except Exception as e:
        logger.warning(f'Failed to fetch QCEW government data: {e}')

    if len(sector_emp) == 0:
        return empty_panel()

    all_levels = aggregate_to_hierarchy(sector_emp)
    return _build_growth_panel(all_levels)


def fetch_qcew_current_with_geography(
    start_year: int,
    end_year: int,
    include_national: bool = True,
    include_states: bool = True,
    state_fips_list: list[str] | None = None,
    client: BLSHttpClient | None = None,
) -> pl.DataFrame:
    """Fetch current QCEW data for national and/or state geographies.

    Downloads private-sector and government data, then aggregates to
    supersector and domain levels.

    Parameters
    ----------
    start_year : int
        First year to fetch.
    end_year : int
        Last year to fetch (inclusive).
    include_national : bool
        Include national-level rows. Defaults to ``True``.
    include_states : bool
        Include state-level rows. Defaults to ``True``.
    state_fips_list : list[str] or None
        Specific 2-digit state FIPS codes. If ``None``, uses all from
        :data:`~alt_nfp.lookups.geography.STATES`.
    client : BLSHttpClient or None
        Optional HTTP client.

    Returns
    -------
    pl.DataFrame
        Observation panel rows conforming to PANEL_SCHEMA with all
        industry levels (sector, supersector, domain) and
        ``geographic_type`` / ``geographic_code`` populated.
    """
    years = list(range(start_year, end_year + 1))

    # Download private + government in one call using multiple ownership codes
    try:
        raw = fetch_qcew_with_geography(
            years=years,
            ownership_codes=['5', '1', '2', '3'],
            include_national=include_national,
            include_states=include_states,
            state_fips_list=state_fips_list,
            client=client,
        )
    except Exception as e:
        logger.warning(f'Failed to fetch QCEW geographic data: {e}')
        return empty_panel()

    if len(raw) == 0:
        return empty_panel()

    # Extract private-sector employment (own_code='5')
    private_raw = raw.filter(pl.col('own_code') == '5') if 'own_code' in raw.columns else raw
    sector_emp = _extract_sector_employment(private_raw)

    # Extract government employment (own_code 1/2/3)
    if 'own_code' in raw.columns:
        govt_raw = raw.filter(pl.col('own_code').is_in(['1', '2', '3']))
        if len(govt_raw) > 0:
            govt_emp = _extract_government_employment(govt_raw)
            if len(govt_emp) > 0:
                sector_emp = (
                    pl.concat([sector_emp, govt_emp], how='diagonal_relaxed')
                    if len(sector_emp) > 0
                    else govt_emp
                )

    if len(sector_emp) == 0:
        return empty_panel()

    all_levels = aggregate_to_hierarchy(sector_emp)
    return _build_growth_panel(all_levels)


def load_qcew_vintages(path: Path) -> pl.DataFrame:
    """Load historical QCEW revision data from a parquet file.

    Parameters
    ----------
    path : Path
        Path to qcew_vintages.parquet conforming to QCEW_VINTAGE_SCHEMA.

    Returns
    -------
    pl.DataFrame
        Observation panel rows conforming to PANEL_SCHEMA.
    """
    if not path.exists():
        logger.info(f'QCEW vintage file not found: {path}')
        return empty_panel()

    raw = pl.read_parquet(path)

    # Validate schema
    for col, _dtype in QCEW_VINTAGE_SCHEMA.items():
        if col not in raw.columns:
            raise ValueError(f'QCEW vintage parquet missing column: {col}')

    # Filter to national, private
    raw = raw.filter(
        (pl.col('area_fips') == 'US000') & (pl.col('own_code') == 5)
    )

    # Build period column: date from ref_year and month within quarter
    raw = raw.with_columns(
        pl.struct(['ref_year', 'ref_quarter', 'ref_month'])
        .map_elements(
            lambda s: date(s['ref_year'], (s['ref_quarter'] - 1) * 3 + s['ref_month'], 1),
            return_dtype=pl.Date,
        )
        .alias('period')
    )

    # Sort by (industry_code, period, revision_number) for growth computation
    raw = raw.sort('industry_code', 'period', 'revision_number')

    rows: list[dict] = []
    for (ind_code, rev_num), group in raw.group_by(
        ['industry_code', 'revision_number'], maintain_order=True
    ):
        group = group.sort('period')
        emp = group['employment'].to_numpy().astype(float)
        periods = group['period'].to_list()
        vintage_dates = group['vintage_date'].to_list()
        ref_quarters = group['ref_quarter'].to_list()

        # Determine if this is the final revision for this quarter
        q_label = f'Q{ref_quarters[0]}' if ref_quarters else 'Q1'
        max_rev = max(s.revision_number for s in QCEW_REVISIONS.get(q_label, []))
        is_final = rev_num == max_rev

        for i in range(1, len(emp)):
            if emp[i] > 0 and emp[i - 1] > 0:
                growth = float(np.log(emp[i]) - np.log(emp[i - 1]))
                vdate = vintage_dates[i]
                lag = (
                    (vdate.year - periods[i].year) * 12
                    + vdate.month
                    - periods[i].month
                )

                rows.append(
                    {
                        'period': periods[i],
                        'geographic_type': 'national',
                        'geographic_code': 'US',
                        'industry_code': str(ind_code),
                        'industry_level': 'sector',
                        'source': 'qcew',
                        'source_type': 'census',
                        'growth': growth,
                        'employment_level': float(emp[i]),
                        'is_seasonally_adjusted': False,
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


def ingest_qcew(
    vintage_dir: Path | None = None,
    start_year: int = 2003,
    end_year: int | None = None,
    include_states: bool = False,
    state_fips_list: list[str] | None = None,
    client: BLSHttpClient | None = None,
) -> pl.DataFrame:
    """Orchestrate QCEW data ingestion from BLS and historical vintages.

    Parameters
    ----------
    vintage_dir : Path, optional
        Directory containing qcew_vintages.parquet.
    start_year : int
        First year for API fetch (default 2003).
    end_year : int, optional
        Last year for API fetch. Defaults to current year.
    include_states : bool
        If ``True``, also fetch state-level data via
        :func:`fetch_qcew_current_with_geography`. Defaults to ``False``.
    state_fips_list : list[str] or None
        Specific 2-digit state FIPS codes. If ``None`` and
        ``include_states`` is ``True``, uses all states.
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
        vintage_path = vintage_dir / 'qcew_vintages.parquet'
        vintage_df = load_qcew_vintages(vintage_path)
        if len(vintage_df) > 0:
            parts.append(vintage_df)

    # Fetch current data via bls/ download layer
    if include_states:
        try:
            current_df = fetch_qcew_current_with_geography(
                start_year,
                end_year,
                include_national=True,
                include_states=True,
                state_fips_list=state_fips_list,
                client=client,
            )
            if len(current_df) > 0:
                parts.append(current_df)
        except Exception as e:
            logger.warning(f'Failed to fetch current QCEW geographic data: {e}')
    else:
        try:
            current_df = fetch_qcew_current(start_year, end_year, client=client)
            if len(current_df) > 0:
                parts.append(current_df)
        except Exception as e:
            logger.warning(f'Failed to fetch current QCEW data: {e}')

    if not parts:
        return empty_panel()

    combined = pl.concat(parts)

    # Deduplicate: prefer vintage data (higher revision_number) over API data
    combined = (
        combined.sort('revision_number', descending=True)
        .unique(
            subset=[
                'period', 'geographic_type', 'geographic_code',
                'source', 'industry_code', 'industry_level', 'revision_number',
            ],
            keep='first',
        )
        .sort('period', 'industry_level', 'industry_code', 'revision_number')
    )

    return combined


