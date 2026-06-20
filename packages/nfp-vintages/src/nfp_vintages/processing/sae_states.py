"""Fetch SAE (State and Area Employment) revisions from the FRED/ALFRED API.

For each state x industry combination:
1. Queries ALFRED for available vintage dates.
2. Downloads wide-format observations (one column per vintage).
3. Extracts the initial (first non-null) and latest (last non-null) employment
   level for every reference date.
4. Splits into revision-0 (initial) and revision-1 (latest), joins vintage
   dates, and writes ``sae_revisions.parquet``.

Requires ``FRED_API_KEY`` in the environment.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from pathlib import Path

import httpx
import polars as pl
from nfp_lookups.industry import CES_SECTOR_TO_NAICS, SINGLE_SECTOR_SUPERSECTORS
from nfp_lookups.paths import INTERMEDIATE_DIR, VINTAGE_DATES_PATH, storage_options_for

OUTPUT_PATH = INTERMEDIATE_DIR / 'sae_revisions.parquet'
CHECKPOINT_PATH = INTERMEDIATE_DIR / 'sae_checkpoint.parquet'

FRED_BASE = 'https://api.stlouisfed.org/fred'

FIPS_TO_ABBREV = {
    '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA',
    '08': 'CO', '09': 'CT', '10': 'DE', '11': 'DC', '12': 'FL',
    '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN',
    '19': 'IA', '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME',
    '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN', '28': 'MS',
    '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH',
    '34': 'NJ', '35': 'NM', '36': 'NY', '37': 'NC', '38': 'ND',
    '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI',
    '45': 'SC', '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT',
    '50': 'VT', '51': 'VA', '53': 'WA', '54': 'WV', '55': 'WI',
    '56': 'WY', '72': 'PR', '78': 'VI',
}

# (ces_8digit_code, industry_code, name, level, short_suffix_or_None)
#
# short_suffix: used for {ABBREV}{SUFFIX} IDs (~222 vintages from 2007)
# None:         falls back to SMU{FIPS}00000{CES_8}01 (~139 vintages from 2014)
INDUSTRIES = [
    ('00000000', '00', 'Total Nonfarm', 'domain', 'NAN'),
    ('05000000', '05', 'Total Private', 'domain', None),
    ('06000000', '06', 'Goods-Producing', 'domain', None),
    ('07000000', '07', 'Service-Providing', 'domain', None),
    ('08000000', '08', 'Private Service-Providing', 'domain', None),
    ('10000000', '10', 'Natural Resources and Mining', 'supersector', 'NRMNN'),
    ('20000000', '20', 'Construction', 'supersector', 'CONSN'),
    ('30000000', '30', 'Manufacturing', 'supersector', 'MFGN'),
    ('40000000', '40', 'Trade, Transportation, and Utilities', 'supersector', 'TRADN'),
    ('50000000', '50', 'Information', 'supersector', 'INFON'),
    ('55000000', '55', 'Financial Activities', 'supersector', 'FIREN'),
    ('60000000', '60', 'Professional and Business Services', 'supersector', 'PBSVN'),
    ('65000000', '65', 'Education and Health Services', 'supersector', 'EDUHN'),
    ('70000000', '70', 'Leisure and Hospitality', 'supersector', 'LEIHN'),
    ('80000000', '80', 'Other Services', 'supersector', 'SRVON'),
    ('90000000', '90', 'Government', 'supersector', 'GOVTN'),
    ('10210000', '21', 'Mining', 'sector', None),
    ('31000000', '31', 'Durable Goods', 'sector', None),
    ('32000000', '32', 'Nondurable Goods', 'sector', None),
    ('41000000', '41', 'Wholesale Trade', 'sector', None),
    ('42000000', '42', 'Retail Trade', 'sector', None),
    ('43000000', '43', 'Transp., Warehousing & Utilities', 'sector', None),
    ('43220000', '22', 'Utilities', 'sector', None),
    ('55520000', '52', 'Finance and Insurance', 'sector', None),
    ('55530000', '53', 'Real Estate', 'sector', None),
    ('60540000', '54', 'Prof., Scientific & Tech. Services', 'sector', None),
    ('60550000', '55', 'Management of Companies', 'sector', None),
    ('60560000', '56', 'Admin. and Support Services', 'sector', None),
    ('65610000', '61', 'Private Educational Services', 'sector', None),
    ('65620000', '62', 'Health Care and Social Assistance', 'sector', None),
    ('70710000', '71', 'Arts, Entertainment & Recreation', 'sector', None),
    ('70720000', '72', 'Accommodation and Food Services', 'sector', None),
    ('90910000', '91', 'Federal Government', 'sector', None),
    ('90920000', '92', 'State Government', 'sector', None),
    ('90930000', '93', 'Local Government', 'sector', None),
]


# ---------------------------------------------------------------------------
# ALFRED API helpers
# ---------------------------------------------------------------------------


def _chunked(xs: list[str], n: int) -> Iterable[list[str]]:
    """Yield successive *n*-sized chunks from *xs*."""
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def _request_with_retry(
    client: httpx.Client,
    url: str,
    params: dict,
    timeout: float = 30.0,
    max_retries: int = 6,
) -> httpx.Response:
    """GET with exponential backoff on 429, transient 5xx, and timeouts."""
    for attempt in range(max_retries):
        try:
            r = client.get(url, params=params, timeout=timeout)
        except (
            httpx.TimeoutException,
            httpx.RemoteProtocolError,
            httpx.NetworkError,
        ) as exc:
            wait = min(2**attempt, 60)
            label = type(exc).__name__
            print(f'    [{label}] retrying in {wait}s ...')
            time.sleep(wait)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            wait = min(2**attempt, 60)
            print(f'    [{r.status_code}] retrying in {wait}s ...')
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r


def get_vintage_dates(
    client: httpx.Client, series_id: str, api_key: str,
) -> list[str]:
    """Return the list of vintage date strings for *series_id* from ALFRED."""
    r = _request_with_retry(
        client,
        f'{FRED_BASE}/series/vintagedates',
        params={
            'series_id': series_id,
            'api_key': api_key,
            'file_type': 'json',
        },
        timeout=30.0,
    )
    return r.json().get('vintage_dates', [])


def get_observations_for_vintages(
    client: httpx.Client,
    series_id: str,
    api_key: str,
    vintage_dates: list[str],
    output_type: int = 2,
    chunk_size: int = 200,
    observation_start: str | None = None,
) -> pl.DataFrame:
    """Pull observations for a set of vintage dates (ALFRED wide format)."""
    frames: list[pl.DataFrame] = []
    for chunk in _chunked(vintage_dates, chunk_size):
        params = {
            'series_id': series_id,
            'api_key': api_key,
            'file_type': 'json',
            'vintage_dates': ','.join(chunk),
            'output_type': str(output_type),
        }
        if observation_start is not None:
            params['observation_start'] = observation_start
        r = _request_with_retry(
            client,
            f'{FRED_BASE}/series/observations',
            params=params,
            timeout=60.0,
        )
        obs = r.json().get('observations', [])
        if obs:
            frames.append(pl.from_dicts(obs))

    if not frames:
        return pl.DataFrame(schema={'date': pl.Utf8})
    return pl.concat(frames, how='vertical_relaxed')


def compute_initial_and_latest_levels(df_wide: pl.DataFrame) -> pl.DataFrame:
    """Extract initial and latest values per date from ALFRED wide format."""
    vintage_cols = [c for c in df_wide.columns if c != 'date']
    if not vintage_cols:
        return pl.DataFrame(
            schema={
                'date': pl.Date,
                'initial_level': pl.Float64,
                'latest_level': pl.Float64,
            }
        )

    def _to_float(col_name: str) -> pl.Expr:
        return (
            pl.col(col_name)
            .cast(pl.Utf8)
            .str.replace_all(r'^\.$', '')
            .str.strip_chars()
            .cast(pl.Float64, strict=False)
        )

    df = df_wide.with_columns(
        pl.col('date').str.to_date(strict=False).alias('date'),
        *[_to_float(c).alias(c) for c in vintage_cols],
    ).filter(pl.col('date').is_not_null())

    return (
        df.with_columns(
            pl.coalesce([pl.col(c) for c in vintage_cols]).alias('initial_level'),
            pl.coalesce([pl.col(c) for c in reversed(vintage_cols)]).alias(
                'latest_level'
            ),
        )
        .select('date', 'initial_level', 'latest_level')
        .sort('date')
    )


# ---------------------------------------------------------------------------
# Series construction
# ---------------------------------------------------------------------------


def _make_series_id(
    fips: str, abbrev: str, ces_code: str, nsa_suffix: str | None, adjusted: bool,
) -> str:
    """Build the FRED/ALFRED series ID for a state x industry combination.

    Two naming conventions exist on FRED:
    - Short form (``{ABBREV}{SUFFIX}``) when *nsa_suffix* is not ``None``.
    - Long form (``SMU{FIPS}00000{CES_CODE}01`` or ``SMS...``).
    """
    if nsa_suffix is not None:
        suffix = nsa_suffix[:-1] if adjusted else nsa_suffix
        return f'{abbrev}{suffix}'
    prefix = 'SMS' if adjusted else 'SMU'
    return f'{prefix}{fips}00000{ces_code}01'


def build_series_df() -> pl.DataFrame:
    """Build a DataFrame of all state x industry x adjustment series to fetch.

    Returns
    -------
    pl.DataFrame
        Columns: series_id, adjusted, geographic_type, geographic_code,
        state_fips, state_abbrev, ces_industry, industry_type, industry_code,
        industry_name.
    """
    rows = []
    for a in ['SA', 'NSA']:
        adjusted = a == 'SA'
        for fips, abbrev in FIPS_TO_ABBREV.items():
            for ces_code, code, name, level, nsa_suffix in INDUSTRIES:
                rows.append(
                    {
                        'series_id': _make_series_id(
                            fips, abbrev, ces_code, nsa_suffix, adjusted,
                        ),
                        'adjusted': adjusted,
                        'geographic_type': 'state',
                        'geographic_code': fips,
                        'state_fips': fips,
                        'state_abbrev': abbrev,
                        'ces_industry': ces_code,
                        'industry_type': level,
                        'industry_code': code,
                        'industry_name': name,
                    }
                )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Batch download
# ---------------------------------------------------------------------------


def _save_checkpoint(results: list[pl.DataFrame], path: Path) -> None:
    """Write intermediate results to a Parquet checkpoint file."""
    if not results:
        return
    pl.concat(results, how='vertical_relaxed').write_parquet(path)
    print(f'  ** checkpoint saved ({len(results)} series) -> {path}')


def fetch_batch_sae_revisions(
    series_df: pl.DataFrame,
    fred_api_key: str,
    last_n_vintages: int = 24,
    chunk_size: int = 200,
    sleep_between: float = 1.0,
    observation_start: str | None = None,
    checkpoint_every: int = 100,
    checkpoint_path: Path = CHECKPOINT_PATH,
) -> pl.DataFrame:
    """Fetch initial-vs-latest levels for every series in *series_df*.

    Series that don't exist on FRED (HTTP 400/404) are silently skipped.
    Intermediate results are checkpointed every *checkpoint_every* series.
    """
    results: list[pl.DataFrame] = []
    skipped: list[str] = []
    done_ids: set[str] = set()

    if checkpoint_path.exists():
        prev = pl.read_parquet(checkpoint_path)
        if prev.height > 0:
            done_ids = set(prev['series_id'].unique().to_list())
            results.append(prev)
            print(f'  Resumed from checkpoint: {len(done_ids)} series already fetched')

    total = len(series_df)
    fetched_this_run = 0

    with httpx.Client() as client:
        for i, row in enumerate(series_df.iter_rows(named=True)):
            sid = row['series_id']
            if sid in done_ids:
                continue
            try:
                vdates = get_vintage_dates(
                    client, series_id=sid, api_key=fred_api_key,
                )
                if last_n_vintages is not None:
                    vdates = vdates[-last_n_vintages:]
                if not vdates:
                    continue

                df_obs = get_observations_for_vintages(
                    client,
                    series_id=sid,
                    api_key=fred_api_key,
                    vintage_dates=vdates,
                    output_type=2,
                    chunk_size=chunk_size,
                    observation_start=observation_start,
                )
                levels = compute_initial_and_latest_levels(df_obs)
                if levels.height > 0:
                    levels = levels.with_columns(
                        pl.lit(sid).alias('series_id'),
                        pl.lit(row['adjusted']).alias('adjusted'),
                        pl.lit(row['state_fips']).alias('state_fips'),
                        pl.lit(row['state_abbrev']).alias('state_abbrev'),
                        pl.lit(row['ces_industry']).alias('ces_industry'),
                        pl.lit(row['geographic_type']).alias('geographic_type'),
                        pl.lit(row['geographic_code']).alias('geographic_code'),
                        pl.lit(row['industry_name']).alias('industry_name'),
                        pl.lit(row['industry_type']).alias('industry_type'),
                        pl.lit(row['industry_code']).alias('industry_code'),
                    )
                    results.append(levels)
                    fetched_this_run += 1

                if fetched_this_run > 0 and fetched_this_run % checkpoint_every == 0:
                    _save_checkpoint(results, checkpoint_path)

                if (i + 1) % 25 == 0:
                    print(
                        f'  [{i + 1}/{total}] fetched {len(results)} series '
                        f'so far ({len(skipped)} skipped)'
                    )

            except httpx.HTTPStatusError as e:
                if e.response.status_code in (400, 404):
                    skipped.append(sid)
                else:
                    _save_checkpoint(results, checkpoint_path)
                    raise

            except Exception:
                _save_checkpoint(results, checkpoint_path)
                raise

            time.sleep(sleep_between)

    print(f'\nDone: {len(results)} series fetched, {len(skipped)} not found on FRED')
    if skipped:
        print(f'  Skipped (first 20): {skipped[:20]}')

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    if not results:
        return pl.DataFrame()

    return (
        pl.concat(results, how='vertical_relaxed').select(
            ref_date=pl.col('date').dt.offset_by('11d'),
            adjusted=pl.col('adjusted'),
            geographic_type=pl.col('geographic_type'),
            geographic_code=pl.col('geographic_code'),
            industry_type=pl.col('industry_type'),
            industry_code=pl.col('industry_code'),
            employment_initial=pl.col('initial_level'),
            employment_latest=pl.col('latest_level'),
        )
    )


# ---------------------------------------------------------------------------
# Processing helpers
# ---------------------------------------------------------------------------


def _split_revisions(
    sae_revisions: pl.DataFrame, *, adjusted_value: bool,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split into revision-0 (initial) and revision-1 (latest) for one SA/NSA slice."""
    subset = sae_revisions.filter(pl.col('adjusted').eq(adjusted_value))

    rev_0 = (
        subset.rename(
            {'adjusted': 'seasonally_adjusted', 'employment_initial': 'employment'},
        )
        .drop('employment_latest')
        .with_columns(revision=pl.lit(0, pl.UInt8))
        .filter(pl.col('employment').is_not_null())
    )

    rev_1 = (
        subset.rename(
            {'adjusted': 'seasonally_adjusted', 'employment_latest': 'employment'},
        )
        .drop('employment_initial')
        .with_columns(revision=pl.lit(1, pl.UInt8))
        .filter(pl.col('employment').is_not_null())
    )

    return rev_0, rev_1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


_SECTOR_RECODE = {
    ces: naics
    for ces, naics in CES_SECTOR_TO_NAICS.items()
    if ces != naics
}


def _recode_to_naics(df: pl.DataFrame) -> pl.DataFrame:
    """Recode CES industry codes to NAICS equivalents and add sector aliases.

    1. Remap CES sector codes (41→42, 42→44, 43→48) to match QCEW NAICS codes.
    2. Duplicate single-sector supersectors (20, 50, 80) as sector rows (23, 51, 81).
    """
    if _SECTOR_RECODE:
        df = df.with_columns(
            pl.when(
                (pl.col('industry_type') == 'sector')
                & pl.col('industry_code').is_in(list(_SECTOR_RECODE.keys()))
            )
            .then(pl.col('industry_code').replace(_SECTOR_RECODE))
            .otherwise(pl.col('industry_code'))
            .alias('industry_code')
        )

    sector_aliases = df.filter(
        (pl.col('industry_type') == 'supersector')
        & pl.col('industry_code').is_in(list(SINGLE_SECTOR_SUPERSECTORS.keys()))
    ).with_columns(
        industry_type=pl.lit('sector'),
        industry_code=pl.col('industry_code').replace(SINGLE_SECTOR_SUPERSECTORS),
    )

    return pl.concat([df, sector_aliases])


def main() -> None:
    """Fetch all SAE series from ALFRED, process, and write Parquet.

    Requires ``FRED_API_KEY`` in the environment.
    """
    fred_api_key = os.environ.get('FRED_API_KEY', '')
    if not fred_api_key:
        raise RuntimeError('FRED_API_KEY environment variable is required')
    obs_start = '2016-01-01'

    series_df = build_series_df()
    print(
        f'{len(series_df)} series | '
        f'{series_df["state_fips"].n_unique()} states | '
        f'{series_df["ces_industry"].n_unique()} industries'
    )

    sae_revisions = fetch_batch_sae_revisions(
        series_df,
        fred_api_key=fred_api_key,
        observation_start=obs_start,
    )

    nsa_0, nsa_1 = _split_revisions(sae_revisions, adjusted_value=False)
    sa_0, sa_1 = _split_revisions(sae_revisions, adjusted_value=True)

    print(f'Number of SAE NSA revision 0 observations: {nsa_0.height:,}')
    print(f'Number of SAE NSA revision 1 observations: {nsa_1.height:,}')
    print(f'Number of SAE SA revision 0 observations: {sa_0.height:,}')
    print(f'Number of SAE SA revision 1 observations: {sa_1.height:,}')

    sae_long = pl.concat([nsa_0, nsa_1, sa_0, sa_1]).with_columns(
        source=pl.lit('sae', pl.Utf8),
    )

    vintage_dates = (
        pl.read_parquet(VINTAGE_DATES_PATH, storage_options=storage_options_for(VINTAGE_DATES_PATH))
        .filter(pl.col('publication').eq('sae'))
        .select(
            ref_date=pl.col('ref_date'),
            revision=pl.col('revision').cast(pl.UInt8),
            benchmark_revision=pl.col('benchmark_revision').cast(pl.UInt8),
            vintage_date=pl.col('vintage_date'),
        )
    )

    sae_df = (
        sae_long.join(vintage_dates, on=['ref_date', 'revision'], how='left')
        .select(
            'source', 'seasonally_adjusted',
            'geographic_type', 'geographic_code',
            'industry_type', 'industry_code',
            'ref_date', 'vintage_date',
            'revision', 'benchmark_revision',
            'employment',
        )
        .filter(pl.col('vintage_date').is_not_null())
    )

    sae_df = _recode_to_naics(sae_df)

    print(f'Number of SAE revision observations (w/ dates): {sae_df.height:,}')

    # Uniqueness assertion
    dedup = sae_df.unique(subset=[
        'source', 'seasonally_adjusted',
        'geographic_type', 'geographic_code',
        'industry_type', 'industry_code',
        'ref_date', 'vintage_date',
        'revision', 'benchmark_revision',
    ])
    assert sae_df.height == dedup.height, (
        f'Duplicate rows: {sae_df.height} total vs {dedup.height} unique'
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sae_df.write_parquet(OUTPUT_PATH)
    print(f'Wrote {sae_df.height:,} rows to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
