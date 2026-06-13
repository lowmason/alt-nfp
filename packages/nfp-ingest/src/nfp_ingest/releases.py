"""Combine estimate parquets into a unified releases.parquet.

Produces a levels-based, source-unified file suitable for revision analysis
and diagnostics. Unlike the growth-rate-oriented ``observation_panel.parquet``,
this file stores employment **levels** with full vintage/revision metadata.

Can also fetch current CES estimates directly from BLS flat files via
:func:`build_releases`.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import polars as pl
from nfp_lookups.paths import DATA_DIR, VINTAGE_DATES_PATH

logger = logging.getLogger(__name__)

RELEASES_PATH: Path = DATA_DIR / 'releases.parquet'

COMBINED_COLUMNS: list[str] = [
    'source',
    'seasonally_adjusted',
    'geographic_type',
    'geographic_code',
    'industry_type',
    'industry_code',
    'ref_date',
    'vintage_date',
    'revision',
    'benchmark_revision',
    'employment',
]

COMBINED_SCHEMA: dict[str, pl.DataType] = {
    'source': pl.Utf8,
    'seasonally_adjusted': pl.Boolean,
    'geographic_type': pl.Utf8,
    'geographic_code': pl.Utf8,
    'industry_type': pl.Utf8,
    'industry_code': pl.Utf8,
    'ref_date': pl.Date,
    'vintage_date': pl.Date,
    'revision': pl.UInt8,
    'benchmark_revision': pl.UInt8,
    'employment': pl.Float64,
}

_DOMAIN_CODES = frozenset({'05', '06', '07', '08'})


def _latest_ces_vintage_dates() -> pl.DataFrame:
    """Per ref_date, the latest *coherent* calendar row in each benchmark track.

    Reads ``vintage_dates.parquet``, filters to CES, and for every
    ``(ref_date, benchmark_revision)`` track keeps the most recently published
    calendar row, holding its ``(vintage_date, revision)`` as a coherent pair
    drawn from a single actual row.

    This mirrors :func:`nfp_ingest.tagger.latest_vintage_lookup`. Keying only
    on ``ref_date`` (the prior behavior) drops a row on annual-benchmark
    release days: the CES benchmark lands with the January first print each
    February, so the prior December carries two near-same-day rows — the
    ordinary second print ``(revision=1, benchmark_revision=0)`` and the
    benchmark reprint ``(revision=2, benchmark_revision=1)``. The benchmark
    row has the later ``vintage_date`` and shadows the rev-1 print, the data
    loss documented in ``specs/ces_growth_convention.md`` §4(c). Treating each
    ``benchmark_revision`` track separately emits **both** rows, so a
    benchmarked ref_date appears on more than one row.
    """
    if not VINTAGE_DATES_PATH.exists():
        return pl.DataFrame(
            schema={
                'ref_date': pl.Date,
                'vintage_date': pl.Date,
                'revision': pl.UInt8,
                'benchmark_revision': pl.UInt8,
            }
        )

    return (
        pl.read_parquet(VINTAGE_DATES_PATH)
        .filter(pl.col('publication') == 'ces')
        # Latest published row wins within each track; break vintage_date ties
        # by revision so the kept row stays a coherent (vintage, revision) pair.
        .sort(['vintage_date', 'revision'], descending=True)
        .unique(subset=['ref_date', 'benchmark_revision'], keep='first')
        .select(
            'ref_date',
            'vintage_date',
            pl.col('revision').cast(pl.UInt8),
            pl.col('benchmark_revision').cast(pl.UInt8),
        )
        .sort(['ref_date', 'benchmark_revision'])
    )


def _fetch_ces_releases() -> pl.DataFrame:
    """Fetch current CES national estimates from BLS.

    Tries the flat-file download first; falls back to the JSON API when
    an API key is available.

    Returns a DataFrame in :data:`COMBINED_SCHEMA` with ``vintage_date``,
    ``revision``, and ``benchmark_revision`` sourced from
    ``vintage_dates.parquet`` (the latest published revision per ref_date).
    """
    import os

    from .bls import BLSHttpClient
    from .bls.ces_national import CES_SERIES_MAP, fetch_ces_national, fetch_ces_national_via_api

    api_key = os.environ.get('BLS_API_KEY')
    client = BLSHttpClient(api_key=api_key)
    try:
        try:
            raw = fetch_ces_national(client=client)
        except Exception as exc:
            logger.info(f'Flat-file download failed ({exc}), trying JSON API')
            if not api_key:
                logger.warning('No BLS_API_KEY set — cannot fall back to JSON API')
                return pl.DataFrame(schema=COMBINED_SCHEMA)
            all_ids = [
                sid
                for pair in CES_SERIES_MAP.values()
                for sid in pair.values()
            ]
            today = date.today()
            raw = fetch_ces_national_via_api(
                all_ids,
                start_year=2003,
                end_year=today.year,
                client=client,
            )
            if not raw.is_empty():
                raw = raw.with_columns([
                    raw['series_id'].str.slice(3, 2).alias('supersector_code'),
                    pl.when(raw['series_id'].str.slice(2, 1) == 'S')
                    .then(pl.lit(True))
                    .otherwise(pl.lit(False))
                    .alias('is_seasonally_adjusted'),
                ])
    finally:
        client.close()

    if raw.is_empty():
        logger.warning('No CES data returned from BLS flat files')
        return pl.DataFrame(schema=COMBINED_SCHEMA)

    latest_vdates = _latest_ces_vintage_dates()

    return (
        raw.select(
            source=pl.lit('ces'),
            seasonally_adjusted=pl.col('is_seasonally_adjusted'),
            geographic_type=pl.lit('national'),
            geographic_code=pl.lit('00'),
            industry_type=(
                pl.when(pl.col('supersector_code') == '00')
                .then(pl.lit('national'))
                .when(pl.col('supersector_code').is_in(list(_DOMAIN_CODES)))
                .then(pl.lit('domain'))
                .otherwise(pl.lit('supersector'))
            ),
            industry_code=pl.col('supersector_code'),
            ref_date=pl.col('date'),
            employment=pl.col('value'),
        )
        .join(latest_vdates, on='ref_date', how='left')
        .cast(COMBINED_SCHEMA)
    )


def build_releases(out_path: Path | None = None) -> pl.DataFrame:
    """Fetch current BLS estimates and write ``releases.parquet``.

    Currently fetches CES national data from BLS flat files. Additional
    sources (QCEW, SAE) can be added by extending this function.

    Parameters
    ----------
    out_path : Path or None
        Output path. Defaults to ``data/releases.parquet``.

    Returns
    -------
    pl.DataFrame
        Combined releases DataFrame with :data:`COMBINED_SCHEMA`.
    """
    out_path = out_path or RELEASES_PATH

    frames: list[pl.DataFrame] = []

    print('  Fetching current CES national estimates from BLS...')
    ces = _fetch_ces_releases()
    if not ces.is_empty():
        frames.append(ces)
        print(f'  CES: {ces.height:,} rows')

    if not frames:
        out_df = pl.DataFrame(schema=COMBINED_SCHEMA)
    else:
        out_df = pl.concat(frames, how='diagonal_relaxed')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(out_path)
    print(f'  Wrote {out_df.height:,} rows to {out_path}')
    return out_df


def combine_estimates(
    *paths: Path,
    out_path: Path | None = None,
) -> pl.DataFrame:
    """Combine estimate parquets into a single releases DataFrame.

    Reads each estimate file (if present), keeps only :data:`COMBINED_COLUMNS`,
    casts to :data:`COMBINED_SCHEMA`, concatenates, and writes to ``out_path``.

    Parameters
    ----------
    *paths : Path
        Paths to estimate parquet files (QCEW, CES, SAE, etc.). Non-existent
        files are silently skipped.
    out_path : Path or None
        Output path. Defaults to ``data/releases.parquet``.

    Returns
    -------
    pl.DataFrame
        Combined DataFrame with :data:`COMBINED_SCHEMA`.
    """
    out_path = out_path or RELEASES_PATH

    frames: list[pl.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        df = pl.read_parquet(path)
        if df.is_empty():
            continue

        # Select only columns that exist
        select_cols = [c for c in COMBINED_COLUMNS if c in df.columns]
        df = df.select(select_cols)

        # Fill missing vintage columns
        for c in COMBINED_COLUMNS:
            if c not in df.columns:
                if c == 'vintage_date':
                    df = df.with_columns(pl.lit(None).cast(pl.Date).alias(c))
                elif c in ('revision', 'benchmark_revision'):
                    df = df.with_columns(pl.lit(None).cast(pl.UInt8).alias(c))
                elif c == 'seasonally_adjusted':
                    df = df.with_columns(pl.lit(None).cast(pl.Boolean).alias(c))
                else:
                    raise ValueError(f'Missing required column {c} in {path}')

        df = df.select(COMBINED_COLUMNS).cast(COMBINED_SCHEMA)
        frames.append(df)

    if not frames:
        out_df = pl.DataFrame(schema=COMBINED_SCHEMA)
    else:
        out_df = pl.concat(frames, how='diagonal_relaxed')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(out_path)
    return out_df
