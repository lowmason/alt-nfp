"""Tag estimate DataFrames with vintage_date, revision, and benchmark_revision.

Reads vintage_dates (from :mod:`nfp_ingest.release_dates`), computes the
latest vintage lookup per publication/ref_date, and joins onto estimate
DataFrames. Can optionally append the tagged rows to the vintage store.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
from nfp_lookups.paths import VINTAGE_DATES_PATH, storage_options_for

logger = logging.getLogger(__name__)


def latest_vintage_lookup(
    vintage_df: pl.DataFrame,
    publication: str,
) -> pl.DataFrame:
    """Per ref_date, the latest *coherent* calendar row in each benchmark track.

    For every ``(ref_date, benchmark_revision)`` track, return the most
    recently published calendar row, keeping its ``(vintage_date, revision)``
    as a coherent triple drawn from a single actual row.

    This replaces an earlier implementation that took *independent* maxes of
    ``vintage_date``, ``revision``, and ``benchmark_revision`` per ref_date.
    On an annual-benchmark release day (the CES benchmark lands with the
    January first print each February) the calendar carries two same-day rows
    for the prior December: the ordinary second print
    ``(revision=1, benchmark_revision=0)`` and the benchmark reprint
    ``(revision=2, benchmark_revision=1)``. The column-wise max fused them
    into one incoherent ``(revision=2, benchmark_revision=1)`` tag, so the
    rev-1 level row was never appended to the vintage store — and it is
    unrecoverable from later captures, because the append anti-joins on the
    ``(revision, benchmark_revision)`` uniqueness key.

    By treating each ``benchmark_revision`` track separately, a ref_date that
    is republished both as an ordinary print and as a benchmark reprint now
    yields **both** rows, so both level rows reach the store. Tagging only
    changes which rows *future* captures emit; it never reinterprets rows
    already written to the append-only store.

    Parameters
    ----------
    vintage_df : pl.DataFrame
        DataFrame with columns: publication, ref_date, vintage_date, revision,
        benchmark_revision.
    publication : str
        One of ``'ces'``, ``'sae'``, ``'qcew'``.

    Returns
    -------
    pl.DataFrame
        One row per ``(ref_date, benchmark_revision)`` track with columns:
        ref_date, vintage_date, revision, benchmark_revision. Benchmarked
        ref_dates therefore appear on more than one row.
    """
    return (
        vintage_df.filter(pl.col('publication') == publication)
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


def tag_estimates(
    estimates_df: pl.DataFrame,
    publication: str,
    vintage_dates_path: Path | None = None,
) -> pl.DataFrame:
    """Tag an estimates DataFrame with vintage_date, revision, benchmark_revision.

    Joins the latest vintage lookup for the given publication onto the estimates
    DataFrame on ``ref_date``. Existing vintage columns (if any) are replaced.

    Parameters
    ----------
    estimates_df : pl.DataFrame
        Estimates DataFrame with a ``ref_date`` column.
    publication : str
        Publication name (``'ces'``, ``'sae'``, ``'qcew'``).
    vintage_dates_path : Path or None
        Path to vintage_dates.parquet. Defaults to the configured path.

    Returns
    -------
    pl.DataFrame
        Input DataFrame with vintage_date, revision, and benchmark_revision
        columns added (or replaced).
    """
    path = vintage_dates_path or VINTAGE_DATES_PATH
    if not path.exists():
        raise FileNotFoundError(
            f'Vintage dates file not found: {path}. '
            'Run the release_dates pipeline first to create it.'
        )
    vintage_df = pl.read_parquet(path, storage_options=storage_options_for(path))
    lookup = latest_vintage_lookup(vintage_df, publication)

    # Drop existing vintage columns to avoid duplicates
    for col in ('vintage_date', 'revision', 'benchmark_revision'):
        if col in estimates_df.columns:
            estimates_df = estimates_df.drop(col)

    return estimates_df.join(lookup, on='ref_date', how='left')


def tag_and_append(
    estimates_df: pl.DataFrame,
    publication: str,
    vintage_dates_path: Path | None = None,
    vintage_store_path: Path | None = None,
) -> pl.DataFrame:
    """Tag estimates and optionally append to the vintage store.

    Parameters
    ----------
    estimates_df : pl.DataFrame
        Estimates DataFrame with a ``ref_date`` column.
    publication : str
        Publication name (``'ces'``, ``'sae'``, ``'qcew'``).
    vintage_dates_path : Path or None
        Path to vintage_dates.parquet.
    vintage_store_path : Path or None
        Path to the vintage store. If provided, tagged rows are appended.

    Returns
    -------
    pl.DataFrame
        Tagged estimates DataFrame.
    """
    tagged = tag_estimates(estimates_df, publication, vintage_dates_path)

    if vintage_store_path is not None:
        from nfp_ingest.vintage_store import append_to_vintage_store

        append_to_vintage_store(tagged, vintage_store_path)
        logger.info(
            'Appended %d tagged %s rows to vintage store',
            tagged.height, publication,
        )

    return tagged
