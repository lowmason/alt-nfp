"""Tag estimate DataFrames with vintage_date, revision, and benchmark_revision.

Reads vintage_dates (from :mod:`alt_nfp.ingest.release_dates`), computes the
latest vintage lookup per publication/ref_date, and joins onto estimate
DataFrames. Can optionally append the tagged rows to the vintage store.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
from nfp_download.release_dates.config import VINTAGE_DATES_PATH

logger = logging.getLogger(__name__)


def latest_vintage_lookup(
    vintage_df: pl.DataFrame,
    publication: str,
) -> pl.DataFrame:
    """Per ref_date, take max revision, max benchmark_revision, max vintage_date.

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
        One row per ref_date with: ref_date, vintage_date, revision,
        benchmark_revision.
    """
    return (
        vintage_df.filter(pl.col('publication') == publication)
        .group_by('ref_date')
        .agg(
            pl.col('vintage_date').max().alias('vintage_date'),
            pl.col('revision').max().alias('revision'),
            pl.col('benchmark_revision').max().alias('benchmark_revision'),
        )
        .with_columns(
            pl.col('revision').cast(pl.UInt8),
            pl.col('benchmark_revision').cast(pl.UInt8),
        )
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
    vintage_df = pl.read_parquet(path)
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
