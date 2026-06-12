"""Aggregate state-level estimates to Census Regions and Divisions.

Takes DataFrames with national and state-level rows, adds region and division
rows by mapping state FIPS to Census Region/Division (via
:mod:`nfp_lookups.geography`) and summing employment within each group.
"""

from __future__ import annotations

import polars as pl
from nfp_lookups.geography import FIPS_TO_DIVISION, FIPS_TO_REGION

# Columns that define a unique observation (used for group_by).
_GROUP_COLS: list[str] = [
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
]


def aggregate_geo(
    df: pl.DataFrame,
    group_cols: list[str] | None = None,
) -> pl.DataFrame:
    """Append region and division rows to a DataFrame with state-level data.

    Keeps existing national and state rows, then creates region and division
    aggregate rows by summing employment. Puerto Rico is included in Region 3
    / Division 05.

    Parameters
    ----------
    df : pl.DataFrame
        Estimates DataFrame with ``geographic_type``, ``geographic_code``,
        and ``employment`` columns. State rows should have
        ``geographic_type='state'`` and ``geographic_code`` set to the
        2-digit state FIPS.
    group_cols : list[str] or None
        Columns to group by for aggregation. Defaults to :data:`_GROUP_COLS`.
        Must include ``geographic_type`` and ``geographic_code``.

    Returns
    -------
    pl.DataFrame
        Input DataFrame with added region and division rows, sorted by
        group_cols.
    """
    if group_cols is None:
        # Use only columns that exist in the DataFrame
        group_cols = [c for c in _GROUP_COLS if c in df.columns]

    national = df.filter(pl.col('geographic_type') == 'national')
    state = df.filter(
        (pl.col('geographic_type') == 'state')
        & (pl.col('geographic_code') != '00')
    )

    region_df = (
        state.with_columns(
            geographic_type=pl.lit('region', pl.Utf8),
            geographic_code=pl.col('geographic_code').replace_strict(
                FIPS_TO_REGION, default=None,
            ),
        )
        .filter(pl.col('geographic_code').is_not_null())
        .group_by(group_cols)
        .agg(employment=pl.col('employment').sum())
    )

    division_df = (
        state.with_columns(
            geographic_type=pl.lit('division', pl.Utf8),
            geographic_code=pl.col('geographic_code').replace_strict(
                FIPS_TO_DIVISION, default=None,
            ),
        )
        .filter(pl.col('geographic_code').is_not_null())
        .group_by(group_cols)
        .agg(employment=pl.col('employment').sum())
    )

    return pl.concat(
        [national, state, region_df, division_df], how='diagonal',
    ).sort(group_cols)
