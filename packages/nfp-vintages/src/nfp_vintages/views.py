"""Vintage view functions over the observation panel.

These are lazy views (LazyFrame -> LazyFrame) that filter the panel to
specific vintages without materializing separate tables.
"""

from __future__ import annotations

from datetime import date

import polars as pl


def real_time_view(panel: pl.LazyFrame, as_of: date) -> pl.LazyFrame:
    """Return the latest vintage available for each observation as of a given date.

    For each (period, source, industry_code), keeps the row with the highest
    revision_number whose vintage_date <= as_of.

    Parameters
    ----------
    panel : pl.LazyFrame
        Full observation panel.
    as_of : date
        Information set cutoff date.

    Returns
    -------
    pl.LazyFrame
        Filtered panel with one row per (period, source, industry_code).
    """
    return (
        panel.filter(pl.col('vintage_date') <= as_of)
        .sort('revision_number', descending=True)
        .unique(subset=['period', 'source', 'industry_code'], keep='first')
        .sort('period', 'source', 'industry_code')
    )


def final_view(panel: pl.LazyFrame) -> pl.LazyFrame:
    """Return only final-vintage rows from the observation panel.

    Parameters
    ----------
    panel : pl.LazyFrame
        Full observation panel.

    Returns
    -------
    pl.LazyFrame
        Rows where is_final == True.
    """
    return panel.filter(pl.col('is_final') == True).sort(  # noqa: E712
        'period', 'source', 'industry_code'
    )


def specific_vintage_view(
    panel: pl.LazyFrame,
    source: str,
    revision_number: int,
) -> pl.LazyFrame:
    """Filter to a specific source and revision stage.

    Useful for studying revision properties of a single vintage.

    Parameters
    ----------
    panel : pl.LazyFrame
        Full observation panel.
    source : str
        Source identifier (e.g., 'ces_sa', 'qcew').
    revision_number : int
        Revision number to select (0 = initial, 1+ = subsequent, -1 = benchmark).

    Returns
    -------
    pl.LazyFrame
        Filtered panel.
    """
    return (
        panel.filter(
            (pl.col('source') == source) & (pl.col('revision_number') == revision_number)
        )
        .sort('period', 'industry_code')
    )
