"""Build CES vintage-store rows from ALFRED for the frontier-patch window.

Implements the spec §5 extraction (1st/2nd/3rd appearance as-published, no
value-dedup, real-time guard) and shapes ``VINTAGE_STORE_SCHEMA`` rows whose
``vintage_date`` comes from the existing release calendar (values from ALFRED,
dates from the schedule). Spec: ``specs/alfred_ces_vintages.md``.
"""
from __future__ import annotations

import polars as pl


def extract_prints(matrix: pl.DataFrame, *, max_gap_days: int = 70) -> pl.DataFrame:
    """Extract the three monthly prints ``(0,0)/(1,0)/(2,0)`` from a vintage matrix.

    The 1st/2nd/3rd appearance of each ref month (in ``vintage_date`` order, **no
    value-dedup**) is revision 0/1/2. The **real-time guard** keeps a ref month
    only when its first appearance lands within *max_gap_days* of ``ref_date`` —
    dropping back-history artifacts (a shallow series' first archived vintage
    carries years-old history).

    Parameters
    ----------
    matrix : pl.DataFrame
        Long frame ``(ref_date: Date, vintage_date: Date, value: Float64)``.
    max_gap_days : int
        Maximum ``vintage_date - ref_date`` (days) for a genuine first print.

    Returns
    -------
    pl.DataFrame
        ``(ref_date, revision: UInt8, vintage_date, value)`` for ``revision ∈ {0,1,2}``.
    """
    ranked = matrix.sort("ref_date", "vintage_date").with_columns(
        pl.col("vintage_date").rank("ordinal").over("ref_date").alias("_rk")
    )
    prints = ranked.filter(pl.col("_rk") <= 3).with_columns(
        (pl.col("_rk") - 1).cast(pl.UInt8).alias("revision")
    )
    genuine = (
        prints.filter(pl.col("revision") == 0)
        .filter(
            (pl.col("vintage_date") - pl.col("ref_date")).dt.total_days() <= max_gap_days
        )
        .select("ref_date")
    )
    return (
        prints.join(genuine, on="ref_date", how="inner")
        .select("ref_date", "revision", "vintage_date", "value")
        .sort("ref_date", "revision")
    )
