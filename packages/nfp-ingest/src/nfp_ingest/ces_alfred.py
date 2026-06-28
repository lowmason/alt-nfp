"""Build CES vintage-store rows from ALFRED for the frontier-patch window.

Implements the spec §5 extraction (1st/2nd/3rd appearance as-published, no
value-dedup, real-time guard) and shapes ``VINTAGE_STORE_SCHEMA`` rows whose
``vintage_date`` comes from the existing release calendar (values from ALFRED,
dates from the schedule). Spec: ``specs/alfred_ces_vintages.md``.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import date

import httpx
import polars as pl
from nfp_download.alfred import (
    CES_SERIES_NSA,
    CES_SERIES_SA,
    fetch_vintage_matrix,
    get_vintage_dates,
    verify_ces_series,
)
from nfp_lookups.industry import ownership_for
from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA


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


def _default_fetch(api_key: str) -> Callable[..., pl.DataFrame]:
    """Return a real-ALFRED fetch closure ``(series_id, *, sa, key) -> matrix``.

    Title-verifies through the public ``verify_ces_series`` (raising on a SA-flag
    or concept-substring mismatch — no download-private import), pulls vintage
    dates, and returns the long vintage matrix. Shares one client.
    """
    client = httpx.Client(http2=True)

    def fetch(series_id: str, *, sa: bool, key: tuple[str, str]) -> pl.DataFrame:
        itype, code = key
        title, ok = verify_ces_series(client, itype, code, sa=sa, api_key=api_key)
        if not ok:
            raise ValueError(
                f"title-verify failed for {series_id} ({key}): title={title!r}"
            )
        obs_start = "2024-01-01"
        vds = get_vintage_dates(client, series_id, api_key=api_key, start=obs_start)
        return fetch_vintage_matrix(
            client, series_id, api_key=api_key, vintage_dates=vds, observation_start=obs_start
        )

    return fetch


def build_ces_alfred_window(
    *,
    store_frontier: date,
    through: date,
    calendar: pl.DataFrame,
    api_key: str,
    adjustments: tuple[bool, ...] = (True, False),
    keys: list[tuple[str, str]] | None = None,
    fetch: Callable[..., pl.DataFrame] | None = None,
) -> pl.DataFrame:
    """Build ``VINTAGE_STORE_SCHEMA`` rows for the cohorts ALFRED must patch.

    The window is the calendar's CES ``benchmark_revision=0`` cohorts with
    ``store_frontier < vintage_date <= through``. For each resolved series the
    §5 prints are extracted and joined to the window on ``(ref-month, revision)``
    — values from ALFRED, ``vintage_date``/``ref_date`` from the calendar.

    Parameters
    ----------
    store_frontier : datetime.date
        The store's current max CES ``vintage_date``; cohorts ``<=`` it are skipped.
    through : datetime.date
        Upper bound on the window's ``vintage_date`` (typically today).
    calendar : pl.DataFrame
        Release calendar with ``publication, ref_date, revision, benchmark_revision,
        vintage_date`` (e.g. ``vintage_dates.parquet``).
    api_key : str
        FRED API key (used only by the default fetch).
    adjustments : tuple[bool, ...]
        Which seasonal adjustments to build (``True`` SA, ``False`` NSA).
    keys : list[tuple[str, str]] or None
        Restrict to these ``(industry_type, industry_code)`` keys (default: all 30).
    fetch : Callable or None
        ``(series_id, *, sa, key) -> long matrix``; defaults to real ALFRED.

    Returns
    -------
    pl.DataFrame
        Rows conforming to ``VINTAGE_STORE_SCHEMA`` (may be empty).
    """
    fetch = fetch or _default_fetch(api_key)

    window = (
        calendar.filter(
            (pl.col("publication") == "ces")
            & (pl.col("benchmark_revision") == 0)
            & pl.col("revision").is_in([0, 1, 2])
            & (pl.col("vintage_date") > store_frontier)
            & (pl.col("vintage_date") <= through)
        )
        .select(
            "ref_date",
            "revision",
            "vintage_date",
            pl.col("ref_date").dt.truncate("1mo").alias("_m"),
        )
    )
    if window.is_empty():
        return pl.DataFrame(schema=VINTAGE_STORE_SCHEMA)

    out: list[pl.DataFrame] = []
    for sa in adjustments:
        table = CES_SERIES_SA if sa else CES_SERIES_NSA
        for key, series_id in table.items():
            if keys is not None and key not in keys:
                continue
            itype, code = key
            matrix = fetch(series_id, sa=sa, key=key)
            prints = extract_prints(matrix).with_columns(
                pl.col("ref_date").dt.truncate("1mo").alias("_m")
            )
            joined = window.join(
                prints.select("_m", "revision", "value"), on=["_m", "revision"], how="inner"
            )
            if joined.is_empty():
                continue
            out.append(
                joined.with_columns(
                    pl.lit("national").alias("geographic_type"),
                    pl.lit("00").alias("geographic_code"),
                    pl.lit(ownership_for(itype, code)).alias("ownership"),
                    pl.lit(itype).alias("industry_type"),
                    pl.lit(code).alias("industry_code"),
                    pl.col("revision").cast(pl.UInt8),
                    pl.lit(0, dtype=pl.UInt8).alias("benchmark_revision"),
                    pl.col("value").alias("employment"),
                    pl.lit(None, dtype=pl.Utf8).alias("size_class_type"),
                    pl.lit(None, dtype=pl.Utf8).alias("size_class_code"),
                    pl.lit("ces").alias("source"),
                    pl.lit(sa).alias("seasonally_adjusted"),
                )
            )

    if not out:
        return pl.DataFrame(schema=VINTAGE_STORE_SCHEMA)
    return (
        pl.concat(out, how="vertical")
        .select(list(VINTAGE_STORE_SCHEMA))
        .cast(VINTAGE_STORE_SCHEMA)
    )
