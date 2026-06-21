"""A5 scoreboard helpers: near-release as-of dates and error metrics.

``release_date_for`` prefers the store's actual rev-0 vintage_date; callers
without a store use ``first_friday_release`` (the BLS first-Friday-of-next-
month rule, holiday-shifted). Kept local to avoid importing a private
``release_dates`` symbol across packages.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl


def _first_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def first_friday_release(ref_month: date) -> date:
    """BLS first-print release date for a reference month (first Friday of the
    next month; shifted +7 days if it lands on Jan 1 or Jul 4)."""
    y, m = ref_month.year, ref_month.month
    pm, py = (1, y + 1) if m == 12 else (m + 1, y)
    friday = _first_friday(py, pm)
    if (friday.month == 1 and friday.day == 1) or (friday.month == 7 and friday.day == 4):
        friday += timedelta(days=7)
    return friday


def release_date_for(
    ref_month: date,
    *,
    store_path: Path | None = None,
) -> date:
    """Actual first-print release date from the store (rev-0 vintage_date),
    falling back to ``first_friday_release`` when the store has no such row."""
    if store_path is not None:
        from nfp_ingest.vintage_store import read_vintage_store

        lf = read_vintage_store(
            store_path,
            source="ces",
            seasonally_adjusted=True,
            geographic_type="national",
            industry_code="00",
        ).filter(
            (pl.col("revision") == 0)
            & (pl.col("benchmark_revision") == 0)
            & (pl.col("ref_date").dt.truncate("1mo") == ref_month.replace(day=1))
        )
        got = lf.select(pl.col("vintage_date").min()).collect()
        if got.height and got["vintage_date"][0] is not None:
            return got["vintage_date"][0]
    return first_friday_release(ref_month)


def near_release_asof(
    ref_month: date,
    *,
    days_before: int,
    release: date | None = None,
    store_path: Path | None = None,
) -> date:
    """As-of date = release(ref_month) - days_before."""
    rel = release if release is not None else release_date_for(ref_month, store_path=store_path)
    return rel - timedelta(days=days_before)


def score(errors: np.ndarray) -> dict:
    """ME / MAE / RMSE over an error array (actual - predicted)."""
    e = np.asarray(errors, dtype=float)
    e = e[~np.isnan(e)]
    if e.size == 0:
        return {"n": 0, "me": None, "mae": None, "rmse": None}
    return {
        "n": int(e.size),
        "me": float(e.mean()),
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt((e**2).mean())),
    }


__all__ = [
    "first_friday_release",
    "release_date_for",
    "near_release_asof",
    "score",
]
