"""Government-wedge model inputs (specs/government_wedge.md).

The wedge target g = 00 - 05 first-print change comes from the store; the
intervention basis comes from nfp_lookups.government, censored by as_of.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from nfp_lookups.government import get_known_interventions_as_of, intervention_column
from nfp_lookups.paths import VINTAGE_STORE_PATH

from nfp_ingest.first_print import first_print_changes

# Largest tolerated gap between the 00 and 05 first-print release stamps. The
# rebuilt store staggers a release's revisions across ~1 week; 00 and 05 rev0 are
# co-released, so their stamps must be within one release window.
_RELEASE_WINDOW_DAYS = 15


def wedge_first_print_changes(*, store_path: Path = VINTAGE_STORE_PATH) -> pl.DataFrame:
    """Per ref_date: wedge_change_k = first_print(00) - first_print(05), same release.

    Returns columns ``ref_date, chg00, chg05, wedge_change_k`` sorted by ref_date.
    Raises ``ValueError`` if the two legs are not from the same release window.
    """
    fp00 = first_print_changes(store_path=store_path, industry_type="total",
                               industry_code="00").select(
        "ref_date", pl.col("first_print_change_k").alias("chg00"),
        pl.col("vintage_date").alias("v00"))
    fp05 = first_print_changes(store_path=store_path, industry_type="total",
                               industry_code="05").select(
        "ref_date", pl.col("first_print_change_k").alias("chg05"),
        pl.col("vintage_date").alias("v05"))
    df = fp00.join(fp05, on="ref_date", how="inner").sort("ref_date").drop_nulls(
        subset=["chg00", "chg05"])
    gap = (df["v00"] - df["v05"]).dt.total_days().abs()
    if (gap > _RELEASE_WINDOW_DAYS).any():
        raise ValueError(
            "wedge legs not from same release: 00/05 vintage_date gap exceeds "
            f"{_RELEASE_WINDOW_DAYS}d — refusing a cross-vintage difference")
    return df.with_columns(
        (pl.col("chg00") - pl.col("chg05")).alias("wedge_change_k")
    ).select("ref_date", "chg00", "chg05", "wedge_change_k")


# COVID and the Oct-2025 shutdown no-print hole are masked, never deleted.
_COVID = (date(2020, 1, 1), date(2021, 12, 1))
_SHUTDOWN_HOLE = {date(2025, 10, 1)}


def _month_range(start: date, end: date) -> list[date]:
    out, y, m = [], start.year, start.month
    while date(y, m, 1) <= end:
        out.append(date(y, m, 1))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def build_wedge_model_data(
    *,
    as_of: date | None,
    target_month: date,
    store_path: Path = VINTAGE_STORE_PATH,
    start: date = date(2017, 1, 1),
) -> dict:
    """Assemble the wedge model input dict for a release-eve nowcast of target_month.

    The target month's own row is present but masked (its first print is the scored
    actual, revealed at release). Interventions are censored to ``as_of`` via the
    announcement-date guard. Returns plain numpy arrays (no Polars reaches JAX).

    The historical wedge as-of censor is bound-safe: the ref-month axis ends at
    ``target_month`` and the target row is masked, so every remaining
    ``rm < target_month`` already had its first print published by any release-eve
    ``as_of``. The load-bearing job of the ``as_of`` arg is therefore the
    intervention censor (the announcement-date guard), not the wedge-history filter.
    """
    wedge = wedge_first_print_changes(store_path=store_path)
    known = {r["ref_date"]: r["wedge_change_k"]
             for r in wedge.iter_rows(named=True)}
    # As-of censor: a wedge month is observed only if its first print is published
    # by as_of. We approximate the first-print publish date by the month's own
    # release (~5 weeks after ref month-start); the harness passes a release-eve
    # as_of, so target_month and anything not-yet-released is excluded.
    ref_months = _month_range(start, target_month)
    T = len(ref_months)
    y = np.full(T, np.nan)
    for i, rm in enumerate(ref_months):
        if rm == target_month:
            continue  # masked: scored actual, not an input
        v = known.get(rm)
        if v is not None and (as_of is None or rm < target_month):
            y[i] = v
    month_of_year = np.array([rm.month for rm in ref_months], dtype=int)
    mask = np.isfinite(y).copy()
    for i, rm in enumerate(ref_months):
        if (_COVID[0] <= rm <= _COVID[1]) or rm in _SHUTDOWN_HOLE:
            mask[i] = False
    y = np.nan_to_num(y, nan=0.0)  # masked entries contribute zero log-prob

    ivs = get_known_interventions_as_of(as_of) if as_of is not None else []
    cols = [intervention_column(iv, ref_months) for iv in ivs]
    X = np.stack(cols, axis=1) if cols else np.zeros((T, 0))
    iv_prior_mean = np.array([iv.magnitude_k for iv in ivs], dtype=float)
    iv_prior_sd = np.array([iv.magnitude_sd_k for iv in ivs], dtype=float)

    return {
        "y": y, "month_of_year": month_of_year, "T": T, "mask": mask,
        "X_intervention": X, "iv_prior_mean": iv_prior_mean, "iv_prior_sd": iv_prior_sd,
        "ref_months": ref_months, "target_idx": ref_months.index(target_month),
    }
