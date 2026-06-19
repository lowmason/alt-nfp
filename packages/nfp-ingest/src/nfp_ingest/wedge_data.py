"""Government-wedge model inputs (specs/government_wedge.md).

The wedge target g = 00 - 05 first-print change comes from the store; the
intervention basis comes from nfp_lookups.government, censored by as_of.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
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
