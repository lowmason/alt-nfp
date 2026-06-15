"""Reconstruct CES private series from QCEW bulk data (store_rebuild T3).

Maps QCEW national, private (``own_code=='5'``) employment to the CES private
industry hierarchy (``specs/ces_qcew_industry.md``), explodes the quarterly
``month{1,2,3}_emplvl`` columns to monthly rows in CES-comparable thousands, and
emits ``VINTAGE_STORE_SCHEMA``-shaped rows (``source='qcew'``, ``ownership=
'private'``, ``benchmark_revision=0``).

Aggregation is **per QCEW vintage**: every CES-code sum stays within one
``(revision, vintage_date)`` (spec §4.2/§5) — mixing vintages would corrupt the
level. ``vintage_date`` comes from
:func:`nfp_lookups.revision_schedules.get_qcew_vintage_date`.

The build order (spec §7) is bottom-up so the private tree is additively closed:
leaf sectors → supersectors (agglvl-13 direct pull, except ``10`` which sums its
sectors) → domains ``06``/``08`` → total ``05``.
"""

from __future__ import annotations

import polars as pl
from nfp_lookups.industry import (
    QCEW_AREA_NATIONAL,
    QCEW_DOMAIN,
    QCEW_OWN_PRIVATE,
    QCEW_SECTOR_PULLS,
    QCEW_SUPERSECTOR,
)
from nfp_lookups.revision_schedules import get_qcew_vintage_date

# Columns that identify one QCEW vintage of one reference month. Every crosswalk
# sum groups on these so it never crosses a (revision, vintage_date).
_VINTAGE_GROUP = ["ref_date", "year", "qtr", "revision"]

# Raw QCEW bulk columns the builder consumes.
_REQUIRED_COLUMNS = (
    "area_fips",
    "own_code",
    "industry_code",
    "agglvl_code",
    "year",
    "qtr",
    "month1_emplvl",
    "month2_emplvl",
    "month3_emplvl",
    "revision",
)


def _to_monthly(raw: pl.DataFrame) -> pl.DataFrame:
    """Filter to national private, explode month1/2/3 → monthly, ÷1000.

    Returns one row per ``(agglvl_code, industry_code, ref_date, year, qtr,
    revision)`` with ``employment`` in thousands of persons.
    """
    df = raw.with_columns(
        pl.col("own_code").cast(pl.Utf8),
        pl.col("area_fips").cast(pl.Utf8),
        pl.col("agglvl_code").cast(pl.Utf8),
        pl.col("industry_code").cast(pl.Utf8),
        pl.col("year").cast(pl.Int32),
        pl.col("qtr").cast(pl.Int32),
        pl.col("revision").cast(pl.Int32),
    ).filter(
        (pl.col("own_code") == QCEW_OWN_PRIVATE)
        & (pl.col("area_fips") == QCEW_AREA_NATIONAL)
    )

    return (
        df.unpivot(
            ["month1_emplvl", "month2_emplvl", "month3_emplvl"],
            index=["agglvl_code", "industry_code", "year", "qtr", "revision"],
            variable_name="month_col",
            value_name="emp",
        )
        .with_columns(
            month_offset=pl.col("month_col").str.extract(r"month(\d)").cast(pl.Int32)
            - 1,
        )
        .with_columns(
            month=((pl.col("qtr") - 1) * 3 + 1 + pl.col("month_offset")).cast(pl.Int32),
        )
        .with_columns(
            ref_date=pl.date(pl.col("year"), pl.col("month"), 12),
            # Sum the measure (never a rate); persons → CES-comparable thousands.
            employment=pl.col("emp").cast(pl.Float64) / 1000.0,
        )
        .select(
            "agglvl_code", "industry_code", "year", "qtr", "revision",
            "ref_date", "employment",
        )
    )


def _pull(monthly: pl.DataFrame, industry_codes: tuple[str, ...], agglvl: str) -> pl.DataFrame:
    """Sum ``employment`` over the named QCEW cells, per vintage."""
    return (
        monthly.filter(
            (pl.col("agglvl_code") == agglvl)
            & pl.col("industry_code").is_in(list(industry_codes))
        )
        .group_by(_VINTAGE_GROUP)
        .agg(employment=pl.col("employment").sum())
    )


def _sum_children(built: pl.DataFrame, child_ids: tuple[str, ...]) -> pl.DataFrame:
    """Roll up already-built CES rows to a parent, per vintage."""
    return (
        built.filter(pl.col("industry_code").is_in(list(child_ids)))
        .group_by(_VINTAGE_GROUP)
        .agg(employment=pl.col("employment").sum())
    )


def _tag(df: pl.DataFrame, industry_type: str, industry_code: str) -> pl.DataFrame:
    return df.with_columns(
        industry_type=pl.lit(industry_type, pl.Utf8),
        industry_code=pl.lit(industry_code, pl.Utf8),
    )


def _vintage_dates(combos: pl.DataFrame) -> pl.DataFrame:
    """Map distinct ``(year, qtr, revision)`` to a QCEW ``vintage_date``."""
    return combos.with_columns(
        vintage_date=pl.struct("year", "qtr", "revision").map_elements(
            lambda s: get_qcew_vintage_date(
                f"Q{int(s['qtr'])}", int(s["year"]), int(s["revision"])
            ),
            return_dtype=pl.Date,
        )
    )


def build_qcew_panel(raw: pl.DataFrame) -> pl.DataFrame:
    """Build CES private vintage-store rows from a raw QCEW bulk frame.

    Parameters
    ----------
    raw : pl.DataFrame
        QCEW ``singlefile`` rows carrying at least :data:`_REQUIRED_COLUMNS`.
        ``revision`` tags the vintage of each ``(year, qtr)`` cell (assigned by
        the acquisition/orchestration layer).

    Returns
    -------
    pl.DataFrame
        ``VINTAGE_STORE_SCHEMA``-conformant rows (minus the Hive partition cols,
        which are carried as plain ``source``/``seasonally_adjusted`` columns):
        20 sectors, 10 supersectors, domains ``06``/``08``, and total ``05``,
        all ``ownership='private'``, monthly, in thousands.
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"raw QCEW frame missing columns: {missing}")

    monthly = _to_monthly(raw)

    # 1. Leaf sectors (agglvl 14/15/16).
    sectors = pl.concat(
        [
            _tag(_pull(monthly, codes, agglvl), "sector", ces_id)
            for ces_id, (codes, agglvl) in QCEW_SECTOR_PULLS.items()
        ]
    )

    # 2. Supersectors: agglvl-13 direct pull, except '10' (sum of its sectors).
    ss_parts: list[pl.DataFrame] = []
    for ss, spec in QCEW_SUPERSECTOR.items():
        qcew_code = spec["qcew_code"]
        if qcew_code is None:
            built = _sum_children(sectors, tuple(spec["sectors"]))
        else:
            built = _pull(monthly, (str(qcew_code),), "13")
        ss_parts.append(_tag(built, "supersector", ss))
    supersectors = pl.concat(ss_parts)

    # 3. Domains 06/08 from supersectors; 05 (total private) from 06 + 08.
    dom06 = _tag(_sum_children(supersectors, QCEW_DOMAIN["06"]), "domain", "06")
    dom08 = _tag(_sum_children(supersectors, QCEW_DOMAIN["08"]), "domain", "08")
    domains = pl.concat([dom06, dom08])
    total05 = _tag(_sum_children(domains, QCEW_DOMAIN["05"]), "total", "05")

    allrows = pl.concat([sectors, supersectors, domains, total05])

    vdates = _vintage_dates(allrows.select("year", "qtr", "revision").unique())
    result = allrows.join(vdates, on=["year", "qtr", "revision"], how="left")

    return result.select(
        geographic_type=pl.lit("national", pl.Utf8),
        geographic_code=pl.lit("00", pl.Utf8),
        ownership=pl.lit("private", pl.Utf8),
        industry_type=pl.col("industry_type"),
        industry_code=pl.col("industry_code"),
        ref_date=pl.col("ref_date"),
        vintage_date=pl.col("vintage_date"),
        revision=pl.col("revision").cast(pl.UInt8),
        benchmark_revision=pl.lit(0, pl.UInt8),
        employment=pl.col("employment"),
        source=pl.lit("qcew", pl.Utf8),
        seasonally_adjusted=pl.lit(False, pl.Boolean),
    ).sort("industry_type", "industry_code", "ref_date", "revision")
