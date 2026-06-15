"""QCEW Q1 size-class cross-product (store_rebuild T4, spec §8).

Takes CES-industry-coded QCEW **Q1** rows delivered at native ``size_code``
(``'1'``-``'9'`` = the ``'large'`` scheme) and expands each industry cell into the
full cross-product ``industry_code × size_class_type``: ``total`` (code ``'0'`` =
all sizes), ``small`` (3 buckets), ``medium`` (5 buckets), and ``large`` (the 9
natives). ``small``/``medium``/``total`` are **derived** from the natives by the
``size_classes.md`` rollup — never sourced or joined from raw QCEW.

Size is a Q1 establishment-size product (assigned by March employment), so rows
exist **only** for ref-months ∈ {01,02,03}. On Q1 the all-sizes level is the
``total``/``'0'`` row **only** — emitting an extra null-size row would
double-count under the §7 ``IS NULL OR size_class_code='0'`` selector
(:func:`all_sizes_predicate`). CES and QCEW Q2/Q3/Q4 carry null ``size_class_*``.
"""

from __future__ import annotations

import polars as pl
from nfp_lookups.size_classes import SIZE_CLASS_TYPES, native_to_scheme

# Series identity excluding the size axis. Size rollups group on these.
_SERIES_KEYS = [
    "geographic_type",
    "geographic_code",
    "ownership",
    "industry_type",
    "industry_code",
    "ref_date",
    "vintage_date",
    "revision",
]

_REQUIRED_COLUMNS = (*_SERIES_KEYS, "size_code", "employment")


def all_sizes_predicate() -> pl.Expr:
    """The canonical continuous all-sizes selector (store_rebuild §7).

    Selects the headline (all-sizes) level across the whole store: the
    null-size rows (CES + QCEW Q2/Q3/Q4) **and** the ``total``/``'0'`` rows
    (QCEW Q1). A bare ``size_class_type IS NULL`` would silently drop every Q1
    month.
    """
    return pl.col("size_class_type").is_null() | (pl.col("size_class_code") == "0")


def _scheme_rows(native: pl.DataFrame, scheme: str) -> pl.DataFrame:
    """Roll native ``size_code`` rows up to one *scheme*, summing per series."""
    mapping = native_to_scheme(scheme)
    return (
        native.with_columns(
            size_class_code=pl.col("size_code").replace_strict(mapping, default=None),
        )
        .group_by([*_SERIES_KEYS, "size_class_code"])
        .agg(employment=pl.col("employment").sum())
        .with_columns(size_class_type=pl.lit(scheme, pl.Utf8))
    )


def build_size_class_panel(native: pl.DataFrame) -> pl.DataFrame:
    """Expand native-``size_code`` Q1 rows into the size cross-product.

    Parameters
    ----------
    native : pl.DataFrame
        CES-industry-coded QCEW Q1 rows at native ``size_code`` (``'1'``-``'9'``),
        carrying :data:`_REQUIRED_COLUMNS`. ``employment`` is in thousands.

    Returns
    -------
    pl.DataFrame
        ``VINTAGE_STORE_SCHEMA``-conformant rows (partition cols carried plain),
        one per ``(series, size_class_type, size_class_code)``. Every row has a
        non-null ``size_class_type``/``size_class_code``; no null-size rows.

    Raises
    ------
    ValueError
        If a required column is missing, or any ``ref_date`` is not in Q1.
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in native.columns]
    if missing:
        raise ValueError(f"native size frame missing columns: {missing}")

    non_q1 = native.filter(~pl.col("ref_date").dt.month().is_in([1, 2, 3]))
    if non_q1.height:
        sample = non_q1.head(3)["ref_date"].to_list()
        raise ValueError(
            f"size-class rows are Q1-only; got {non_q1.height} non-Q1 ref_dates "
            f"(e.g. {sample})"
        )

    rows = pl.concat([_scheme_rows(native, s) for s in SIZE_CLASS_TYPES])

    return rows.with_columns(
        benchmark_revision=pl.lit(0, pl.UInt8),
        revision=pl.col("revision").cast(pl.UInt8),
        source=pl.lit("qcew", pl.Utf8),
        seasonally_adjusted=pl.lit(False, pl.Boolean),
    ).select(
        "geographic_type",
        "geographic_code",
        "ownership",
        "industry_type",
        "industry_code",
        "ref_date",
        "vintage_date",
        "revision",
        "benchmark_revision",
        "employment",
        "size_class_type",
        "size_class_code",
        "source",
        "seasonally_adjusted",
    ).sort("industry_type", "industry_code", "ref_date", "size_class_type", "size_class_code")
