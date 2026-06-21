"""Vintage store reader, panel transformer, and write utilities.

Bridges the Hive-partitioned vintage parquet store to the model-ready
observation panel (PANEL_SCHEMA).

Functions
---------
read_vintage_store : Lazy scan with Hive partition predicate pushdown.
transform_to_panel : Vintage store → PANEL_SCHEMA (growth, metadata).
append_to_vintage_store : Append new observations to the Hive store.
compact_partition : Merge small files within a single partition.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import date
from pathlib import Path

import polars as pl
from nfp_lookups.paths import VINTAGE_STORE_PATH, is_remote, storage_options_for

# Single source of truth lives in nfp_lookups.schemas; re-exported here for the
# many existing callers that import it from this module (was a duplicate
# definition — audit IND-XC-3).
from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA

from nfp_ingest.size_class import all_sizes_predicate

logger = logging.getLogger(__name__)

# Boundary between pre-2017 (max-revision) and 2017+ (rank-based) QCEW regimes.
# Before this date only a single revision exists per ref_date, so we just keep
# the maximum revision.  The BLS started publishing quarterly QCEW revisions
# in a structured cadence beginning with 2017 data.
_QCEW_MONTHLY_START = date(2017, 1, 12)

# Series keys for rank-based censoring.
_CES_SERIES_KEY = [
    "seasonally_adjusted",
    "geographic_type",
    "geographic_code",
    "industry_type",
    "industry_code",
]

_QCEW_SERIES_KEY = [
    "geographic_type",
    "geographic_code",
    "industry_type",
    "industry_code",
]


# ---------------------------------------------------------------------------
# Rank-based horizon selection helpers
# ---------------------------------------------------------------------------


def _select_ces_at_horizon(ces_df: pl.DataFrame, D: date) -> pl.DataFrame:
    """Select one row per (series, ref_date) using CES revision rank rules.

    Assumes *ces_df* has already been filtered to ``ref_date < D`` and
    ``vintage_date <= D``.  Adds a dense descending rank of ``ref_date``
    within each series and applies the triangular diagonal selection:

    * Rank 1 → ``(revision=0, benchmark_revision=0)``
    * Rank 2 → ``(revision=1, benchmark_revision=0)``
    * Rank 3+ → ``(revision=2, benchmark_revision=0)`` (actual 3rd print)

    The rank rules above never select a benchmark-revised row; benchmark-
    quality information is designed to enter the model through QCEW
    observations instead.  The fallback is the one in-principle exception:
    for any ``(series, ref_date)`` not matched by the prescribed rules it
    keeps ``max(revision), max(benchmark_revision)``, which *can* be a
    ``benchmark_revision > 0`` row when that key has no ``bmr=0`` sibling
    (e.g. the store's oldest months, backfilled from benchmark-revised data
    only).

    In the real pipeline this does not leak benchmark CES into the panel:
    :func:`transform_to_panel` computes per-cohort log-growth and drops
    null/non-finite growth *before* this selection (removing the first
    ``ref_date`` of each benchmark cohort), and where a ``bmr=0`` triangle
    exists it out-ranks the ``bmr>0`` row.  Empirically, zero benchmark-
    revised CES rows reach ``build_panel`` output across all monthly
    horizons 2022-2026 (audit finding I-2).  The fallback is kept verbatim
    with the frozen reference, so the behaviour is intentionally unchanged.
    """
    if ces_df.is_empty():
        return ces_df

    sk = _CES_SERIES_KEY

    df = ces_df.with_columns(
        pl.col("ref_date")
        .rank(descending=True, method="dense")
        .over(sk)
        .alias("_rank")
    )

    _rev = pl.col("revision")
    _bmr = pl.col("benchmark_revision")
    _rk = pl.col("_rank")

    r1 = df.filter((_rk == 1) & (_rev == 0) & (_bmr == 0))
    r2 = df.filter((_rk == 2) & (_rev == 1) & (_bmr == 0))
    r3plus = df.filter((_rk >= 3) & (_rev == 2) & (_bmr == 0))

    selected = pl.concat([r1, r2, r3plus])

    # Fallback: any (series, ref_date) not yet selected
    all_keys = df.select(sk + ["ref_date"]).unique()
    sel_keys = selected.select(sk + ["ref_date"]).unique()
    missing = all_keys.join(sel_keys, on=sk + ["ref_date"], how="anti")

    if len(missing) > 0:
        remainder = df.join(missing, on=sk + ["ref_date"])
        remainder = (
            remainder.sort(["revision", "benchmark_revision"], descending=True)
            .unique(subset=sk + ["ref_date"], keep="first")
        )
        selected = pl.concat([selected, remainder])
        logger.debug(
            "CES horizon D=%s: %d (series, ref_date) used fallback selection",
            D,
            len(missing),
        )

    return selected.drop("_rank")


def _select_qcew_at_horizon(qcew_df: pl.DataFrame, D: date) -> pl.DataFrame:
    """Select one row per (series, ref_date) using QCEW revision rank rules.

    Assumes *qcew_df* has already been filtered to ``ref_date < D`` and
    ``vintage_date <= D``.

    * **Pre-2017**: keep ``max(revision)`` per (series, ref_date).
    * **2017+**: dense-rank ``ref_date`` descending; quarter-dependent
      revision rules with fallback to ``max(revision)``.
    """
    if qcew_df.is_empty():
        return qcew_df

    sk = _QCEW_SERIES_KEY

    # --- Pre-2017: max revision per (series, ref_date) --------------------
    pre = qcew_df.filter(pl.col("ref_date") < _QCEW_MONTHLY_START)
    if not pre.is_empty():
        pre = (
            pre.sort("revision", descending=True)
            .unique(subset=sk + ["ref_date"], keep="first")
        )

    # --- 2017+: rank + quarter rules -------------------------------------
    post = qcew_df.filter(pl.col("ref_date") >= _QCEW_MONTHLY_START)
    if post.is_empty():
        return pre

    post = post.with_columns(
        pl.col("ref_date")
        .rank(descending=True, method="dense")
        .over(sk)
        .alias("_rank"),
        ((pl.col("ref_date").dt.month() - 1) // 3 + 1).alias("_quarter"),
    )

    _rk = pl.col("_rank")
    _rev = pl.col("revision")
    _q = pl.col("_quarter")

    r1 = post.filter((_rk == 1) & (_rev == 0))
    r2 = post.filter((_rk == 2) & (_rev == 1))

    # Rank 3: rev=1 if Q4, else rev=2
    r3 = post.filter(
        (_rk == 3) & (
            ((_q == 4) & (_rev == 1)) | ((_q != 4) & (_rev == 2))
        )
    )

    # Rank 4: rev=1 if Q4, rev=2 if Q3, else rev=3
    r4 = post.filter(
        (_rk == 4) & (
            ((_q == 4) & (_rev == 1))
            | ((_q == 3) & (_rev == 2))
            | (~_q.is_in([3, 4]) & (_rev == 3))
        )
    )

    # Rank 5+: max(revision) per (series, ref_date)
    r5plus = post.filter(_rk >= 5)
    r5plus = (
        r5plus.sort("revision", descending=True)
        .unique(subset=sk + ["ref_date"], keep="first")
    )

    selected = pl.concat([r1, r2, r3, r4, r5plus])

    # Fallback for any unmatched (series, ref_date) in 2017+
    all_keys = post.select(sk + ["ref_date"]).unique()
    sel_keys = selected.select(sk + ["ref_date"]).unique()
    missing = all_keys.join(sel_keys, on=sk + ["ref_date"], how="anti")

    if len(missing) > 0:
        remainder = post.join(missing, on=sk + ["ref_date"])
        remainder = (
            remainder.sort("revision", descending=True)
            .unique(subset=sk + ["ref_date"], keep="first")
        )
        selected = pl.concat([selected, remainder])
        logger.debug(
            "QCEW horizon D=%s: %d (series, ref_date) used fallback selection",
            D,
            len(missing),
        )

    selected = selected.drop("_rank", "_quarter")

    if pre.is_empty():
        return selected
    return pl.concat([pre, selected])


# ---------------------------------------------------------------------------
# Validation guards for censored selection
# ---------------------------------------------------------------------------


def _validate_censored_selection(
    selected: pl.DataFrame,
    source_label: str,
    D: date,
    series_key: list[str],
) -> None:
    """Validate rank-based selection output.  Raises on fatal problems.

    Checks run in O(n) via Polars aggregations — negligible overhead.
    """
    if selected.is_empty():
        return

    n_rows = len(selected)
    n_unique = selected.select(series_key + ["ref_date"]).unique().height

    # 1+2. One row per (series, ref_date) — no duplicates, no drops
    if n_unique != n_rows:
        raise ValueError(
            f"{source_label} horizon D={D}: expected {n_unique} unique "
            f"(series, ref_date) but got {n_rows} rows "
            f"({n_rows - n_unique} duplicates)"
        )

    # 3. Consecutive months within each series
    for key_vals, grp in selected.group_by(series_key):
        months = grp["ref_date"].sort()
        if len(months) < 2:
            continue
        yms = months.dt.year() * 12 + months.dt.month()
        diffs = yms.diff().drop_nulls()
        bad = diffs.filter(diffs != 1)
        if len(bad) > 0:
            # Find first gap for the error message
            idx = (diffs != 1).arg_max()
            gap_from = months[idx]
            gap_to = months[idx + 1]
            key = key_vals if isinstance(key_vals, tuple) else (key_vals,)
            raise ValueError(
                f"{source_label} horizon D={D} series {key}: "
                f"ref_date gap {gap_from} → {gap_to}"
            )

    # 4. No null/zero employment
    bad_emp = selected.filter(
        pl.col("employment").is_null()
        | pl.col("employment").is_nan()
        | (pl.col("employment") <= 0)
    )
    if len(bad_emp) > 0:
        sample = bad_emp.head(3).select("ref_date", "employment").to_dicts()
        raise ValueError(
            f"{source_label} horizon D={D}: "
            f"{len(bad_emp)} rows with null/zero/NaN employment: {sample}"
        )

    # 5. No null growth
    bad_growth = selected.filter(
        pl.col("growth").is_null() | ~pl.col("growth").is_finite()
    )
    if len(bad_growth) > 0:
        sample = bad_growth.head(3).select("ref_date", "growth").to_dicts()
        raise ValueError(
            f"{source_label} horizon D={D}: "
            f"{len(bad_growth)} rows with null/non-finite growth: {sample}"
        )

    # 6. CES diagonal structure (warning only — frontier fallback may deviate)
    if source_label.startswith("CES"):
        for key_vals, grp in selected.group_by(series_key):
            n0 = (grp["revision"] == 0).sum()
            n1 = (grp["revision"] == 1).sum()
            if n0 != 1 or n1 != 1:
                key = key_vals if isinstance(key_vals, tuple) else (key_vals,)
                logger.warning(
                    "%s horizon D=%s series %s: revision counts "
                    "(0,1)=(%d,%d) expected (1,1); fallback was used",
                    source_label,
                    D,
                    key,
                    n0,
                    n1,
                )

    # 7. Row count sanity per series (>= 90% of input's distinct ref_dates)
    counts = selected.group_by(series_key).agg(
        pl.col("ref_date").n_unique().alias("n_selected")
    )
    low = counts.filter(pl.col("n_selected") < 10)
    if len(low) > 0 and selected["ref_date"].n_unique() > 20:
        sample = low.head(3).to_dicts()
        logger.warning(
            "%s horizon D=%s: %d series with < 10 ref_dates: %s",
            source_label,
            D,
            len(low),
            sample,
        )


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def read_vintage_store(
    store_path: Path = VINTAGE_STORE_PATH,
    *,
    source: str | None = None,
    seasonally_adjusted: bool | None = None,
    geographic_type: str | None = None,
    geographic_code: str | None = None,
    industry_type: str | None = None,
    industry_code: str | None = None,
    ref_date_range: tuple[date, date] | None = None,
) -> pl.LazyFrame:
    """Read from the Hive-partitioned vintage store with predicate pushdown.

    Partition-level filters (``source``, ``seasonally_adjusted``) are pushed
    to the scan so only relevant parquet files are opened.  All other filters
    are applied lazily.

    Parameters
    ----------
    store_path : Path
        Root of the Hive-partitioned parquet store.
    source : {'ces', 'qcew'} or None
        Filter to a specific source.
    seasonally_adjusted : bool or None
        Filter by seasonal adjustment flag.
    geographic_type : {'national', 'state', 'region', 'division'} or None
        Filter by geographic type.
    geographic_code : str or None
        Filter by geographic code (e.g. ``'00'`` for national).
    industry_type : {'total', 'domain', 'supersector', 'sector'} or None
        Filter by industry type. (Legacy pre-rebuild stores use ``'national'``
        for the ``00`` total; see ``nfp_lookups.remap_industry_type``.)
    industry_code : str or None
        Filter by industry code (e.g. ``'05'`` for total private).
    ref_date_range : tuple[date, date] or None
        Inclusive date range filter on ``ref_date``.

    Returns
    -------
    pl.LazyFrame
        Lazy scan of the vintage store with filters applied.
    """
    lf = pl.scan_parquet(
        str(store_path / "**/*.parquet"),
        hive_partitioning=True,
        schema=VINTAGE_STORE_SCHEMA,
        # Legacy stores predate the `ownership` column (store_rebuild §3); insert
        # it as null so pre-rebuild parquet reads cleanly (the T6 remap derives
        # ownership from (industry_type, industry_code) for the history join).
        missing_columns="insert",
        storage_options=storage_options_for(store_path),
    )

    if source is not None:
        lf = lf.filter(pl.col("source") == source)
    if seasonally_adjusted is not None:
        lf = lf.filter(pl.col("seasonally_adjusted") == seasonally_adjusted)
    if geographic_type is not None:
        lf = lf.filter(pl.col("geographic_type") == geographic_type)
    if geographic_code is not None:
        lf = lf.filter(pl.col("geographic_code") == geographic_code)
    if industry_type is not None:
        lf = lf.filter(pl.col("industry_type") == industry_type)
    if industry_code is not None:
        lf = lf.filter(pl.col("industry_code") == industry_code)
    if ref_date_range is not None:
        lf = lf.filter(
            pl.col("ref_date").is_between(ref_date_range[0], ref_date_range[1])
        )

    return lf


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------


def transform_to_panel(
    lf: pl.LazyFrame,
    *,
    geographic_scope: str | None = "national",
    as_of_ref: date | None = None,
) -> pl.DataFrame:
    """Transform vintage store data into a model-ready observation panel.

    Pipeline stages:

    1. Filter by geographic scope; drop nulls and zero employment.
       When *as_of_ref* is set, also filter ``vintage_date <= as_of_ref``
       and ``ref_date < as_of_ref``.
    2. Derive source tags (``ces_sa``, ``ces_nsa``, ``qcew``) and
       ``source_type``.
    3. Compute log-growth within each consistent revision group.
    3b. *(as_of_ref only)* Rank-based CES/QCEW selection — one row per
       (series, ref_date), preserving the triangular diagonal.
    4. Map ``revision_number``: CES observations with publication lag
       > 10 months become ``-1`` (benchmark); others keep the store's
       ``revision`` value.
    5. Determine ``is_final`` flag.
    6. Deduplicate across benchmark generations (keep latest vintage).
    7. Project to ``PANEL_SCHEMA`` columns.

    Parameters
    ----------
    lf : pl.LazyFrame
        Lazy frame from :func:`read_vintage_store`.
    geographic_scope : str or None
        If provided, filter to this ``geographic_type`` before transforming.
        Defaults to ``'national'``.  Pass ``None`` to keep all geographies
        (the caller must then handle the missing geography columns in
        ``PANEL_SCHEMA``).
    as_of_ref : date or None
        Horizon date for rank-based censoring.  Should use the BLS
        convention ``YYYY-MM-12``.  When set, only rows with
        ``ref_date < as_of_ref`` and ``vintage_date <= as_of_ref`` are
        considered, and CES/QCEW rows are selected via rank-based
        diagonal rules.

    Returns
    -------
    pl.DataFrame
        Observation panel conforming to ``PANEL_SCHEMA``.
    """
    if as_of_ref is not None and as_of_ref.day != 12:
        logger.warning(
            "as_of_ref=%s has day != 12; BLS convention is YYYY-MM-12",
            as_of_ref,
        )

    # --- 1. Filter --------------------------------------------------------
    if geographic_scope is not None:
        lf = lf.filter(pl.col("geographic_type") == geographic_scope)

    if as_of_ref is not None:
        lf = lf.filter(
            (pl.col("vintage_date") <= as_of_ref)
            & (pl.col("ref_date") < as_of_ref)
        )

    lf = lf.filter(
        pl.col("vintage_date").is_not_null()
        & pl.col("industry_code").is_not_null()
        & pl.col("geographic_code").is_not_null()
        & pl.col("employment").is_not_null()
        & (pl.col("employment") > 0)
    )

    # Reduce to the headline (all-sizes) level before growth + censoring. The
    # rebuilt store carries a QCEW Q1 size-class cross-product (store_rebuild §8)
    # the model does not consume; left in, the buckets corrupt the headline
    # log-growth (the growth group excludes size_class) and duplicate (series,
    # ref_date) at censoring (``_QCEW_SERIES_KEY`` excludes it too). Canonical
    # frames carry no size rows (null size_class), so this is a no-op there; the
    # guard skips frames that lack the size columns entirely.
    if {"size_class_type", "size_class_code"}.issubset(lf.collect_schema().names()):
        lf = lf.filter(all_sizes_predicate())

    # --- 2. Source tags ----------------------------------------------------
    lf = _derive_source_tags(lf)

    # --- 3. Log-growth within each revision group -------------------------
    # Growth must be computed within a consistent vintage: same source tag,
    # geography, industry, and (revision, benchmark_revision) pair.
    growth_group = [
        "source_tag",
        "geographic_type",
        "geographic_code",
        "industry_type",
        "industry_code",
        "revision",
        "benchmark_revision",
    ]

    lf = (
        lf.sort(*growth_group, "ref_date")
        .with_columns(
            pl.col("employment")
            .log()
            .diff()
            .over(growth_group)
            .alias("growth")
        )
        .filter(pl.col("growth").is_not_null() & pl.col("growth").is_finite())
    )

    # --- 3b. Rank-based horizon selection (as_of_ref only) ----------------
    if as_of_ref is not None:
        df_collected = lf.collect()
        ces_df = df_collected.filter(pl.col("source") == "ces")
        qcew_df = df_collected.filter(pl.col("source") == "qcew")

        ces_selected = _select_ces_at_horizon(ces_df, as_of_ref)
        _validate_censored_selection(
            ces_selected, "CES", as_of_ref, _CES_SERIES_KEY
        )

        qcew_selected = _select_qcew_at_horizon(qcew_df, as_of_ref)
        _validate_censored_selection(
            qcew_selected, "QCEW", as_of_ref, _QCEW_SERIES_KEY
        )

        parts = [p for p in (ces_selected, qcew_selected) if not p.is_empty()]
        if not parts:
            logger.warning("No CES or QCEW data after horizon selection D=%s", as_of_ref)
            lf = pl.DataFrame(schema=df_collected.schema).lazy()
        else:
            lf = pl.concat(parts).lazy()

    # --- 4. revision_number + publication_lag -----------------------------
    lf = lf.with_columns(
        (
            (pl.col("vintage_date").dt.year() - pl.col("ref_date").dt.year()) * 12
            + pl.col("vintage_date").dt.month()
            - pl.col("ref_date").dt.month()
        )
        .cast(pl.Int32)
        .alias("publication_lag_months"),
    )

    lf = lf.with_columns(
        pl.when(
            (pl.col("source") == "ces")
            & (pl.col("benchmark_revision") > 0)
        )
        .then(pl.lit(-1).cast(pl.Int32))
        .otherwise(pl.col("revision").cast(pl.Int32))
        .alias("revision_number"),
    )

    # --- 5. is_final ------------------------------------------------------
    # CES: final once benchmarked.
    # QCEW pre-2017: only revision-0 data exists; treat as final.
    # QCEW 2017+: final at the maximum revision for the reference quarter.
    qtr_expr = (pl.col("ref_date").dt.month() - 1) // 3 + 1

    lf = lf.with_columns(
        pl.when(pl.col("source") == "ces")
        .then(pl.col("revision_number") == -1)
        .when(
            (pl.col("source") == "qcew")
            & (pl.col("ref_date") < pl.date(2017, 1, 1))
        )
        .then(pl.lit(True))
        .when((pl.col("source") == "qcew") & (qtr_expr == 1))
        .then(pl.col("revision").cast(pl.Int32) == 4)
        .when((pl.col("source") == "qcew") & (qtr_expr == 2))
        .then(pl.col("revision").cast(pl.Int32) == 3)
        .when((pl.col("source") == "qcew") & (qtr_expr == 3))
        .then(pl.col("revision").cast(pl.Int32) == 2)
        .when((pl.col("source") == "qcew") & (qtr_expr == 4))
        .then(pl.col("revision").cast(pl.Int32) == 1)
        .otherwise(pl.lit(False))
        .alias("is_final"),
    )

    # --- 6. Deduplicate across benchmark generations ----------------------
    # Multiple benchmark_revision values can map to the same revision_number
    # (e.g. CES rev=2 pre-benchmark and post-benchmark both map to 2 or -1).
    # Keep the row with the latest vintage_date.
    dedup_key = [
        "ref_date",
        "source_tag",
        "industry_type",
        "industry_code",
        "geographic_type",
        "geographic_code",
        "revision_number",
    ]
    lf = lf.sort("vintage_date", descending=True).unique(
        subset=dedup_key, keep="first"
    )

    # --- 7. Project to PANEL_SCHEMA ---------------------------------------
    result = (
        lf.select(
            pl.col("ref_date").alias("period"),
            pl.col("geographic_type"),
            pl.col("geographic_code"),
            pl.col("industry_code"),
            pl.col("industry_type").alias("industry_level"),
            pl.col("source_tag").alias("source"),
            pl.col("source_type"),
            pl.col("growth"),
            pl.col("employment").alias("employment_level"),
            pl.col("seasonally_adjusted").alias("is_seasonally_adjusted"),
            pl.col("vintage_date"),
            pl.col("revision_number"),
            pl.col("is_final"),
            pl.col("publication_lag_months"),
            pl.lit(None).cast(pl.Float64).alias("coverage_ratio"),
        )
        .sort("period", "source", "industry_code", "revision_number")
        .collect()
    )

    logger.info("Panel built from vintage store: %d rows", len(result))
    return result


def _derive_source_tags(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Add ``source_tag`` and ``source_type`` columns."""
    return lf.with_columns(
        pl.when((pl.col("source") == "ces") & pl.col("seasonally_adjusted"))
        .then(pl.lit("ces_sa"))
        .when((pl.col("source") == "ces") & ~pl.col("seasonally_adjusted"))
        .then(pl.lit("ces_nsa"))
        .when(pl.col("source") == "qcew")
        .then(pl.lit("qcew"))
        # .when((pl.col("source") == "sae") & pl.col("seasonally_adjusted"))
        # .then(pl.lit("sae_sa"))
        # .when((pl.col("source") == "sae") & ~pl.col("seasonally_adjusted"))
        # .then(pl.lit("sae_nsa"))
        .otherwise(pl.col("source"))
        .alias("source_tag"),
        #
        pl.when((pl.col("source") == "ces") & pl.col("seasonally_adjusted"))
        .then(pl.lit("official_sa"))
        .when((pl.col("source") == "ces") & ~pl.col("seasonally_adjusted"))
        .then(pl.lit("official_nsa"))
        .when(pl.col("source") == "qcew")
        .then(pl.lit("census"))
        # .when((pl.col("source") == "sae") & pl.col("seasonally_adjusted"))
        # .then(pl.lit("official_sa"))
        # .when((pl.col("source") == "sae") & ~pl.col("seasonally_adjusted"))
        # .then(pl.lit("official_nsa"))
        .otherwise(pl.lit("other"))
        .alias("source_type"),
    )


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def _content_suffix(df: pl.DataFrame) -> str:
    """Short, order-insensitive content hash for a partition write batch."""
    h = df.hash_rows().sort()  # Series[UInt64]; sort → order-insensitive
    return hashlib.sha1(h.to_numpy().tobytes()).hexdigest()[:12]


def append_to_vintage_store(
    new_rows: pl.DataFrame,
    store_path: Path = VINTAGE_STORE_PATH,
) -> int:
    """Append new vintage observations to the Hive-partitioned store.

    Rows are partitioned by ``(source, seasonally_adjusted)`` and written as
    new parquet files.  Rows whose uniqueness key already exists in the
    partition are silently skipped.

    Parameters
    ----------
    new_rows : pl.DataFrame
        Must include all columns in :data:`VINTAGE_STORE_SCHEMA`.
    store_path : Path
        Root of the Hive-partitioned parquet store.

    Returns
    -------
    int
        Number of new rows actually appended.
    """
    for col in VINTAGE_STORE_SCHEMA:
        if col not in new_rows.columns:
            raise ValueError(f"Missing required column: {col}")

    # vintage_date is excluded from the key, so on collision the FIRST-WRITTEN row
    # per ukey is kept: a later-arriving row with a matching ukey is anti-joined out
    # below, regardless of its vintage_date. (compact_partition instead keeps
    # min(vintage_date) deterministically; the two rules coincide only when appends
    # arrived in chronological — ascending vintage_date — order.)
    ukey = [
        "ref_date",
        "industry_type",
        "industry_code",
        "geographic_type",
        "geographic_code",
        "revision",
        "benchmark_revision",
        "ownership",
        "size_class_type",
        "size_class_code",
    ]

    total_appended = 0

    for (source, sa), partition_df in new_rows.group_by(
        ["source", "seasonally_adjusted"], maintain_order=True
    ):
        sa_str = str(sa).lower()
        partition_dir = store_path / f"source={source}" / f"seasonally_adjusted={sa_str}"

        if partition_dir.exists():
            existing = pl.read_parquet(
                str(partition_dir / "*.parquet"),
                storage_options=storage_options_for(store_path),
            )
            partition_df = partition_df.join(
                existing.select(ukey).unique(),
                on=ukey,
                how="anti",
                nulls_equal=True,
            )

        if len(partition_df) == 0:
            continue

        if not is_remote(store_path):  # object storage has no directories
            partition_dir.mkdir(parents=True, exist_ok=True)

        vmin = partition_df["vintage_date"].min()
        vmax = partition_df["vintage_date"].max()
        fname = f"v_{vmin}_{vmax}_{_content_suffix(partition_df)}.parquet"

        partition_df.drop(["source", "seasonally_adjusted"]).write_parquet(
            str(partition_dir / fname),
            storage_options=storage_options_for(store_path),
        )
        total_appended += len(partition_df)
        logger.info("Appended %d rows to %s", len(partition_df), partition_dir / fname)

    return total_appended


def compact_partition(
    store_path: Path,
    source: str,
    seasonally_adjusted: bool,
) -> None:
    """Merge all parquet files within a single partition into one file.

    Reads every file in the partition, deduplicates on the store uniqueness
    key — keeping the row with the minimum ``vintage_date`` (deterministic by
    value), unlike append which keeps the first-written row — and writes a
    single consolidated file.  Original fragment files are then removed.

    Parameters
    ----------
    store_path : Path
        Root of the Hive-partitioned parquet store.
    source : str
        Source partition key (``'ces'``, ``'qcew'``).
    seasonally_adjusted : bool
        Seasonal adjustment partition key.
    """
    sa_str = str(seasonally_adjusted).lower()
    partition_dir = store_path / f"source={source}" / f"seasonally_adjusted={sa_str}"

    if not partition_dir.exists():
        logger.warning("Partition directory does not exist: %s", partition_dir)
        return

    files = sorted(partition_dir.glob("*.parquet"))
    if len(files) <= 1:
        logger.info(
            "Partition already compact (%d file): %s", len(files), partition_dir
        )
        return

    # vintage_date is excluded from the key; compact deterministically keeps the row
    # with the minimum vintage_date per ukey (the sort+unique below). This differs
    # from append_to_vintage_store, which keeps the FIRST-WRITTEN row per ukey — the
    # two rules coincide only when rows were appended in chronological order.
    ukey = [
        "ref_date",
        "industry_type",
        "industry_code",
        "geographic_type",
        "geographic_code",
        "revision",
        "benchmark_revision",
        "ownership",
        "size_class_type",
        "size_class_code",
    ]

    combined = (
        pl.read_parquet(
            str(partition_dir / "*.parquet"),
            storage_options=storage_options_for(store_path),
        )
        .sort("vintage_date", descending=False)  # min(vintage_date) per ukey wins
        .unique(subset=ukey, keep="first")
        .sort("ref_date", "industry_code", "revision")
    )

    compacted_path = partition_dir / "compacted.parquet"
    combined.write_parquet(
        str(compacted_path),
        storage_options=storage_options_for(store_path),
    )

    for f in files:
        if f != compacted_path:
            f.unlink()

    logger.info(
        "Compacted %d files → %d rows in %s",
        len(files),
        len(combined),
        compacted_path,
    )
