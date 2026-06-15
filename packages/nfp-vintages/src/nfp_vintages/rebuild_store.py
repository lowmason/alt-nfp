"""Store-rebuild orchestration: compose panels + guarded scratch write (store_rebuild T5).

Composes the three source panels (CES, QCEW levels, QCEW size) into one
``VINTAGE_STORE_SCHEMA`` frame and writes it to a **scratch** store,
refusing the canonical ``s3://alt-nfp/store``.

The acquire layer (QCEW API-slice fetchers + size NAICS→CES crosswalk) is
**deferred** (see ``specs/store_rebuild_acquire.md``).  The two seam functions
:func:`_acquire_qcew_levels` and :func:`_acquire_qcew_size_native` raise
``NotImplementedError`` until a maintainer implements them.

Usage (once acquire is wired)::

    uv run alt-nfp build-rebuild [--allow-canonical]
"""

from __future__ import annotations

from typing import Any

import polars as pl
from nfp_lookups.paths import (
    VINTAGE_STORE_PATH,
    is_canonical_store,
    is_remote,
    storage_options_for,
)
from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA

# The 6-column series identity that uniquely identifies one industry-month
# *independent* of the size axis, vintage, or revision.  Used as the anti-join
# key to detect which qcew_levels rows have size coverage.
#
# NOTE: do NOT include ``vintage_date`` or ``revision`` — a qcew_levels row at
# revision=1 and a size row at revision=0 cover the same series-month and the
# level row must still be dropped.
_SERIES_IDENTITY_KEY = [
    "geographic_type",
    "geographic_code",
    "ownership",
    "industry_type",
    "industry_code",
    "ref_date",
]


# ---------------------------------------------------------------------------
# Deferred seams (acquire layer — NOT IMPLEMENTED)
# ---------------------------------------------------------------------------


def _acquire_qcew_levels() -> pl.DataFrame:
    """Acquire raw QCEW area-endpoint rows for crosswalk → :func:`build_qcew_panel`.

    **NOT IMPLEMENTED** — deferred to the acquire layer.

    Fetches per-quarter slices from the QCEW API (``/api/{y}/{q}/area/US000.csv``)
    for each (year, quarter, revision) combination and attaches ``revision`` tags.
    See ``specs/store_rebuild_acquire.md`` for the API-slice URLs and revision
    convention.

    Raises
    ------
    NotImplementedError
        Always.  Implement this function to enable ``build-rebuild`` QCEW levels.
    """
    raise NotImplementedError(
        "_acquire_qcew_levels is not yet implemented. "
        "See specs/store_rebuild_acquire.md for the QCEW API-slice URLs "
        "and agglvl-13/14/15/16 revision convention."
    )


def _acquire_qcew_size_native() -> pl.DataFrame:
    """Acquire raw QCEW Q1 size-endpoint rows for :func:`build_size_class_panel`.

    **NOT IMPLEMENTED** — deferred to the acquire layer.

    Fetches Q1-only per-size slices from the QCEW API
    (``/api/{y}/1/size/{1-9}.csv``) and crosswalks agglvl-21–28 NAICS codes to
    CES industry codes.  Note: the lookups pull-tables (``QCEW_SECTOR_PULLS``,
    ``_QCEW_AGGLVL``) only cover agglvl 13/14/15/16 (the area endpoint); the
    size endpoint delivers 21–28 and requires NEW pull mappings validated
    against real size rows.  See ``specs/store_rebuild_acquire.md``.

    Raises
    ------
    NotImplementedError
        Always.  Implement this function to enable ``build-rebuild`` size classes.
    """
    raise NotImplementedError(
        "_acquire_qcew_size_native is not yet implemented. "
        "See specs/store_rebuild_acquire.md for the size API-slice URLs "
        "and the agglvl-21–28 NAICS→CES crosswalk gap."
    )


# ---------------------------------------------------------------------------
# Core compose function
# ---------------------------------------------------------------------------


def compose_rebuild_panel(
    ces: pl.DataFrame,
    qcew_levels: pl.DataFrame,
    size: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Compose CES + QCEW-levels + optional QCEW-size into one store-schema frame.

    Parameters
    ----------
    ces : pl.DataFrame
        ``VINTAGE_STORE_SCHEMA``-conformant rows from
        :func:`nfp_ingest.ces_builder.build_ces_panel`.
        Has ``size_class_type``/``size_class_code`` (both null).
    qcew_levels : pl.DataFrame
        ``VINTAGE_STORE_SCHEMA``-conformant rows from
        :func:`nfp_ingest.qcew_crosswalk.build_qcew_panel`.
        **Omits** ``size_class_type``/``size_class_code`` entirely (the builder's
        ``.select(...)`` ends before those cols); ``diagonal_relaxed`` null-fills them.
    size : pl.DataFrame or None
        ``VINTAGE_STORE_SCHEMA``-conformant rows from
        :func:`nfp_ingest.size_class.build_size_class_panel` with non-null
        ``size_class_type``/``size_class_code``.  When provided, QCEW level rows
        whose series-month has size coverage are replaced by the size frame's
        ``total``/``'0'`` (all-sizes) + bucket rows — preventing double-counting
        under the §7 ``IS NULL OR size_class_code='0'`` selector.

    Returns
    -------
    pl.DataFrame
        ``VINTAGE_STORE_SCHEMA``-conformant frame (all 14 cols including
        ``source``/``seasonally_adjusted``) sorted deterministically.

    Notes
    -----
    §7 anti-join: when *size* is provided, a ``qcew_levels`` row is dropped
    **only** when the size frame contains a ``total``/``'0'`` row for the
    same ``(geographic_type, geographic_code, ownership, industry_type,
    industry_code, ref_date)`` — i.e. row existence on a 6-col key, not a
    month/quarter filter.  Industry-months without size coverage keep their
    null-size level row (partial coverage).
    """
    if size is not None:
        # Derive the "has size coverage" key set from size total/'0' rows only.
        # These are exactly the rows that would double-count against a null-size
        # qcew_levels row under the §7 all-sizes predicate.
        coverage_keys = (
            size.filter(pl.col("size_class_code") == "0")
            .select(_SERIES_IDENTITY_KEY)
            .unique()
        )
        # Anti-join: drop qcew_levels rows that have size coverage.
        # Done BEFORE concat so the coverage keys never collide with the CES
        # or size frame rows during the join.
        qcew_to_union = qcew_levels.join(coverage_keys, on=_SERIES_IDENTITY_KEY, how="anti")
    else:
        qcew_to_union = qcew_levels

    # Build the parts list for diagonal_relaxed concat.
    # diagonal_relaxed null-fills qcew_to_union's missing size_class_* columns.
    parts: list[pl.DataFrame] = [ces, qcew_to_union]
    if size is not None:
        parts.append(size)

    combined = pl.concat(parts, how="diagonal_relaxed")

    # Ensure size columns have the schema dtype (Utf8, nullable).
    combined = combined.with_columns(
        pl.col("size_class_type").cast(pl.Utf8),
        pl.col("size_class_code").cast(pl.Utf8),
    )

    # Select into canonical VINTAGE_STORE_SCHEMA column order.
    combined = combined.select(list(VINTAGE_STORE_SCHEMA.keys()))

    return combined.sort(
        "source",
        "industry_type",
        "industry_code",
        "ref_date",
        "size_class_type",
        "size_class_code",
        "vintage_date",
        "revision",
    )


# ---------------------------------------------------------------------------
# Guarded Hive-partition write
# ---------------------------------------------------------------------------


def write_rebuild_store(
    panel: pl.DataFrame,
    store_path: Any = None,
    *,
    allow_canonical: bool = False,
) -> None:
    """Write *panel* as a Hive-partitioned parquet store, targeting a scratch prefix.

    Mirrors the write half of :func:`nfp_vintages.build_store.build_store`.
    The canonical guard is the **first** statement — no I/O happens before it fires.

    Parameters
    ----------
    panel : pl.DataFrame
        ``VINTAGE_STORE_SCHEMA``-conformant frame including ``source`` and
        ``seasonally_adjusted`` partition columns.
    store_path : Path-like or None
        Output store root.  Defaults to :data:`nfp_lookups.paths.VINTAGE_STORE_PATH`.
        The intended target for rebuilds is the scratch prefix
        ``s3://alt-nfp/store-rebuild`` (set via ``NFP_STORE_URI``).
    allow_canonical : bool
        Permit writing to the canonical store.  Defaults to ``False``.
        Passing ``True`` is dangerous — see root ``CLAUDE.md``.

    Raises
    ------
    RuntimeError
        If *store_path* is the canonical store (``s3://alt-nfp/store`` or
        equivalent) and *allow_canonical* is ``False``.
    """
    out_path = store_path if store_path is not None else VINTAGE_STORE_PATH

    # Guard is first — no I/O before this check.
    if is_canonical_store(out_path) and not allow_canonical:
        raise RuntimeError(
            "refusing to write the canonical store in place "
            f"({out_path}); target a scratch prefix (e.g. s3://alt-nfp/store-rebuild) "
            "or pass allow_canonical=True. "
            "See CLAUDE.md 'Never rebuild the canonical store in place'."
        )

    if not is_remote(out_path):
        out_path.mkdir(parents=True, exist_ok=True)

    for (source, sa), partition_df in panel.group_by(
        ["source", "seasonally_adjusted"], maintain_order=True,
    ):
        sa_str = str(sa).lower()
        partition_dir = out_path / f"source={source}" / f"seasonally_adjusted={sa_str}"

        if not is_remote(out_path):
            partition_dir.mkdir(parents=True, exist_ok=True)

        # Remove existing parquet files in this partition before writing.
        if partition_dir.exists():
            for f in partition_dir.glob("*.parquet"):
                f.unlink()

        write_df = partition_df.drop(["source", "seasonally_adjusted"])
        vmin = write_df["vintage_date"].min()
        vmax = write_df["vintage_date"].max()
        # Polars aggregates skip nulls; an all-null vintage_date would silently
        # name the file ``v_None_None.parquet``. Fail loud — the rebuild path
        # pulls from API slices where a missing vintage_date signals bad data.
        if vmin is None or vmax is None:
            raise ValueError(
                f"partition (source={source}, seasonally_adjusted={sa_str}) has "
                "null vintage_date values; cannot name the output file"
            )
        fname = f"v_{vmin}_{vmax}.parquet"

        write_df.write_parquet(
            str(partition_dir / fname),
            storage_options=storage_options_for(out_path),
        )
        print(f"  {partition_dir.name}: {write_df.height:,} rows → {fname}")

    print(f"Wrote rebuild store to {out_path}")
