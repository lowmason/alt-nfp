"""Store-rebuild orchestration: compose panels + guarded scratch write (store_rebuild T5).

Composes the three source panels (CES, QCEW levels, QCEW size) into one
``VINTAGE_STORE_SCHEMA`` frame and writes it to a **scratch** store,
refusing the canonical ``s3://alt-nfp/store``.

The acquire layer (QCEW API-slice fetchers + size NAICS→CES crosswalk) fetches
public BLS API slices over plain httpx (``data.bls.gov`` needs no impersonation;
only www.bls.gov is Akamai-fingerprinted) and transforms them into frames ready
for :func:`~nfp_ingest.qcew_crosswalk.build_qcew_panel`
and :func:`~nfp_ingest.size_class.build_size_class_panel`.

Usage::

    uv run alt-nfp build-rebuild [--allow-canonical]
"""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

# Acquire layer relocated to nfp_ingest.qcew_acquire (CLI production workflow spec
# §5.2/§14): nfp-ingest sits below nfp-vintages, so capture.py can now import these
# without an illegal upward import. Re-exported here (private aliases) so the existing
# test_rebuild_acquire.py / test_rebuild_gates.py imports keep resolving.
from nfp_ingest.qcew_acquire import (  # noqa: F401  (re-export for back-compat)
    _QCEW_LEVELS_REQUIRED,
    _REBUILD_START_YEAR,
    _fetch_qcew_csv,
    _prep_area_raw,
    _size_raw_to_native,
)
from nfp_ingest.qcew_acquire import (
    acquire_qcew_levels as _acquire_qcew_levels,  # noqa: F401  (back-compat alias)
)
from nfp_ingest.qcew_acquire import (
    acquire_qcew_size_native as _acquire_qcew_size_native,  # noqa: F401
)
from nfp_lookups.paths import (
    VINTAGE_STORE_PATH,
    is_canonical_store,
    is_remote,
    storage_options_for,
)
from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA

logger = logging.getLogger(__name__)

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
        under the §7 ``IS NULL OR size_class_code='0'`` selector.  The all-sizes
        ``'0'`` row's *value* is overridden to the area-levels total (the
        published un-suppressed headline) — the size bucket-sum undercounts it
        under suppression; see the §7 fix below.

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
        # §7 Q1 headline fix (store_rebuild §7/§8): the size frame's all-sizes
        # ``'0'`` row is a sum over native buckets with suppressed
        # (``disclosure_code='N'``) cells dropped, so it UNDERCOUNTS the
        # published, un-suppressed area-levels total — uneven per industry, which
        # also breaks §3 additive closure (``05 = 06 + 08``) at Q1.  The area
        # endpoint publishes the un-suppressed total at agglvl 13–16, so override
        # ONLY the ``'0'`` row's *employment* with it (metadata/vintage/revision
        # untouched, so vintage-integrity is unchanged; buckets legitimately need
        # not sum to the total under suppression).  Area totals nest by BLS
        # construction, so this restores additive closure at Q1.
        # Align on series identity + ``revision``, not the 6-col identity alone.
        # QCEW per-industry is rev-0 single-vintage today (Decision A;
        # ``_prep_area_raw`` stamps revision=0), so this is a no-op on every input
        # the system produces.  But the sibling anti-join below is deliberately
        # multi-revision-tolerant (see ``_SERIES_IDENTITY_KEY``'s note and
        # ``test_different_revision_still_deduped``); a *value*-carrying dedup on
        # the 6-col key alone would use ``unique``'s arbitrary ``keep="any"`` and
        # pick some revision's employment at random — non-deterministic, and able
        # to break the §3 additive closure this override restores (parent and
        # children landing on different revisions).  Keying on ``revision`` aligns
        # each size ``'0'`` row to the area total at its *own* revision.
        # ``vintage_date`` is intentionally NOT in the key: size and area share it
        # by construction today, but coupling on it would risk a join miss (→
        # silent fallback to the undercounting bucket-sum) if they ever diverged.
        _AREA_JOIN_KEY = [*_SERIES_IDENTITY_KEY, "revision"]
        area_lvl = (
            qcew_levels.select([*_AREA_JOIN_KEY, "employment"])
            .unique(subset=_AREA_JOIN_KEY)
            .rename({"employment": "_area_emp"})
        )
        size = (
            size.join(area_lvl, on=_AREA_JOIN_KEY, how="left")
            .with_columns(
                employment=pl.when(pl.col("size_class_code") == "0")
                # coalesce: keep the bucket-sum if the area endpoint lacked the
                # series at this revision (never null the headline).
                .then(pl.coalesce("_area_emp", "employment"))
                .otherwise(pl.col("employment"))
            )
            .drop("_area_emp")
        )

        # Derive the "has size coverage" key set from size total/'0' rows only.
        # These are exactly the rows that would double-count against a null-size
        # qcew_levels row under the §7 all-sizes predicate.  (Built AFTER the
        # value override above, which only touches employment — not the keys.)
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

    # Pin canonical column order AND every dtype. ``diagonal_relaxed`` is the
    # *relaxed* concat: when the three input frames disagree on a column's dtype
    # it coerces to a common supertype (e.g. a builder emitting ``revision`` as
    # i64, or the all-null ``size_class_*`` columns landing as Null instead of
    # Utf8). The explicit cast makes the "VINTAGE_STORE_SCHEMA-conformant"
    # contract enforced here, not dependent on the builders happening to agree.
    combined = combined.select(list(VINTAGE_STORE_SCHEMA.keys())).cast(
        dict(VINTAGE_STORE_SCHEMA)
    )

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
