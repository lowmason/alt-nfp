"""CES (and QCEW) month-T capture-to-store adapter (spec §5.1).

Bridges the legacy ``COMBINED_SCHEMA`` CES release frame emitted by
:func:`nfp_ingest.releases._fetch_ces_releases` to the rebuilt
``VINTAGE_STORE_SCHEMA`` and appends it incrementally to the vintage store.
Production captures the current print BLS publishes for a month ``T`` and
appends it; the triangular bulk extract is never re-run here (that is the
bootstrap path).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl
from nfp_lookups.industry import ownership_for
from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA

from nfp_ingest.vintage_store import read_vintage_store

logger = logging.getLogger(__name__)


@dataclass
class CorrectedLevel:
    """A capture row whose ukey already exists in the store with a different level.

    Surfaced by :func:`_detect_corrected_levels` (spec §5.1.4 / §6.3): the store
    ukey excludes both ``vintage_date`` and ``employment``, so a re-stamped
    same-revision level would be silently dropped by the append anti-join. This
    record is the runtime detection signal — no auto-replacement is performed.
    """

    ref_date: date
    industry_code: str
    revision: int
    benchmark_revision: int
    stored_employment: float
    incoming_employment: float


@dataclass
class CaptureResult:
    """Outcome of a single ``capture_*`` call.

    Attributes
    ----------
    appended : int
        Rows actually written to the store (post anti-join).
    corrected : list[CorrectedLevel]
        Existing-ukey rows whose incoming level differs from the stored level.
    skipped : int
        Rows present in the capture but already in the store (anti-joined out).
    """

    appended: int
    corrected: list[CorrectedLevel]
    skipped: int


def _remap_ces_to_store_schema(df: pl.DataFrame) -> pl.DataFrame:
    """Remap a ``COMBINED_SCHEMA`` CES frame to ``VINTAGE_STORE_SCHEMA``.

    Derives the rebuilt ``(industry_type, ownership)`` axes (spec §5.1.3):
    ``'00'``→(``total``, ``total``), ``'05'``→(``total``, ``private``), every
    other supersector code →(``supersector``, ``private``). ``size_class_*`` are
    null (CES has no size dimension). ``ownership`` is resolved through
    :func:`nfp_lookups.industry.ownership_for` on the rebuilt pair — never the
    legacy ``'national'/'domain'`` mapping in ``releases.py``.

    Parameters
    ----------
    df : pl.DataFrame
        A frame in ``nfp_ingest.releases.COMBINED_SCHEMA`` (CES release shape).

    Returns
    -------
    pl.DataFrame
        A frame in ``VINTAGE_STORE_SCHEMA`` column order and dtypes.
    """
    rebuilt_type = (
        pl.when(pl.col("industry_code").is_in(["00", "05"]))
        .then(pl.lit("total"))
        .otherwise(pl.lit("supersector"))
    )
    df = df.with_columns(rebuilt_type.alias("industry_type"))

    # ownership_for is keyed on the rebuilt (industry_type, industry_code) pair.
    pairs = (
        df.select("industry_type", "industry_code")
        .unique()
        .to_dicts()
    )
    own_map = {
        (p["industry_type"], p["industry_code"]): ownership_for(
            p["industry_type"], p["industry_code"]
        )
        for p in pairs
    }
    ownership = pl.struct("industry_type", "industry_code").map_elements(
        lambda s: own_map[(s["industry_type"], s["industry_code"])],
        return_dtype=pl.Utf8,
    )

    return (
        df.with_columns(
            ownership.alias("ownership"),
            pl.lit(None, dtype=pl.Utf8).alias("size_class_type"),
            pl.lit(None, dtype=pl.Utf8).alias("size_class_code"),
        )
        .select(list(VINTAGE_STORE_SCHEMA))
        .cast(VINTAGE_STORE_SCHEMA)
    )


# Extended store ukey: the 7-col append/compact key (vintage_store.py:709-717)
# plus the rebuilt axes added in spec §6.1. Excludes vintage_date + employment.
_CES_CORRECTED_UKEY: list[str] = [
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


def _detect_corrected_levels(
    new_rows: pl.DataFrame,
    store_path: Path,
    source: str,
    seasonally_adjusted: bool,
) -> list[CorrectedLevel]:
    """Flag incoming rows whose ukey exists in the store with a *different* level.

    Compares each incoming ``employment`` against the stored value for the same
    extended ukey (spec §5.1.4 / §6.3), *before* the append anti-join would drop
    it. Returns one :class:`CorrectedLevel` per divergence; an absent partition
    yields ``[]``.

    Parameters
    ----------
    new_rows : pl.DataFrame
        Capture rows in ``VINTAGE_STORE_SCHEMA`` for one ``(source, sa)``.
    store_path : Path
        Root of the Hive-partitioned vintage store.
    source : str
        Source partition key (``'ces'``, ``'qcew'``).
    seasonally_adjusted : bool
        Seasonal-adjustment partition key.

    Returns
    -------
    list[CorrectedLevel]
        One record per same-ukey/different-level row, sorted by ref_date.
    """
    partition_dir = (
        store_path
        / f"source={source}"
        / f"seasonally_adjusted={str(seasonally_adjusted).lower()}"
    )
    if not partition_dir.exists():
        return []

    stored = (
        read_vintage_store(
            store_path, source=source, seasonally_adjusted=seasonally_adjusted
        )
        .select([*_CES_CORRECTED_UKEY, "employment"])
        .rename({"employment": "stored_employment"})
        .collect()
    )
    if stored.is_empty():
        return []

    joined = new_rows.select([*_CES_CORRECTED_UKEY, "employment"]).join(
        stored, on=_CES_CORRECTED_UKEY, how="inner", nulls_equal=True
    )
    diverged = joined.filter(
        pl.col("employment") != pl.col("stored_employment")
    ).sort("ref_date")

    return [
        CorrectedLevel(
            ref_date=r["ref_date"],
            industry_code=r["industry_code"],
            revision=r["revision"],
            benchmark_revision=r["benchmark_revision"],
            stored_employment=r["stored_employment"],
            incoming_employment=r["employment"],
        )
        for r in diverged.iter_rows(named=True)
    ]
