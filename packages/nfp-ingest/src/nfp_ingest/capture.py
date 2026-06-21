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

import polars as pl
from nfp_lookups.industry import ownership_for
from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA

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
