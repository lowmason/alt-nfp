"""Read-only store health + knowability report (spec §8).

Built on ``read_vintage_store`` (partition-prune + projection pushdown,
LazyFrame) ONLY — never ``transform_to_panel`` (the expensive growth/censoring
path) and never ``views.py`` (panel-grain, post-transform). Coverage is raw
row presence (no ``employment > 0`` filter) so the Oct-2025 ``-1`` "no print"
sentinel (``first_print.py:79-84``) counts as present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import polars as pl
from nfp_ingest.vintage_store import read_vintage_store
from nfp_lookups.paths import VINTAGE_STORE_PATH, is_canonical_store, is_remote

# (source, seasonally_adjusted) partitions present in the rebuilt store.
_PARTITIONS: tuple[tuple[str, bool], ...] = (
    ("ces", True),
    ("ces", False),
    ("qcew", False),
)


@dataclass(frozen=True)
class PartitionCoverage:
    """Coverage of one ``(source, seasonally_adjusted)`` store partition."""

    source: str
    seasonally_adjusted: bool
    earliest_ref: date | None
    latest_ref: date | None
    row_count: int
    last_capture: date | None
    distinct_vintages: int


@dataclass(frozen=True)
class StoreStatus:
    """The full ``status`` report — header flags, coverage, and alarms."""

    store_uri: str
    is_remote: bool
    is_canonical: bool
    per_partition: list[PartitionCoverage] = field(default_factory=list)
    uncaptured: list[str] = field(default_factory=list)
    missing_months: list[str] = field(default_factory=list)
    corrected: list[str] = field(default_factory=list)


def _partition_coverage(store_path, source: str, sa: bool) -> PartitionCoverage | None:
    """Aggregate one partition via ``read_vintage_store``; None if empty/absent."""
    lf = read_vintage_store(store_path, source=source, seasonally_adjusted=sa)
    agg = lf.select(
        pl.len().alias("row_count"),
        pl.col("ref_date").min().alias("earliest_ref"),
        pl.col("ref_date").max().alias("latest_ref"),
        pl.col("vintage_date").max().alias("last_capture"),
        pl.col("vintage_date").n_unique().alias("distinct_vintages"),
    ).collect()
    row_count = int(agg.item(0, "row_count"))
    if row_count == 0:
        return None
    return PartitionCoverage(
        source=source,
        seasonally_adjusted=sa,
        earliest_ref=agg.item(0, "earliest_ref"),
        latest_ref=agg.item(0, "latest_ref"),
        row_count=row_count,
        last_capture=agg.item(0, "last_capture"),
        distinct_vintages=int(agg.item(0, "distinct_vintages")),
    )


def compute_status(
    store_path=VINTAGE_STORE_PATH,
    as_of: date | None = None,
) -> StoreStatus:
    """Read-only coverage + knowability report for the vintage store.

    Reads via ``read_vintage_store`` only. ``as_of`` (default: today) bounds the
    forward UNCAPTURED alarm (Task 7.2). Never calls ``transform_to_panel``.
    """
    if as_of is None:
        as_of = date.today()

    per_partition: list[PartitionCoverage] = []
    for source, sa in _PARTITIONS:
        cov = _partition_coverage(store_path, source, sa)
        if cov is not None:
            per_partition.append(cov)

    return StoreStatus(
        store_uri=str(store_path),
        is_remote=is_remote(store_path),
        is_canonical=is_canonical_store(store_path),
        per_partition=per_partition,
        uncaptured=[],
        missing_months=[],
        corrected=[],
    )
