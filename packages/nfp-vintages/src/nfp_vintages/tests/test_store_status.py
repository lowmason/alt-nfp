"""Tests for the read-only ``status`` report (spec §8).

All store I/O is against a synthetic Hive-partitioned tmp_path store built by
``_write_store_rows`` below — NEVER a real MinIO store (conftest auto-loads prod
creds). ``compute_status`` reads via ``read_vintage_store`` and must never call
``transform_to_panel``.
"""

from __future__ import annotations

from datetime import date

import polars as pl
from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA
from nfp_vintages.store_status import (
    PartitionCoverage,
    StoreStatus,
    compute_status,
)


def _row(
    *,
    source: str,
    sa: bool,
    ref_date: date,
    vintage_date: date,
    revision: int = 0,
    benchmark_revision: int = 0,
    employment: float = 100.0,
    industry_code: str = "00",
    geographic_code: str = "00",
) -> dict:
    """One VINTAGE_STORE_SCHEMA row as a dict (defaults = national total headline)."""
    return {
        "geographic_type": "national",
        "geographic_code": geographic_code,
        "ownership": "total",
        "industry_type": "total",
        "industry_code": industry_code,
        "ref_date": ref_date,
        "vintage_date": vintage_date,
        "revision": revision,
        "benchmark_revision": benchmark_revision,
        "employment": employment,
        "size_class_type": None,
        "size_class_code": None,
        "source": source,
        "seasonally_adjusted": sa,
    }


def _write_store_rows(store_path, rows: list[dict]) -> None:
    """Write rows into the Hive layout the store reader expects.

    Partitions on (source, seasonally_adjusted); the partition columns are
    encoded in the directory names (Hive), so they are dropped from the file.
    """
    df = pl.DataFrame(rows, schema=VINTAGE_STORE_SCHEMA)
    for (source, sa), part in df.group_by(["source", "seasonally_adjusted"]):
        part_dir = (
            store_path
            / f"source={source}"
            / f"seasonally_adjusted={str(sa).lower()}"
        )
        part_dir.mkdir(parents=True, exist_ok=True)
        part.drop("source", "seasonally_adjusted").write_parquet(
            part_dir / "part-0.parquet"
        )


def test_compute_status_partition_coverage(tmp_path):
    """One PartitionCoverage per (source, sa); raw row presence, sentinel counts."""
    rows = [
        # CES SA: two months, the second is the Oct-2025 -1 sentinel slot.
        _row(
            source="ces",
            sa=True,
            ref_date=date(2025, 9, 1),
            vintage_date=date(2025, 11, 20),
            employment=159000.0,
        ),
        _row(
            source="ces",
            sa=True,
            ref_date=date(2025, 10, 1),
            vintage_date=date(2025, 12, 16),
            employment=-1.0,
        ),
        # QCEW NSA: one quarter.
        _row(
            source="qcew",
            sa=False,
            ref_date=date(2025, 1, 1),
            vintage_date=date(2025, 9, 1),
            employment=140000.0,
        ),
    ]
    _write_store_rows(tmp_path, rows)

    status = compute_status(tmp_path, as_of=date(2026, 1, 12))

    assert isinstance(status, StoreStatus)
    parts = {(p.source, p.seasonally_adjusted): p for p in status.per_partition}
    assert set(parts) == {("ces", True), ("qcew", False)}

    ces = parts[("ces", True)]
    assert isinstance(ces, PartitionCoverage)
    # Both CES rows counted — the -1 sentinel is NOT filtered out.
    assert ces.row_count == 2
    assert ces.earliest_ref == date(2025, 9, 1)
    assert ces.latest_ref == date(2025, 10, 1)
    assert ces.last_capture == date(2025, 12, 16)
    assert ces.distinct_vintages == 2

    qcew = parts[("qcew", False)]
    assert qcew.row_count == 1
    assert qcew.latest_ref == date(2025, 1, 1)
    assert qcew.last_capture == date(2025, 9, 1)
