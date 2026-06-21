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
from nfp_vintages.__main__ import app
from nfp_vintages.store_status import (
    PartitionCoverage,
    StoreStatus,
    compute_status,
    format_status,
)
from typer.testing import CliRunner


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


def test_compute_status_flags_uncaptured_ces_month(tmp_path):
    """Store lags the calendar: published-but-absent CES ref-months are flagged."""
    # Store stops at Aug-2025; as-of 2026-01-12 → Sep/Oct/Nov rev0 are out by then.
    rows = [
        _row(
            source="ces",
            sa=True,
            ref_date=date(2025, 7, 1),
            vintage_date=date(2025, 8, 1),
            employment=158000.0,
        ),
        _row(
            source="ces",
            sa=True,
            ref_date=date(2025, 8, 1),
            vintage_date=date(2025, 9, 5),
            employment=158200.0,
        ),
    ]
    _write_store_rows(tmp_path, rows)

    status = compute_status(tmp_path, as_of=date(2026, 1, 12))

    joined = " ".join(status.uncaptured)
    # Entries use "ces:<ISO-date>" format (Phase 8 contract).
    assert "ces:" in joined
    # At least Sep-2025 should be reported uncaptured (rev0 published ~Oct-2025).
    assert "2025-09-01" in joined


def test_compute_status_uncaptured_uses_colon_format(tmp_path):
    """uncaptured entries must be 'src:<ref_token>' (Phase 8 parse contract)."""
    rows = [
        _row(
            source="ces",
            sa=True,
            ref_date=date(2025, 8, 1),
            vintage_date=date(2025, 9, 5),
            employment=158200.0,
        ),
    ]
    _write_store_rows(tmp_path, rows)
    status = compute_status(tmp_path, as_of=date(2026, 1, 12))
    for entry in status.uncaptured:
        src, _, ref = entry.partition(":")
        assert src in {"ces", "qcew"}, f"bad source in {entry!r}"
        assert ref, f"empty ref_token in {entry!r}"


def test_oct_2025_sentinel_not_flagged_missing(tmp_path):
    """A -1 sentinel row at Oct-2025 counts as present (raw row presence)."""
    rows = [
        _row(
            source="ces",
            sa=True,
            ref_date=date(2025, 9, 1),
            vintage_date=date(2025, 11, 20),
            employment=159000.0,
        ),
        # The shutdown "no print" sentinel: literal -1.0 at the Oct-2025 slot.
        _row(
            source="ces",
            sa=True,
            ref_date=date(2025, 10, 1),
            vintage_date=date(2025, 12, 16),
            employment=-1.0,
        ),
        _row(
            source="ces",
            sa=True,
            ref_date=date(2025, 11, 1),
            vintage_date=date(2025, 12, 16),
            employment=159100.0,
        ),
    ]
    _write_store_rows(tmp_path, rows)

    status = compute_status(tmp_path, as_of=date(2026, 1, 12))

    # Oct-2025 has a (sentinel) row → NOT an interior hole.
    missing = " ".join(status.missing_months)
    assert "2025-10" not in missing


def test_format_status_local_fallback_warning(tmp_path):
    """A local (non-remote) store renders the .env LOCAL-FALLBACK warning."""
    rows = [
        _row(
            source="ces",
            sa=True,
            ref_date=date(2025, 9, 1),
            vintage_date=date(2025, 11, 20),
            employment=159000.0,
        ),
    ]
    _write_store_rows(tmp_path, rows)

    text = format_status(compute_status(tmp_path, as_of=date(2025, 12, 12)))

    assert "LOCAL FALLBACK" in text
    assert "NFP_STORE_URI" in text
    assert "ces" in text


def test_status_command_renders_report(tmp_path):
    """`alt-nfp status --store <tmp> --as-of D` prints the coverage report."""
    rows = [
        _row(
            source="ces",
            sa=True,
            ref_date=date(2025, 9, 1),
            vintage_date=date(2025, 11, 20),
            employment=159000.0,
        ),
        _row(
            source="qcew",
            sa=False,
            ref_date=date(2025, 1, 1),
            vintage_date=date(2025, 9, 1),
            employment=140000.0,
        ),
    ]
    _write_store_rows(tmp_path, rows)

    result = CliRunner().invoke(
        app,
        ["status", "--store", str(tmp_path), "--as-of", "2025-12-12"],
    )

    assert result.exit_code == 0, result.output
    assert "coverage" in result.output
    assert "ces" in result.output
    assert "qcew" in result.output


def test_missing_months_flags_real_interior_gap(tmp_path):
    """A real interior headline gap (no row, not a shutdown) is reported bare.

    Positive coverage for _missing_headline_months — without this the body could
    be stubbed to ``return []`` and every other test still passes.
    """
    # CES-SA headline present Jan + Mar 2024; Feb absent → a true interior hole.
    rows = [
        _row(
            source="ces",
            sa=True,
            ref_date=date(2024, 1, 1),
            vintage_date=date(2024, 2, 5),
            employment=158000.0,
        ),
        _row(
            source="ces",
            sa=True,
            ref_date=date(2024, 3, 1),
            vintage_date=date(2024, 4, 5),
            employment=158400.0,
        ),
    ]
    _write_store_rows(tmp_path, rows)

    status = compute_status(tmp_path, as_of=date(2024, 6, 12))

    # Feb-2024 is a bare interior gap (not a known-shutdown month).
    assert "2024-02" in status.missing_months
    assert "[known-shutdown]" not in " ".join(status.missing_months)


def test_missing_months_annotates_known_shutdown(tmp_path):
    """A wholly-absent known-shutdown month is annotated, not bare-flagged."""
    # Sep + Nov 2025 present, Oct-2025 (a known shutdown month) wholly absent.
    rows = [
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
            ref_date=date(2025, 11, 1),
            vintage_date=date(2025, 12, 16),
            employment=159100.0,
        ),
    ]
    _write_store_rows(tmp_path, rows)

    status = compute_status(tmp_path, as_of=date(2026, 1, 12))

    assert "2025-10 [known-shutdown]" in status.missing_months


def test_qcew_uncaptured_token_format(tmp_path):
    """QCEW uncaptured entries are 'qcew:<YYYY>-Q<n>' (Phase 8 split('-Q') contract).

    The CES format test's store has only a CES row, so the QCEW leg never runs
    there; this exercises the Phase-8 'do NOT change' QCEW token round-trip.
    """
    # Store holds only QCEW Q1-2024; by mid-2025 later quarters are knowable.
    rows = [
        _row(
            source="qcew",
            sa=False,
            ref_date=date(2024, 1, 1),
            vintage_date=date(2024, 9, 1),
            employment=140000.0,
        ),
    ]
    _write_store_rows(tmp_path, rows)

    status = compute_status(tmp_path, as_of=date(2025, 7, 1))

    qcew_entries = [u for u in status.uncaptured if u.startswith("qcew:")]
    assert qcew_entries, "expected at least one qcew: uncaptured entry"
    for entry in qcew_entries:
        ref_token = entry.split(":", 1)[1]
        # Phase 8 detects QCEW via '-Q' then splits on it.
        assert "-Q" in ref_token
        year_str, q_str = ref_token.split("-Q")
        assert year_str.isdigit()
        assert q_str.isdigit() and 1 <= int(q_str) <= 4
