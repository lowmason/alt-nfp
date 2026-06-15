"""Tests for nfp_vintages.rebuild_store — compose + guarded write (store_rebuild T5).

All tests use synthetic Polars frames only. No network access, no real/remote
store writes. Write tests use pytest tmp_path (local dirs) exclusively.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from nfp_ingest.size_class import all_sizes_predicate
from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA
from nfp_vintages.rebuild_store import compose_rebuild_panel, write_rebuild_store

# ---------------------------------------------------------------------------
# Synthetic frame helpers
# ---------------------------------------------------------------------------

_VINT = date(2024, 8, 1)
_VINT2 = date(2024, 11, 1)


def _schema_row(**overrides) -> dict:
    """Minimal VINTAGE_STORE_SCHEMA row; override any field."""
    base = {
        "geographic_type": "national",
        "geographic_code": "00",
        "ownership": "private",
        "industry_type": "sector",
        "industry_code": "21",
        "ref_date": date(2024, 1, 12),
        "vintage_date": _VINT,
        "revision": 0,
        "benchmark_revision": 0,
        "employment": 100.0,
        "size_class_type": None,
        "size_class_code": None,
        "source": "qcew",
        "seasonally_adjusted": False,
    }
    base.update(overrides)
    return base


def _ces_row(**overrides) -> dict:
    row = _schema_row(source="ces", seasonally_adjusted=False)
    row.update(overrides)
    return row


def _qcew_level_row(**overrides) -> dict:
    """A null-size QCEW levels row (size columns absent — omitted entirely)."""
    row = _schema_row(source="qcew", seasonally_adjusted=False)
    # qcew_levels builder omits size_class_type/size_class_code entirely
    row.pop("size_class_type")
    row.pop("size_class_code")
    row.update(overrides)
    return row


def _size_row(size_class_type="total", size_class_code="0", **overrides) -> dict:
    """A QCEW Q1 size row (non-null size columns)."""
    row = _schema_row(
        source="qcew",
        seasonally_adjusted=False,
        size_class_type=size_class_type,
        size_class_code=size_class_code,
        employment=200.0,
    )
    row.update(overrides)
    return row


def _make_ces(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=dict(VINTAGE_STORE_SCHEMA))


def _make_qcew_levels(rows: list[dict]) -> pl.DataFrame:
    """Build a qcew_levels frame WITHOUT size columns (as build_qcew_panel returns)."""
    schema_no_size = {
        k: v
        for k, v in VINTAGE_STORE_SCHEMA.items()
        if k not in ("size_class_type", "size_class_code")
    }
    return pl.DataFrame(rows, schema=schema_no_size)


def _make_size(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=dict(VINTAGE_STORE_SCHEMA))


# ---------------------------------------------------------------------------
# compose_rebuild_panel — size=None (CES + qcew_levels union)
# ---------------------------------------------------------------------------


class TestComposeNone:
    """compose_rebuild_panel with size=None: straight union, null size cols."""

    def test_union_row_count(self):
        ces = _make_ces([_ces_row(industry_code="00"), _ces_row(industry_code="05")])
        qcew = _make_qcew_levels([_qcew_level_row(industry_code="21")])
        result = compose_rebuild_panel(ces, qcew)
        assert result.height == 3

    def test_schema_conformant(self):
        ces = _make_ces([_ces_row()])
        qcew = _make_qcew_levels([_qcew_level_row()])
        result = compose_rebuild_panel(ces, qcew)
        for col, dtype in VINTAGE_STORE_SCHEMA.items():
            assert col in result.columns, f"missing column {col!r}"
            assert result[col].dtype == dtype, (
                f"column {col!r}: expected {dtype}, got {result[col].dtype}"
            )

    def test_drifted_input_dtype_is_corrected(self):
        """A builder emitting a wrong dtype must not leak through diagonal_relaxed.

        The relaxed concat coerces to a common supertype; without the explicit
        schema cast in compose, an i64 ``revision``/``employment`` in any input
        frame would widen the composed column away from VINTAGE_STORE_SCHEMA and
        silently corrupt the store write. The aligned-input happy path
        (``test_schema_conformant``) cannot catch this — only a drifted input can.
        """
        ces = _make_ces([_ces_row()])
        schema_drift = {
            k: v
            for k, v in VINTAGE_STORE_SCHEMA.items()
            if k not in ("size_class_type", "size_class_code")
        }
        schema_drift["revision"] = pl.Int64
        schema_drift["employment"] = pl.Int64
        qcew = pl.DataFrame([_qcew_level_row(industry_code="21", employment=100)], schema=schema_drift)
        result = compose_rebuild_panel(ces, qcew)
        assert result["revision"].dtype == pl.UInt8
        assert result["employment"].dtype == pl.Float64
        assert dict(result.schema) == dict(VINTAGE_STORE_SCHEMA)

    def test_qcew_levels_size_cols_become_null(self):
        # Use distinct industry_codes so the filter is unambiguous.
        ces = _make_ces([_ces_row(industry_code="00", size_class_type=None, size_class_code=None)])
        qcew = _make_qcew_levels([_qcew_level_row(industry_code="21")])
        result = compose_rebuild_panel(ces, qcew)
        qcew_rows = result.filter(
            (pl.col("industry_code") == "21") & (pl.col("source") == "qcew")
        )
        assert qcew_rows.height == 1
        assert qcew_rows["size_class_type"][0] is None
        assert qcew_rows["size_class_code"][0] is None

    def test_no_rows_dropped_when_size_none(self):
        # Twelve QCEW rows (all months), no dedup should happen.
        rows = [
            _qcew_level_row(ref_date=date(2024, m, 12), revision=0)
            for m in range(1, 13)
        ]
        qcew = _make_qcew_levels(rows)
        ces = _make_ces([])
        result = compose_rebuild_panel(ces, qcew)
        assert result.height == 12

    def test_sources_preserved(self):
        ces = _make_ces([_ces_row()])
        qcew = _make_qcew_levels([_qcew_level_row()])
        result = compose_rebuild_panel(ces, qcew)
        assert set(result["source"].to_list()) == {"ces", "qcew"}


# ---------------------------------------------------------------------------
# compose_rebuild_panel — with size: anti-join dedup
# ---------------------------------------------------------------------------


class TestComposeWithSize:
    """With size provided: covered Q1 level rows are replaced by size '0' rows."""

    def _covered_setup(self):
        """One Q1 industry-month in both qcew_levels and size."""
        q1_date = date(2024, 1, 12)
        ces = _make_ces([_ces_row(ref_date=q1_date)])
        qcew = _make_qcew_levels([_qcew_level_row(ref_date=q1_date)])
        # size frame: total/'0' + one size bucket
        size = _make_size([
            _size_row(size_class_type="total", size_class_code="0", ref_date=q1_date),
            _size_row(size_class_type="large", size_class_code="9", ref_date=q1_date, employment=80.0),
        ])
        return ces, qcew, size, q1_date

    def test_exactly_one_all_sizes_row_for_covered_month(self):
        """The qcew_levels null-size row is dropped; the size total/'0' row survives."""
        ces, qcew, size, q1_date = self._covered_setup()
        result = compose_rebuild_panel(ces, qcew, size)

        all_sizes = result.filter(
            all_sizes_predicate()
            & (pl.col("source") == "qcew")
            & (pl.col("ref_date") == q1_date)
        )
        # Exactly one all-sizes QCEW row: the size '0' row, not the null-size level row.
        assert all_sizes.height == 1, (
            f"expected 1 all-sizes QCEW row, got {all_sizes.height};\n{all_sizes}"
        )
        assert all_sizes["size_class_code"][0] == "0"
        assert all_sizes["size_class_type"][0] == "total"

    def test_null_size_level_row_is_dropped(self):
        """The null-size qcew_levels row for the covered month must not appear."""
        ces, qcew, size, q1_date = self._covered_setup()
        result = compose_rebuild_panel(ces, qcew, size)

        null_size_qcew = result.filter(
            (pl.col("source") == "qcew")
            & (pl.col("ref_date") == q1_date)
            & pl.col("size_class_type").is_null()
        )
        assert null_size_qcew.height == 0, (
            "null-size qcew_levels row for covered month must be dropped"
        )

    def test_ces_null_size_row_survives(self):
        """CES rows are never part of the anti-join (they are not qcew_levels)."""
        ces, qcew, size, q1_date = self._covered_setup()
        result = compose_rebuild_panel(ces, qcew, size)

        ces_rows = result.filter(pl.col("source") == "ces")
        assert ces_rows.height == 1
        assert ces_rows["size_class_type"][0] is None

    def test_size_bucket_rows_present(self):
        """Non-total size rows survive in the result."""
        ces, qcew, size, q1_date = self._covered_setup()
        result = compose_rebuild_panel(ces, qcew, size)
        large_rows = result.filter(
            (pl.col("size_class_type") == "large")
            & (pl.col("ref_date") == q1_date)
        )
        assert large_rows.height == 1
        assert large_rows["size_class_code"][0] == "9"


# ---------------------------------------------------------------------------
# Partial coverage (critical): some months covered, some not
# ---------------------------------------------------------------------------


class TestPartialCoverage:
    """An industry-month without a size row must keep its null-size level row."""

    def _partial_setup(self):
        q1_covered = date(2024, 1, 12)   # has size coverage
        q1_uncovered = date(2024, 2, 12)  # no size row for this month

        ces = _make_ces([])  # no CES rows needed for this test
        qcew = _make_qcew_levels([
            _qcew_level_row(ref_date=q1_covered, employment=100.0),
            _qcew_level_row(ref_date=q1_uncovered, employment=150.0),
        ])
        # Only Jan has size coverage
        size = _make_size([
            _size_row(size_class_type="total", size_class_code="0",
                      ref_date=q1_covered, employment=100.0),
            _size_row(size_class_type="large", size_class_code="9",
                      ref_date=q1_covered, employment=80.0),
        ])
        return ces, qcew, size, q1_covered, q1_uncovered

    def test_uncovered_month_keeps_level_row(self):
        ces, qcew, size, q1_cov, q1_uncov = self._partial_setup()
        result = compose_rebuild_panel(ces, qcew, size)

        uncov_rows = result.filter(
            (pl.col("source") == "qcew") & (pl.col("ref_date") == q1_uncov)
        )
        # Must have exactly one row: the null-size level row.
        assert uncov_rows.height == 1
        assert uncov_rows["size_class_type"][0] is None
        assert uncov_rows["employment"][0] == pytest.approx(150.0)

    def test_covered_month_level_row_dropped(self):
        ces, qcew, size, q1_cov, q1_uncov = self._partial_setup()
        result = compose_rebuild_panel(ces, qcew, size)

        cov_null_size = result.filter(
            (pl.col("source") == "qcew")
            & (pl.col("ref_date") == q1_cov)
            & pl.col("size_class_type").is_null()
        )
        assert cov_null_size.height == 0

    def test_both_months_have_all_sizes_row(self):
        """All-sizes selector must return exactly one row per Q1 month."""
        ces, qcew, size, q1_cov, q1_uncov = self._partial_setup()
        result = compose_rebuild_panel(ces, qcew, size)

        all_sizes_qcew = result.filter(
            all_sizes_predicate() & (pl.col("source") == "qcew")
        ).sort("ref_date")
        assert all_sizes_qcew.height == 2
        dates = all_sizes_qcew["ref_date"].to_list()
        assert q1_cov in dates
        assert q1_uncov in dates


# ---------------------------------------------------------------------------
# Non-Q1 (e.g. June) rows are never dropped
# ---------------------------------------------------------------------------


class TestNonQ1NeverDropped:
    """June QCEW level rows are never touched — no size rows exist for them."""

    def test_non_q1_qcew_level_rows_survive(self):
        june = date(2024, 6, 12)
        jan = date(2024, 1, 12)  # covered

        ces = _make_ces([])
        qcew = _make_qcew_levels([
            _qcew_level_row(ref_date=june, employment=200.0),
        ])
        # Size coverage only for Jan
        size = _make_size([
            _size_row(size_class_type="total", size_class_code="0",
                      ref_date=jan, employment=100.0),
        ])
        result = compose_rebuild_panel(ces, qcew, size)

        june_rows = result.filter(pl.col("ref_date") == june)
        assert june_rows.height == 1
        assert june_rows["size_class_type"][0] is None


# ---------------------------------------------------------------------------
# Anti-join uses 6-col key, not vintage_date / revision
# ---------------------------------------------------------------------------


class TestAntiJoinKey:
    """Anti-join must match on series identity only (6 cols), not vintage/revision."""

    def test_different_revision_still_deduped(self):
        """
        A qcew_levels row with revision=1 and a size row with revision=0 for the
        same series-month: the level row must still be dropped (same series identity).
        """
        q1_date = date(2024, 1, 12)
        ces = _make_ces([])
        qcew = _make_qcew_levels([
            _qcew_level_row(ref_date=q1_date, revision=1, vintage_date=_VINT2),
        ])
        size = _make_size([
            _size_row(ref_date=q1_date, revision=0, vintage_date=_VINT,
                      size_class_type="total", size_class_code="0"),
        ])
        result = compose_rebuild_panel(ces, qcew, size)

        null_size_qcew = result.filter(
            (pl.col("source") == "qcew") & pl.col("size_class_type").is_null()
        )
        assert null_size_qcew.height == 0, (
            "level row with different revision must still be dropped when size coverage exists"
        )


# ---------------------------------------------------------------------------
# write_rebuild_store — guard + local write
# ---------------------------------------------------------------------------


class TestWriteRebuildStore:
    """Guard fires before any write; local write succeeds and is readable."""

    @pytest.fixture()
    def _no_real_store(self, monkeypatch):
        """Ensure no real store creds are active during write tests."""
        for var in (
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_ENDPOINT_URL",
            "AWS_SESSION_TOKEN",
            "NFP_STORE_URI",
        ):
            monkeypatch.delenv(var, raising=False)

    def test_raises_for_canonical_store(self, _no_real_store):
        """RuntimeError raised before any write when target is the canonical store."""
        from upath import UPath

        panel = _make_ces([_ces_row()])
        with pytest.raises(RuntimeError, match="canonical"):
            write_rebuild_store(panel, store_path=UPath("s3://alt-nfp/store"))

    def test_raises_for_canonical_store_trailing_slash(self, _no_real_store):
        """Canonical guard triggers even when the path has a trailing slash."""
        from upath import UPath

        panel = _make_ces([_ces_row()])
        with pytest.raises(RuntimeError, match="canonical"):
            write_rebuild_store(panel, store_path=UPath("s3://alt-nfp/store/"))

    def test_allow_canonical_flag_accepted_on_local_write(self, tmp_path, _no_real_store):
        """``allow_canonical=True`` is a valid kwarg and does not break a normal write.

        This does NOT exercise the guard *bypass* — ``tmp_path`` is local, so
        ``is_canonical_store`` returns False and the guard never fires regardless
        of the flag. Proving the bypass against a real canonical target would
        require an unsafe remote write; the guard's False branch (raise when the
        flag is absent) is covered by ``test_raises_for_canonical_store`` above.
        """
        panel = _make_ces([_ces_row()])
        write_rebuild_store(panel, store_path=tmp_path, allow_canonical=True)

    def test_local_write_success(self, tmp_path, _no_real_store):
        """Writing to a local tmp_path succeeds; partitions are readable."""
        ces_row = _ces_row(industry_code="21", source="ces", seasonally_adjusted=False)
        qcew_row = _schema_row(source="qcew", seasonally_adjusted=False)
        panel = _make_ces([ces_row, qcew_row])
        write_rebuild_store(panel, store_path=tmp_path)

        # Partitions written under source=ces/seasonally_adjusted=false/
        ces_part = tmp_path / "source=ces" / "seasonally_adjusted=false"
        qcew_part = tmp_path / "source=qcew" / "seasonally_adjusted=false"
        assert ces_part.exists(), f"CES partition dir missing: {ces_part}"
        assert qcew_part.exists(), f"QCEW partition dir missing: {qcew_part}"

        ces_files = list(ces_part.glob("*.parquet"))
        qcew_files = list(qcew_part.glob("*.parquet"))
        assert len(ces_files) == 1
        assert len(qcew_files) == 1

        # Read back and check row counts; partition cols are dropped in the file
        ces_df = pl.read_parquet(ces_files[0])
        qcew_df = pl.read_parquet(qcew_files[0])
        assert ces_df.height == 1
        assert qcew_df.height == 1

        # source/seasonally_adjusted are NOT in the parquet (they are partition dirs)
        assert "source" not in ces_df.columns
        assert "seasonally_adjusted" not in ces_df.columns

        # Schema columns (minus partition cols) must be present
        for col in VINTAGE_STORE_SCHEMA:
            if col not in ("source", "seasonally_adjusted"):
                assert col in ces_df.columns, f"missing column {col!r} in written parquet"

    def test_local_write_scratch_prefix_not_guarded(self, tmp_path, _no_real_store):
        """A scratch-like local path is never guarded."""
        panel = _make_ces([_ces_row()])
        # Should not raise
        write_rebuild_store(panel, store_path=tmp_path / "store-rebuild")

    def test_scratch_remote_not_canonical(self):
        """s3://alt-nfp/store-rebuild is NOT the canonical store — no-I/O assertion."""
        from nfp_lookups.paths import is_canonical_store
        from upath import UPath

        assert is_canonical_store(UPath("s3://alt-nfp/store-rebuild")) is False
