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


class TestComposeQ1HeadlineCarriesAreaTotal:
    """§7 fix: the Q1 all-sizes ``'0'`` headline carries the area-levels total,
    not the disclosed-bucket sum.

    The size frame's ``'0'`` row is a sum over native buckets with suppressed
    (``disclosure_code='N'``) cells dropped, so it undercuts the published,
    un-suppressed area-levels total.  The compose overrides only the ``'0'``
    row's *employment* (metadata/vintage untouched) to the area value — buckets
    legitimately need not sum to it under suppression.
    """

    def test_suppressed_headline_overridden_to_area_total(self):
        q1 = date(2024, 1, 12)
        ces = _make_ces([])
        # Area-levels total = 100; bucket-sum '0' = 90 (10 suppressed).
        qcew = _make_qcew_levels([
            _qcew_level_row(ref_date=q1, industry_type="sector",
                            industry_code="32", employment=100.0),
        ])
        size = _make_size([
            _size_row(ref_date=q1, industry_type="sector", industry_code="32",
                      size_class_type="total", size_class_code="0", employment=90.0),
            _size_row(ref_date=q1, industry_type="sector", industry_code="32",
                      size_class_type="large", size_class_code="9", employment=90.0),
        ])
        result = compose_rebuild_panel(ces, qcew, size)

        zero = result.filter(
            (pl.col("industry_code") == "32") & (pl.col("size_class_code") == "0")
        )
        assert zero.height == 1
        # The headline is the area total (100), NOT the disclosed-bucket sum (90).
        assert zero["employment"][0] == pytest.approx(100.0)
        # Bucket rows are untouched (they legitimately undercount under suppression).
        large = result.filter(
            (pl.col("industry_code") == "32") & (pl.col("size_class_code") == "9")
        )
        assert large["employment"][0] == pytest.approx(90.0)
        # Still exactly one all-sizes QCEW row for the month.
        alls = result.filter(
            all_sizes_predicate()
            & (pl.col("source") == "qcew")
            & (pl.col("ref_date") == q1)
        )
        assert alls.height == 1

    def test_unsuppressed_headline_unchanged(self):
        """When no cells are suppressed (bucket-sum == area), '0' is unchanged."""
        q1 = date(2024, 1, 12)
        ces = _make_ces([])
        qcew = _make_qcew_levels([_qcew_level_row(ref_date=q1, employment=100.0)])
        size = _make_size([
            _size_row(ref_date=q1, size_class_type="total", size_class_code="0",
                      employment=100.0),
            _size_row(ref_date=q1, size_class_type="large", size_class_code="9",
                      employment=100.0),
        ])
        result = compose_rebuild_panel(ces, qcew, size)
        zero = result.filter(pl.col("size_class_code") == "0")
        assert zero.height == 1
        assert zero["employment"][0] == pytest.approx(100.0)

    def test_no_area_row_falls_back_to_bucket_sum(self):
        """A size '0' row whose series-month has no area-levels row keeps its sum.

        Fallback (``coalesce``): if the area endpoint did not return the series,
        the disclosed-bucket sum is the best available headline — never null it.
        """
        q1 = date(2024, 1, 12)
        ces = _make_ces([])
        qcew = _make_qcew_levels([])  # no area row at all
        size = _make_size([
            _size_row(ref_date=q1, size_class_type="total", size_class_code="0",
                      employment=90.0),
        ])
        result = compose_rebuild_panel(ces, qcew, size)
        zero = result.filter(pl.col("size_class_code") == "0")
        assert zero.height == 1
        assert zero["employment"][0] == pytest.approx(90.0)

    def test_additive_nesting_restored_at_q1(self):
        """'0'(05) == '0'(06) + '0'(08) after the override (area totals nest).

        The bucket-sum '0' rows break §3 additive closure at Q1 because
        suppression is uneven per industry; the area-levels totals nest by BLS
        construction, so carrying them restores ``05 = 06 + 08``.
        """
        q1 = date(2024, 1, 12)
        ces = _make_ces([])
        qcew = _make_qcew_levels([
            _qcew_level_row(ref_date=q1, industry_type="total", industry_code="05",
                            employment=100.0),
            _qcew_level_row(ref_date=q1, industry_type="domain", industry_code="06",
                            employment=30.0),
            _qcew_level_row(ref_date=q1, industry_type="domain", industry_code="08",
                            employment=70.0),
        ])
        # Bucket sums undercount unevenly (05=95, 06=28, 08=66 → 28+66=94 ≠ 95).
        size = _make_size([
            _size_row(ref_date=q1, industry_type="total", industry_code="05",
                      size_class_code="0", employment=95.0),
            _size_row(ref_date=q1, industry_type="domain", industry_code="06",
                      size_class_code="0", employment=28.0),
            _size_row(ref_date=q1, industry_type="domain", industry_code="08",
                      size_class_code="0", employment=66.0),
        ])
        result = compose_rebuild_panel(ces, qcew, size)

        def zero(code: str) -> float:
            return result.filter(
                (pl.col("industry_code") == code) & (pl.col("size_class_code") == "0")
            )["employment"][0]

        assert zero("05") == pytest.approx(100.0)
        assert zero("05") == pytest.approx(zero("06") + zero("08"))

    def test_multi_revision_area_aligns_to_size_revision(self):
        """With multi-revision qcew_levels, the override picks the area value at
        the size row's OWN revision — deterministically, not an arbitrary pick.

        The sibling anti-join (and ``test_different_revision_still_deduped``) is
        designed to tolerate multi-revision qcew_levels.  The value-override must
        be consistent: a rev-0 size '0' row aligns to the rev-0 area row, so the
        additive closure the fix restores can't be broken by mixing revisions
        across a parent and its children.  (A bare ``unique`` on the 6-col series
        identity would keep an arbitrary revision's employment — verified
        non-deterministic across physical orderings.)
        """
        q1 = date(2024, 1, 12)
        ces = _make_ces([])
        # Same series-month at three revisions with DIFFERENT area totals.
        qcew = _make_qcew_levels([
            _qcew_level_row(ref_date=q1, revision=0, vintage_date=_VINT, employment=100.0),
            _qcew_level_row(ref_date=q1, revision=1, vintage_date=_VINT2, employment=110.0),
            _qcew_level_row(ref_date=q1, revision=2, vintage_date=date(2025, 2, 1), employment=120.0),
        ])
        # The size '0' row is rev-0 (Decision A: QCEW size is rev-0) with a
        # bucket-sum that undercounts.
        size = _make_size([
            _size_row(ref_date=q1, revision=0, vintage_date=_VINT,
                      size_class_type="total", size_class_code="0", employment=90.0),
        ])
        result = compose_rebuild_panel(ces, qcew, size)
        zero = result.filter(pl.col("size_class_code") == "0")
        assert zero.height == 1
        # Must be the rev-0 area total (100), never 110/120 or the bucket-sum 90.
        assert zero["employment"][0] == pytest.approx(100.0)
        assert zero["revision"][0] == 0

    def test_non_zero_buckets_not_overridden(self):
        """Only the '0' row is overridden; small/medium/large/native are not."""
        q1 = date(2024, 1, 12)
        ces = _make_ces([])
        qcew = _make_qcew_levels([_qcew_level_row(ref_date=q1, employment=100.0)])
        size = _make_size([
            _size_row(ref_date=q1, size_class_type="total", size_class_code="0",
                      employment=90.0),
            _size_row(ref_date=q1, size_class_type="small", size_class_code="S",
                      employment=40.0),
            _size_row(ref_date=q1, size_class_type="large", size_class_code="9",
                      employment=50.0),
        ])
        result = compose_rebuild_panel(ces, qcew, size)
        assert result.filter(pl.col("size_class_code") == "S")["employment"][0] == pytest.approx(40.0)
        assert result.filter(pl.col("size_class_code") == "9")["employment"][0] == pytest.approx(50.0)
        assert result.filter(pl.col("size_class_code") == "0")["employment"][0] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# compose_rebuild_panel — SA CES + QCEW '00' total flow through (plans/11 T3)
# ---------------------------------------------------------------------------


class TestComposeCarriesSaAndTotal:
    """Regression guard: SA CES rows + the QCEW ``'00'`` total pass through the
    §7 Q1 override untouched.

    After plans/11 T1/T2, ``build_ces_panel`` emits SA rows (null size) and
    ``build_qcew_panel`` emits a ``('total','00','total')`` total track (null
    size, ``source='qcew'``).  ``compose_rebuild_panel`` unions ``ces``
    wholesale and the §7 override anti-joins only on the size frame's ``'0'``
    rows (private size cross-product), so neither SA CES nor the ``'00'`` total
    is ever dropped or mutated.  **This test is expected to PASS without any
    change to ``compose_rebuild_panel`` — it is the regression guard.**  If it
    fails, the §7 override is wrongly reaching CES or the ``'00'`` total and must
    be re-scoped to ``source='qcew'`` private size ``'0'`` rows.
    """

    def _setup(self):
        q1 = date(2024, 1, 12)
        # CES: an NSA + an SA row for total/00 (null size, share vintage_date).
        ces = _make_ces([
            _ces_row(industry_type="total", industry_code="00", ownership="total",
                     ref_date=q1, employment=158_000.0, seasonally_adjusted=False),
            _ces_row(industry_type="total", industry_code="00", ownership="total",
                     ref_date=q1, employment=156_000.0, seasonally_adjusted=True),
        ])
        # QCEW levels: the '00' total track + a private sector/32 (which the size
        # frame covers, so it is the one exercised by the §7 override/anti-join).
        qcew = _make_qcew_levels([
            _qcew_level_row(industry_type="total", industry_code="00",
                            ownership="total", ref_date=q1, employment=152_000.0),
            _qcew_level_row(industry_type="sector", industry_code="32",
                            ownership="private", ref_date=q1, employment=100.0),
        ])
        # Size: private sector/32 cross-product. '0' bucket-sum undercounts (90)
        # the area total (100) — so the §7 override must lift it to 100.
        size = _make_size([
            _size_row(ref_date=q1, industry_type="sector", industry_code="32",
                      ownership="private", size_class_type="total",
                      size_class_code="0", employment=90.0),
            _size_row(ref_date=q1, industry_type="sector", industry_code="32",
                      ownership="private", size_class_type="large",
                      size_class_code="9", employment=90.0),
        ])
        return ces, qcew, size, q1

    def test_sa_ces_rows_present_and_unchanged(self):
        ces, qcew, size, q1 = self._setup()
        result = compose_rebuild_panel(ces, qcew, size)
        sa = result.filter(
            (pl.col("source") == "ces") & pl.col("seasonally_adjusted")
        )
        assert sa.height == 1
        # Value untouched (SA 00 == 156_000) and null size.
        assert sa["employment"][0] == pytest.approx(156_000.0)
        assert sa["size_class_type"][0] is None
        # The NSA CES 00 row also survives unchanged.
        nsa = result.filter(
            (pl.col("source") == "ces")
            & (pl.col("industry_code") == "00")
            & ~pl.col("seasonally_adjusted")
        )
        assert nsa.height == 1
        assert nsa["employment"][0] == pytest.approx(158_000.0)

    def test_qcew_00_total_present_and_unchanged(self):
        ces, qcew, size, q1 = self._setup()
        result = compose_rebuild_panel(ces, qcew, size)
        total = result.filter(
            (pl.col("source") == "qcew")
            & (pl.col("industry_code") == "00")
            & (pl.col("ownership") == "total")
        )
        # The '00' total is a source='qcew' null-size row; the §7 anti-join keys
        # off the size frame's private '0' rows, so '00' is never dropped.
        assert total.height == 1
        assert total["employment"][0] == pytest.approx(152_000.0)
        assert total["size_class_type"][0] is None

    def test_section_7_override_still_works_on_private_size_zero(self):
        ces, qcew, size, q1 = self._setup()
        result = compose_rebuild_panel(ces, qcew, size)
        # The private size '0' headline is lifted to the area total (100), and the
        # null-size private level row is anti-joined away.
        zero = result.filter(
            (pl.col("industry_code") == "32") & (pl.col("size_class_code") == "0")
        )
        assert zero.height == 1
        assert zero["employment"][0] == pytest.approx(100.0)
        # The null-size qcew_levels row for the covered private sector is dropped.
        dropped = result.filter(
            (pl.col("source") == "qcew")
            & (pl.col("industry_code") == "32")
            & pl.col("size_class_type").is_null()
        )
        assert dropped.height == 0


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
