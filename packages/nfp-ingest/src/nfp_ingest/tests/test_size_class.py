"""Tests for nfp_ingest.size_class — QCEW Q1 size cross-product (T4)."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from nfp_ingest.size_class import all_sizes_predicate, build_size_class_panel

_VINT = date(2024, 8, 1)


def _native(
    *,
    industry_code="21",
    industry_type="sector",
    ref_date=date(2024, 1, 12),
    revision=0,
    values=None,
) -> list[dict]:
    """One industry-month delivered at the nine native size_codes."""
    if values is None:
        values = {str(i): float(i * 10) for i in range(1, 10)}  # 10,20,...,90
    return [
        {
            "geographic_type": "national",
            "geographic_code": "00",
            "ownership": "private",
            "industry_type": industry_type,
            "industry_code": industry_code,
            "ref_date": ref_date,
            "vintage_date": _VINT,
            "revision": revision,
            "size_code": code,
            "employment": emp,
        }
        for code, emp in values.items()
    ]


def _emp(panel, sct, scc):
    return panel.filter(
        (pl.col("size_class_type") == sct) & (pl.col("size_class_code") == scc)
    )["employment"].item()


# ---------------------------------------------------------------------------
# Rollups
# ---------------------------------------------------------------------------


def test_large_passes_natives_through():
    panel = build_size_class_panel(pl.DataFrame(_native()))
    large = panel.filter(pl.col("size_class_type") == "large").sort("size_class_code")
    assert large["size_class_code"].to_list() == [str(i) for i in range(1, 10)]
    assert large["employment"].to_list() == pytest.approx([float(i * 10) for i in range(1, 10)])


def test_total_sums_all_natives():
    panel = build_size_class_panel(pl.DataFrame(_native()))
    assert _emp(panel, "total", "0") == pytest.approx(450.0)  # 10+...+90


def test_small_rollup():
    panel = build_size_class_panel(pl.DataFrame(_native()))
    assert _emp(panel, "small", "1") == pytest.approx(150.0)  # 10+20+30+40+50
    assert _emp(panel, "small", "2") == pytest.approx(130.0)  # 60+70
    assert _emp(panel, "small", "3") == pytest.approx(170.0)  # 80+90


def test_medium_rollup():
    panel = build_size_class_panel(pl.DataFrame(_native()))
    assert _emp(panel, "medium", "1") == pytest.approx(100.0)  # 10+20+30+40
    assert _emp(panel, "medium", "2") == pytest.approx(50.0)
    assert _emp(panel, "medium", "3") == pytest.approx(60.0)
    assert _emp(panel, "medium", "4") == pytest.approx(70.0)
    assert _emp(panel, "medium", "5") == pytest.approx(170.0)  # 80+90


def test_scheme_totals_all_equal():
    # Every scheme partitions the same employment, so each sums to the total.
    panel = build_size_class_panel(pl.DataFrame(_native()))
    for scheme in ("total", "small", "medium", "large"):
        s = panel.filter(pl.col("size_class_type") == scheme)["employment"].sum()
        assert s == pytest.approx(450.0)


# ---------------------------------------------------------------------------
# Invariants: no null-size rows; metadata; Q1-only
# ---------------------------------------------------------------------------


def test_no_null_size_rows():
    panel = build_size_class_panel(pl.DataFrame(_native()))
    assert panel.filter(pl.col("size_class_type").is_null()).height == 0
    assert panel.filter(pl.col("size_class_code").is_null()).height == 0


def test_metadata_inherited():
    panel = build_size_class_panel(pl.DataFrame(_native(revision=2)))
    assert (panel["ownership"] == "private").all()
    assert (panel["source"] == "qcew").all()
    assert (panel["benchmark_revision"] == 0).all()
    assert (panel["revision"] == 2).all()
    assert (~panel["seasonally_adjusted"]).all()


def test_q1_only_guard():
    with pytest.raises(ValueError, match="Q1-only"):
        build_size_class_panel(pl.DataFrame(_native(ref_date=date(2024, 4, 12))))


def test_missing_column_raises():
    with pytest.raises(ValueError, match="missing columns"):
        build_size_class_panel(pl.DataFrame({"industry_code": ["21"]}))


# ---------------------------------------------------------------------------
# All-sizes selector (§7): one row per Q1 industry-month
# ---------------------------------------------------------------------------


def test_all_sizes_selector_one_row_per_q1_month():
    rows = (
        _native(ref_date=date(2024, 1, 12))
        + _native(ref_date=date(2024, 2, 12))
        + _native(ref_date=date(2024, 3, 12))
    )
    panel = build_size_class_panel(pl.DataFrame(rows))
    all_sizes = panel.filter(all_sizes_predicate())
    # Exactly the three total/'0' rows — no double-count, no Q1 month dropped.
    assert all_sizes.height == 3
    assert (all_sizes["size_class_code"] == "0").all()
    assert all_sizes["ref_date"].sort().to_list() == [
        date(2024, 1, 12), date(2024, 2, 12), date(2024, 3, 12)
    ]
    assert all_sizes["employment"].to_list() == pytest.approx([450.0, 450.0, 450.0])


def test_all_sizes_selector_also_keeps_null_size_rows():
    # A null-size row (CES / QCEW Q2-Q4 convention) is selected too.
    null_row = pl.DataFrame({"size_class_type": [None], "size_class_code": [None]})
    assert null_row.filter(all_sizes_predicate()).height == 1
