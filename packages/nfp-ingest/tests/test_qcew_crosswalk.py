"""Tests for nfp_ingest.qcew_crosswalk — QCEW→CES private rebuild (T3).

Synthetic QCEW bulk frames only (no network, no store). The fixtures are built
internally consistent (each agglvl-13 supersector equals the sum of its member
sectors) so additive nesting is exact and checkable.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from nfp_ingest.qcew_crosswalk import build_qcew_panel
from nfp_lookups.revision_schedules import get_qcew_vintage_date

# --- Synthetic leaf values (persons) -------------------------------------

# agglvl-14 NAICS sectors → value (equal across the quarter's three months).
_SECTOR14 = {
    "21": ("21", 6_000),
    "22": ("22", 7_000),
    "23": ("23", 80_000),
    "42": ("42", 41_000),
    "44": ("44-45", 42_000),
    "48": ("48-49", 43_000),
    "51": ("51", 51_000),
    "52": ("52", 52_000),
    "53": ("53", 53_000),
    "54": ("54", 54_000),
    "55": ("55", 55_000),
    "56": ("56", 56_000),
    "61": ("61", 61_000),
    "62": ("62", 62_000),
    "71": ("71", 71_000),
    "72": ("72", 72_000),
    "81": ("81", 81_000),
}
# agglvl-16 logging (CES sector 11).
_LOGGING = ("1133", 5_000)
# agglvl-15 3-digit mfg → durable (31)=31_000, nondurable (32)=32_000.
_DURABLE = {"321": 30_000, "331": 1_000}
_NONDURABLE = {"311": 31_000, "322": 1_000}

# Derived CES sector totals (for computing the consistent agglvl-13 rollups).
_SECTOR_TOTAL = {
    "11": 5_000,
    **{k: v for k, (_, v) in _SECTOR14.items()},
    "31": sum(_DURABLE.values()),      # 31_000
    "32": sum(_NONDURABLE.values()),   # 32_000
}
# CES supersector → (agglvl-13 QCEW code, member CES sectors).
_SS13 = {
    "20": ("1012", ["23"]),
    "30": ("1013", ["31", "32"]),
    "40": ("1021", ["42", "44", "48", "22"]),
    "50": ("1022", ["51"]),
    "55": ("1023", ["52", "53"]),
    "60": ("1024", ["54", "55", "56"]),
    "65": ("1025", ["61", "62"]),
    "70": ("1026", ["71", "72"]),
    "80": ("1027", ["81"]),
}


def _row(industry_code, agglvl, m, *, year, qtr, revision, own_code: str = "5"):
    return {
        "area_fips": "US000",
        "own_code": own_code,
        "industry_code": industry_code,
        "agglvl_code": agglvl,
        "year": year,
        "qtr": qtr,
        "month1_emplvl": m,
        "month2_emplvl": m,
        "month3_emplvl": m,
        "revision": revision,
    }


def _make_qcew(*, year=2024, qtr=1, revision=0, scale=1.0) -> list[dict]:
    """Build a complete, internally consistent national-private QCEW frame."""
    rows: list[dict] = []
    s = lambda v: int(v * scale)  # noqa: E731
    # Leaf sectors at agglvl 14.
    for _ces, (qcode, val) in _SECTOR14.items():
        rows.append(_row(qcode, "14", s(val), year=year, qtr=qtr, revision=revision))
    # Logging (agglvl 16) and 3-digit mfg (agglvl 15).
    rows.append(_row(_LOGGING[0], "16", s(_LOGGING[1]), year=year, qtr=qtr, revision=revision))
    for code, val in {**_DURABLE, **_NONDURABLE}.items():
        rows.append(_row(code, "15", s(val), year=year, qtr=qtr, revision=revision))
    # agglvl-13 supersector aggregates = sum of member CES sectors (consistent).
    for _ss, (qcode, members) in _SS13.items():
        val = sum(_SECTOR_TOTAL[m] for m in members)
        rows.append(_row(qcode, "13", s(val), year=year, qtr=qtr, revision=revision))
    return rows


def _emp(panel, industry_type, industry_code, ref_date):
    return panel.filter(
        (pl.col("industry_type") == industry_type)
        & (pl.col("industry_code") == industry_code)
        & (pl.col("ref_date") == ref_date)
    )["employment"].item()


_JAN = date(2024, 1, 12)


# ---------------------------------------------------------------------------
# Crosswalk sums + ÷1000 units
# ---------------------------------------------------------------------------


def test_units_divided_by_1000():
    panel = build_qcew_panel(pl.DataFrame(_make_qcew()))
    # Mining (21) = 6_000 persons → 6.0 thousand.
    assert _emp(panel, "sector", "21", _JAN) == pytest.approx(6.0)


def test_durable_nondurable_three_digit_sums():
    panel = build_qcew_panel(pl.DataFrame(_make_qcew()))
    # 31 = 321 + 331 = 31.0; 32 = 311 + 322 = 32.0.
    assert _emp(panel, "sector", "31", _JAN) == pytest.approx(31.0)
    assert _emp(panel, "sector", "32", _JAN) == pytest.approx(32.0)


def test_logging_only_for_sector_11():
    panel = build_qcew_panel(pl.DataFrame(_make_qcew()))
    assert _emp(panel, "sector", "11", _JAN) == pytest.approx(5.0)


def test_mining_and_logging_supersector_sums_11_and_21():
    panel = build_qcew_panel(pl.DataFrame(_make_qcew()))
    # 10 = sector 11 (5.0) + sector 21 (6.0) = 11.0 (no agglvl-13 pull).
    assert _emp(panel, "supersector", "10", _JAN) == pytest.approx(11.0)


# ---------------------------------------------------------------------------
# Additive nesting (05 = 06 + 08; domains sum supersectors; supersectors sum sectors)
# ---------------------------------------------------------------------------


def test_additive_nesting_full_tree():
    panel = build_qcew_panel(pl.DataFrame(_make_qcew()))
    e = lambda t, c: _emp(panel, t, c, _JAN)  # noqa: E731

    # supersector 40 == sectors 42+44+48+22
    assert e("supersector", "40") == pytest.approx(
        e("sector", "42") + e("sector", "44") + e("sector", "48") + e("sector", "22")
    )
    # supersector 30 == sectors 31 + 32
    assert e("supersector", "30") == pytest.approx(e("sector", "31") + e("sector", "32"))
    # domain 06 == supersectors 10+20+30
    assert e("domain", "06") == pytest.approx(
        e("supersector", "10") + e("supersector", "20") + e("supersector", "30")
    )
    # domain 08 == the seven private-service supersectors
    assert e("domain", "08") == pytest.approx(
        sum(e("supersector", c) for c in ("40", "50", "55", "60", "65", "70", "80"))
    )
    # total 05 == 06 + 08
    assert e("total", "05") == pytest.approx(e("domain", "06") + e("domain", "08"))


def test_total_private_value():
    panel = build_qcew_panel(pl.DataFrame(_make_qcew()))
    # 05 root carries industry_type='total', ownership='private'.
    row = panel.filter(
        (pl.col("industry_code") == "05") & (pl.col("ref_date") == _JAN)
    )
    assert row["industry_type"].item() == "total"
    assert row["ownership"].item() == "private"
    assert row["employment"].item() == pytest.approx(955.0)


# ---------------------------------------------------------------------------
# Metadata: ownership / bmr / source / SA
# ---------------------------------------------------------------------------


def test_row_metadata():
    panel = build_qcew_panel(pl.DataFrame(_make_qcew()))
    assert (panel["ownership"] == "private").all()
    assert (panel["benchmark_revision"] == 0).all()
    assert (panel["source"] == "qcew").all()
    assert (~panel["seasonally_adjusted"]).all()
    assert (panel["geographic_type"] == "national").all()
    # No government / total-anchor codes leak in.
    assert panel.filter(pl.col("industry_code").is_in(["00", "07", "90"])).height == 0


# ---------------------------------------------------------------------------
# Monthly explode → 3 ref_dates, correct month mapping
# ---------------------------------------------------------------------------


def test_monthly_explode_q2():
    raw = pl.DataFrame([
        {
            "area_fips": "US000", "own_code": "5", "industry_code": "21",
            "agglvl_code": "14", "year": 2024, "qtr": 2,
            "month1_emplvl": 1_000, "month2_emplvl": 2_000, "month3_emplvl": 3_000,
            "revision": 0,
        }
    ])
    panel = build_qcew_panel(raw)
    sec21 = panel.filter(
        (pl.col("industry_type") == "sector") & (pl.col("industry_code") == "21")
    ).sort("ref_date")
    assert sec21["ref_date"].to_list() == [date(2024, 4, 12), date(2024, 5, 12), date(2024, 6, 12)]
    assert sec21["employment"].to_list() == pytest.approx([1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Per-vintage isolation: two revisions never mix; distinct vintage_dates
# ---------------------------------------------------------------------------


def test_per_vintage_isolation():
    rev0 = _make_qcew(year=2024, qtr=1, revision=0, scale=1.0)
    rev1 = _make_qcew(year=2024, qtr=1, revision=1, scale=2.0)
    panel = build_qcew_panel(pl.DataFrame(rev0 + rev1))

    def emp_rev(rev):
        return panel.filter(
            (pl.col("industry_code") == "05")
            & (pl.col("ref_date") == _JAN)
            & (pl.col("revision") == rev)
        )["employment"].item()

    # Each revision is aggregated independently (rev1 doubled), never blended.
    assert emp_rev(0) == pytest.approx(955.0)
    assert emp_rev(1) == pytest.approx(1910.0)

    # vintage_date is the QCEW quarterly publication, distinct per (quarter, rev).
    vd = {
        rev: panel.filter(
            (pl.col("industry_code") == "05") & (pl.col("revision") == rev)
        )["vintage_date"][0]
        for rev in (0, 1)
    }
    assert vd[0] == get_qcew_vintage_date("Q1", 2024, 0)
    assert vd[1] == get_qcew_vintage_date("Q1", 2024, 1)
    assert vd[0] != vd[1]


def test_missing_required_column_raises():
    bad = pl.DataFrame({"area_fips": ["US000"], "own_code": ["5"]})
    with pytest.raises(ValueError, match="missing columns"):
        build_qcew_panel(bad)


# ---------------------------------------------------------------------------
# T2: QCEW own_code=0 total → CES '00' (ownership='total')
# ---------------------------------------------------------------------------


def _total_row(*, year=2024, qtr=1, revision=0, emp=160_000_000):
    """One own_code=0 national all-industries row (agglvl 10, industry '10')."""
    return _row("10", "10", emp, year=year, qtr=qtr, revision=revision, own_code="0")


def test_qcew_total_maps_to_00():
    """own_code=0 (agglvl 10, industry '10') → CES ('total','00', ownership='total').

    Verified primary source 2026-06-17: own_code=0 returns exactly one area row
    (agglvl_code='10', industry_code='10') at US000.  Jan-2024 = 152,393,725 persons
    = 152,393.725 thousand.  Our fixture uses 160,000,000 persons = 160,000.0 thousand.
    """
    raw = pl.DataFrame(_make_qcew() + [_total_row(emp=160_000_000)])
    panel = build_qcew_panel(raw)

    # 3 monthly rows (Q1 → Jan/Feb/Mar 2024, day 12).
    total_rows = panel.filter(
        (pl.col("industry_type") == "total")
        & (pl.col("industry_code") == "00")
        & (pl.col("ownership") == "total")
    )
    assert total_rows.height == 3, (
        f"expected 3 monthly rows for '00' total, got {total_rows.height}"
    )

    # All three months have the same employment value (fixture uses same m for all months).
    jan = date(2024, 1, 12)
    total_jan = panel.filter(
        (pl.col("industry_type") == "total")
        & (pl.col("industry_code") == "00")
        & (pl.col("ownership") == "total")
        & (pl.col("ref_date") == jan)
    )
    assert total_jan.height == 1
    assert total_jan["employment"].item() == pytest.approx(160_000.0)

    # The '00' total is distinct from the private '05' total.
    priv05 = panel.filter(
        (pl.col("industry_code") == "05") & (pl.col("ref_date") == jan)
    )
    assert priv05.height == 1
    assert priv05["ownership"].item() == "private"
    assert total_jan["employment"].item() != pytest.approx(priv05["employment"].item())


def test_qcew_total_private_tree_regression_guard():
    """Private outputs must be byte-identical with vs. without a total row in input.

    This is the size-path robustness guard: build_qcew_panel is called on
    own_code=5-only data in _size_raw_to_native; the total path must be a
    clean no-op in that context, emitting zero '00' rows and never perturbing
    the private sums.
    """
    raw_private_only = pl.DataFrame(_make_qcew())
    raw_with_total = pl.DataFrame(_make_qcew() + [_total_row(emp=160_000_000)])

    panel_private_only = build_qcew_panel(raw_private_only)
    panel_with_total = build_qcew_panel(raw_with_total)

    # Filter to private ownership in both — '00' total row only in panel_with_total.
    priv_only = panel_private_only.filter(pl.col("ownership") == "private")
    priv_with_total = panel_with_total.filter(pl.col("ownership") == "private")

    # Column order and row order must be identical so .equals() is meaningful.
    priv_only_sorted = priv_only.sort("industry_type", "industry_code", "ref_date", "revision")
    priv_with_total_sorted = priv_with_total.sort(
        "industry_type", "industry_code", "ref_date", "revision"
    )

    assert priv_only_sorted.equals(priv_with_total_sorted), (
        "private tree changed when a total row was added to input — "
        "the total path is leaking into the private sums"
    )


def test_qcew_total_absent_when_no_own_code_0_rows():
    """Private-only input (size path) emits zero '00' rows — no error, no '00' leak."""
    raw = pl.DataFrame(_make_qcew())  # own_code='5' only
    panel = build_qcew_panel(raw)

    total_rows = panel.filter(
        (pl.col("industry_type") == "total") & (pl.col("industry_code") == "00")
    )
    assert total_rows.height == 0, (
        f"expected 0 '00' rows for private-only input, got {total_rows.height}"
    )
