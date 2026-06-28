import datetime as dt

import polars as pl
from nfp_ingest.ces_alfred import build_ces_alfred_window, extract_prints
from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA


def _row(ref, vint, val):
    return {"ref_date": dt.date(*ref), "vintage_date": dt.date(*vint), "value": float(val)}


def test_first_three_appearances_are_rev_0_1_2():
    m = pl.DataFrame([
        _row((2026, 2, 1), (2026, 3, 6), 159000),   # rev0
        _row((2026, 2, 1), (2026, 4, 3), 159050),   # rev1
        _row((2026, 2, 1), (2026, 5, 8), 159040),   # rev2
        _row((2026, 2, 1), (2026, 6, 5), 159040),   # 4th appearance -> dropped
    ])
    out = extract_prints(m).sort("revision")
    assert out["revision"].to_list() == [0, 1, 2]
    assert out["value"].to_list() == [159000.0, 159050.0, 159040.0]


def test_no_value_dedup_keeps_unchanged_revision():
    # rev1 == rev0 value (rounding); positional rule must still emit it as rev1.
    m = pl.DataFrame([
        _row((2026, 2, 1), (2026, 3, 6), 159000),
        _row((2026, 2, 1), (2026, 4, 3), 159000),  # same value, real revision
        _row((2026, 2, 1), (2026, 5, 8), 159010),
    ])
    out = extract_prints(m).sort("revision")
    assert out["revision"].to_list() == [0, 1, 2]
    assert out.filter(pl.col("revision") == 1)["vintage_date"].item() == dt.date(2026, 4, 3)


def test_real_time_guard_drops_back_history_artifact():
    # ref 2003-01 first appears only in a 2011 vintage (gap ~8y) -> artifact, dropped.
    artifact = pl.DataFrame([
        _row((2003, 1, 1), (2011, 3, 4), 130000),
        _row((2003, 1, 1), (2011, 4, 1), 130100),
        _row((2003, 1, 1), (2011, 5, 6), 130200),
    ])
    assert extract_prints(artifact).height == 0


def test_frontier_month_with_only_rev0_kept():
    m = pl.DataFrame([_row((2026, 5, 1), (2026, 6, 5), 160000)])
    out = extract_prints(m)
    assert out["revision"].to_list() == [0]


def _calendar():
    # ces, bmr=0; ref_date day-12; only Feb-2026 rev0/1/2 in-window.
    return pl.DataFrame({
        "publication": ["ces"] * 3,
        "ref_date": [dt.date(2026, 2, 12)] * 3,
        "revision": [0, 1, 2],
        "benchmark_revision": [0, 0, 0],
        "vintage_date": [dt.date(2026, 3, 6), dt.date(2026, 4, 3), dt.date(2026, 5, 8)],
    })


def _stub_fetch(series_id, *, sa, key):
    # Returns (title_ok, matrix) for one series; Feb-2026 three genuine prints.
    matrix = pl.DataFrame({
        "ref_date": [dt.date(2026, 2, 1)] * 3,
        "vintage_date": [dt.date(2026, 3, 6), dt.date(2026, 4, 3), dt.date(2026, 5, 8)],
        "value": [159000.0, 159050.0, 159040.0],
    })
    return matrix


def test_build_window_shapes_store_rows_for_one_key():
    rows = build_ces_alfred_window(
        store_frontier=dt.date(2026, 2, 11),
        through=dt.date(2026, 6, 28),
        calendar=_calendar(),
        api_key="x",
        adjustments=(True,),
        keys=[("total", "00")],
        fetch=_stub_fetch,
    )
    assert list(rows.columns) == list(VINTAGE_STORE_SCHEMA)
    assert rows.height == 3  # rev 0/1/2 for Feb-2026
    r = rows.sort("revision")
    assert r["revision"].to_list() == [0, 1, 2]
    assert r["ref_date"].unique().to_list() == [dt.date(2026, 2, 12)]   # calendar day-12
    assert r["vintage_date"].to_list() == [dt.date(2026, 3, 6), dt.date(2026, 4, 3), dt.date(2026, 5, 8)]
    assert r["employment"].to_list() == [159000.0, 159050.0, 159040.0]
    assert r["ownership"].unique().to_list() == ["total"]
    assert r["source"].unique().to_list() == ["ces"]
    assert r["seasonally_adjusted"].unique().to_list() == [True]


def test_build_window_excludes_cohorts_at_or_before_frontier():
    # frontier AFTER all window vintages -> nothing to add.
    rows = build_ces_alfred_window(
        store_frontier=dt.date(2026, 5, 8),
        through=dt.date(2026, 6, 28),
        calendar=_calendar(),
        api_key="x",
        adjustments=(True,),
        keys=[("total", "00")],
        fetch=_stub_fetch,
    )
    assert rows.height == 0
