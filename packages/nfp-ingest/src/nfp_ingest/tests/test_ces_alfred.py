import datetime as dt
import polars as pl
from nfp_ingest.ces_alfred import extract_prints


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
