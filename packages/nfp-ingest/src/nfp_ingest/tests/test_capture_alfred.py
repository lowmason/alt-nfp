import datetime as dt

import polars as pl
from nfp_ingest.capture import capture_ces_alfred_window
from nfp_ingest.vintage_store import append_to_vintage_store, read_vintage_store
from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA


def _store_row(ref, vint, rev, emp, sa=True):
    return {
        "geographic_type": "national",
        "geographic_code": "00",
        "ownership": "total",
        "industry_type": "total",
        "industry_code": "00",
        "ref_date": ref,
        "vintage_date": vint,
        "revision": rev,
        "benchmark_revision": 0,
        "employment": float(emp),
        "size_class_type": None,
        "size_class_code": None,
        "source": "ces",
        "seasonally_adjusted": sa,
    }


def _seed_store(tmp_path):
    # Existing frontier: Jan-2026 rev0 only (vintage 2026-02-11).
    df = pl.DataFrame(
        [_store_row(dt.date(2026, 1, 12), dt.date(2026, 2, 11), 0, 158000)]
    ).cast(VINTAGE_STORE_SCHEMA)
    append_to_vintage_store(df, tmp_path)
    return tmp_path


def _builder_two_new_rows(**_kw):
    # Jan-2026 rev1/rev2 — the missing cohorts.
    return pl.DataFrame(
        [
            _store_row(dt.date(2026, 1, 12), dt.date(2026, 3, 6), 1, 157800),
            _store_row(dt.date(2026, 1, 12), dt.date(2026, 4, 3), 2, 157820),
        ]
    ).cast(VINTAGE_STORE_SCHEMA)


def test_apply_appends_missing_rows(tmp_path):
    _seed_store(tmp_path)
    res = capture_ces_alfred_window(
        through=dt.date(2026, 6, 28),
        store_path=tmp_path,
        api_key="x",
        builder=_builder_two_new_rows,
        calendar=pl.DataFrame(),
    )
    assert res.appended == 2
    stored = read_vintage_store(
        tmp_path, source="ces", seasonally_adjusted=True
    ).collect()
    assert sorted(stored["revision"].to_list()) == [0, 1, 2]


def test_idempotent_rerun_appends_zero(tmp_path):
    _seed_store(tmp_path)
    capture_ces_alfred_window(
        through=dt.date(2026, 6, 28),
        store_path=tmp_path,
        api_key="x",
        builder=_builder_two_new_rows,
        calendar=pl.DataFrame(),
    )
    res2 = capture_ces_alfred_window(
        through=dt.date(2026, 6, 28),
        store_path=tmp_path,
        api_key="x",
        builder=_builder_two_new_rows,
        calendar=pl.DataFrame(),
    )
    assert res2.appended == 0


def test_dry_run_writes_nothing(tmp_path):
    _seed_store(tmp_path)
    res = capture_ces_alfred_window(
        through=dt.date(2026, 6, 28),
        store_path=tmp_path,
        api_key="x",
        dry_run=True,
        builder=_builder_two_new_rows,
        calendar=pl.DataFrame(),
    )
    assert res.appended == 0
    assert res.would_append == 2  # both rev1 + rev2 are new ukeys
    stored = read_vintage_store(
        tmp_path, source="ces", seasonally_adjusted=True
    ).collect()
    assert stored.height == 1  # only the seed row, nothing written
