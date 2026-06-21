"""CLI tests for `alt-nfp watch` (feed-driven trigger).

Monkeypatches feed.fetch_feed and the _run_update/_run_snapshot helpers; lets
compute_status run for real against a tmp store so the present/absent ref-month
decides trigger-vs-no-op. Store-write-free and hermetic: the store is passed
EXPLICITLY via `--store <tmp>` (compute_status's default VINTAGE_STORE_PATH binds
at import to canonical MinIO under .env, so an env swap cannot redirect it — the
explicit --store is what makes the test read the seeded tmp store). NFP_STORE_URI
is also cleared as belt-and-suspenders so nothing can reach MinIO.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from nfp_download.release_dates.feed import FeedItem
from typer.testing import CliRunner

runner = CliRunner()


def _seed_store(store_root, *, ref_dates, vintage_date):
    """Write a minimal CES SA partition with the given headline ref_dates."""
    part = store_root / "source=ces" / "seasonally_adjusted=true"
    part.mkdir(parents=True, exist_ok=True)
    n = len(ref_dates)
    rows = {
        "ref_date": list(ref_dates),
        "industry_type": ["total"] * n,
        "industry_code": ["00"] * n,
        "ownership": ["total"] * n,
        "size_class_type": [None] * n,
        "size_class_code": [None] * n,
        "geographic_type": ["national"] * n,
        "geographic_code": ["00"] * n,
        "revision": [0] * n,
        "benchmark_revision": [0] * n,
        "vintage_date": [vintage_date] * n,
        "employment": [150_000.0 + i for i in range(n)],
    }
    pl.DataFrame(rows).write_parquet(str(part / "part-0.parquet"))


def _seed_qcew_store(store_root, *, ref_dates, vintage_date):
    """Write a minimal QCEW NSA partition with the given ref_dates (sets latest_ref)."""
    part = store_root / "source=qcew" / "seasonally_adjusted=false"
    part.mkdir(parents=True, exist_ok=True)
    n = len(ref_dates)
    rows = {
        "ref_date": list(ref_dates),
        "industry_type": ["total"] * n,
        "industry_code": ["05"] * n,
        "ownership": ["private"] * n,
        "size_class_type": [None] * n,
        "size_class_code": [None] * n,
        "geographic_type": ["national"] * n,
        "geographic_code": ["00"] * n,
        "revision": [0] * n,
        "benchmark_revision": [0] * n,
        "vintage_date": [vintage_date] * n,
        "employment": [130_000.0 + i for i in range(n)],
    }
    pl.DataFrame(rows).write_parquet(str(part / "part-0.parquet"))


@pytest.fixture
def watch_store(tmp_path, monkeypatch):
    """A local tmp store for watch; NFP_STORE_URI cleared so nothing reaches MinIO."""
    monkeypatch.delenv("NFP_STORE_URI", raising=False)
    return tmp_path / "store"


def _patch_feed(monkeypatch, pub_date: date):
    """Make fetch_feed return one empsit item published on ``pub_date``."""
    item = FeedItem(
        title="Employment Situation Summary",
        pub_date=pub_date,
        guid=f"empsit_{pub_date.isoformat()}",
    )
    import nfp_download.release_dates.feed as feed_mod

    monkeypatch.setattr(feed_mod, "fetch_feed", lambda url, **kw: [item])


def test_triggers_update_when_refmonth_uncaptured(watch_store, monkeypatch):
    """A feed release whose ref-month is NOT in the store triggers update."""
    store_root = watch_store
    # Store has CES through 2025-04; the 2025-05 print is published 2025-06-06.
    _seed_store(
        store_root,
        ref_dates=[date(2025, 3, 1), date(2025, 4, 1)],
        vintage_date=date(2025, 5, 2),
    )
    _patch_feed(monkeypatch, date(2025, 6, 6))

    calls = []
    import nfp_vintages.__main__ as cli

    monkeypatch.setattr(cli, "_run_update", lambda as_of, **kw: calls.append({"as_of": as_of, **kw}))
    monkeypatch.setattr(cli, "_run_snapshot", lambda as_of, **kw: calls.append(("snap", as_of, kw)))

    result = runner.invoke(
        cli.app, ["watch", "--source", "ces", "--store", str(store_root)]
    )
    assert result.exit_code == 0, result.output
    update_calls = [c for c in calls if isinstance(c, dict)]
    assert len(update_calls) == 1
    # as_of is a date object, not a string (contract: _run_update(as_of: date, ...))
    assert update_calls[0]["as_of"] == date(2025, 6, 6)
    assert update_calls[0]["only"] == "ces"


def test_no_op_when_refmonth_already_present(watch_store, monkeypatch):
    """A feed release whose ref-month IS captured triggers nothing."""
    store_root = watch_store
    _seed_store(
        store_root,
        ref_dates=[date(2025, 4, 1), date(2025, 5, 1)],
        vintage_date=date(2025, 6, 6),
    )
    _patch_feed(monkeypatch, date(2025, 6, 6))

    calls = []
    import nfp_vintages.__main__ as cli

    monkeypatch.setattr(cli, "_run_update", lambda as_of, **kw: calls.append({"as_of": as_of, **kw}))
    monkeypatch.setattr(cli, "_run_snapshot", lambda as_of, **kw: calls.append(("snap", as_of, kw)))

    result = runner.invoke(
        cli.app, ["watch", "--source", "ces", "--store", str(store_root)]
    )
    assert result.exit_code == 0, result.output
    assert calls == []  # nothing uncaptured → clean no-op
    assert "already captured" in result.output
    # Strong non-vacuity: an empty/unreadable store would ALSO yield calls==[] and
    # the same echo (compute_status returns ces_sa=None → uncaptured=[]). Assert the
    # actual precondition — the seed wrote a readable partition AND the latest month
    # (2025-05) is present so nothing is genuinely uncaptured. This also catches the
    # "store read but a row dropped" anomaly: a missing 2025-05 would make uncaptured
    # non-empty here and fail loudly instead of silently flipping the no-op.
    from nfp_vintages.store_status import compute_status

    st = compute_status(store_root, as_of=date(2025, 6, 6))
    assert st.per_partition  # the store WAS read (not empty/unreadable)
    assert st.uncaptured == []  # 2025-05 present → nothing uncaptured (the real reason)


def test_snapshot_uses_day12_anchor_not_pubdate(watch_store, monkeypatch):
    """With --snapshot, snapshot as-of is date(refmonth.year, refmonth.month, 12), not pubDate."""
    store_root = watch_store
    _seed_store(
        store_root,
        ref_dates=[date(2025, 3, 1), date(2025, 4, 1)],
        vintage_date=date(2025, 5, 2),
    )
    _patch_feed(monkeypatch, date(2025, 6, 6))

    snaps = []
    import nfp_vintages.__main__ as cli

    monkeypatch.setattr(cli, "_run_update", lambda as_of, **kw: None)
    monkeypatch.setattr(cli, "_run_snapshot", lambda as_of, **kw: snaps.append(as_of))

    result = runner.invoke(
        cli.app, ["watch", "--source", "ces", "--snapshot", "--store", str(store_root)]
    )
    assert result.exit_code == 0, result.output
    assert len(snaps) == 1
    # Captured ref-month is 2025-05 → anchor date(2025, 5, 12), NOT pubDate date(2025, 6, 6).
    assert snaps[0] == date(2025, 5, 12)


def test_qcew_source_triggers_update_and_quarter_day12_snapshot(watch_store, monkeypatch):
    """--source qcew drives the QCEW leg: only='qcew', snapshot at the quarter day-12 anchor.

    The three baseline watch tests are all CES; this exercises the QCEW token
    (``qcew:YYYY-Qn``) → ``_watch_snapshot_anchor`` ``-Q`` branch → ``date(year, q*3, 12)``.
    """
    store_root = watch_store
    # QCEW captured through 2024-Q4 (ref 2024-10-01); as-of mid-Sep-2025 makes the
    # 2025 quarters knowable, so the oldest uncaptured is 2025-Q1.
    _seed_qcew_store(
        store_root,
        ref_dates=[date(2024, 7, 1), date(2024, 10, 1)],
        vintage_date=date(2025, 6, 4),
    )
    _patch_feed(monkeypatch, date(2025, 9, 10))

    updates, snaps = [], []
    import nfp_vintages.__main__ as cli

    monkeypatch.setattr(cli, "_run_update", lambda as_of, **kw: updates.append({"as_of": as_of, **kw}))
    monkeypatch.setattr(cli, "_run_snapshot", lambda as_of, **kw: snaps.append(as_of))

    result = runner.invoke(
        cli.app, ["watch", "--source", "qcew", "--snapshot", "--store", str(store_root)]
    )
    assert result.exit_code == 0, result.output
    assert len(updates) == 1
    assert updates[0]["only"] == "qcew"
    assert updates[0]["as_of"] == date(2025, 9, 10)  # the feed pubDate drives update --as-of
    # uncaptured[0] = "qcew:2025-Q1" → day-12 of the quarter's closing month (Mar).
    assert snaps == [date(2025, 3, 12)]


def test_source_all_triggers_both_legs_on_co_release(watch_store, monkeypatch):
    """--source all fires BOTH the CES and QCEW legs on a same-day co-release (§9)."""
    store_root = watch_store
    # CES lagging (through 2025-04) and QCEW lagging (through 2024-Q4) → both uncaptured.
    _seed_store(
        store_root, ref_dates=[date(2025, 3, 1), date(2025, 4, 1)], vintage_date=date(2025, 5, 2)
    )
    _seed_qcew_store(
        store_root, ref_dates=[date(2024, 7, 1), date(2024, 10, 1)], vintage_date=date(2025, 6, 4)
    )
    _patch_feed(monkeypatch, date(2025, 9, 10))

    onlys = []
    import nfp_vintages.__main__ as cli

    monkeypatch.setattr(cli, "_run_update", lambda as_of, **kw: onlys.append(kw.get("only")))
    monkeypatch.setattr(cli, "_run_snapshot", lambda as_of, **kw: None)

    result = runner.invoke(cli.app, ["watch", "--source", "all", "--store", str(store_root)])
    assert result.exit_code == 0, result.output
    assert sorted(onlys) == ["ces", "qcew"]  # both legs triggered in one run


def test_empty_feed_is_a_clean_no_op(watch_store, monkeypatch):
    """An empty feed (no items) skips the source — no update, clean exit."""
    store_root = watch_store
    _seed_store(
        store_root, ref_dates=[date(2025, 3, 1), date(2025, 4, 1)], vintage_date=date(2025, 5, 2)
    )
    import nfp_download.release_dates.feed as feed_mod

    monkeypatch.setattr(feed_mod, "fetch_feed", lambda url, **kw: [])

    calls = []
    import nfp_vintages.__main__ as cli

    monkeypatch.setattr(cli, "_run_update", lambda as_of, **kw: calls.append(kw))
    monkeypatch.setattr(cli, "_run_snapshot", lambda as_of, **kw: calls.append(kw))

    result = runner.invoke(cli.app, ["watch", "--source", "ces", "--store", str(store_root)])
    assert result.exit_code == 0, result.output
    assert calls == []
    assert "feed empty" in result.output
