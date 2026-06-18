"""A5 competitor adapters — naive floors + consensus (synthetic, no store)."""
from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from nfp_vintages.competitors.naive import RandomWalk, TrailingMean


def _history() -> pl.DataFrame:
    # months 2024-01..2024-06, change_k = 100..150, released the next month's 5th
    rows = []
    for i, m in enumerate(range(1, 7)):
        rows.append(
            {
                "ref_date": date(2024, m, 1),
                "first_print_change_k": 100.0 + 10 * i,
                "vintage_date": date(2024, m + 1, 5),
            }
        )
    return pl.DataFrame(rows)


def test_random_walk_repeats_last_published_print():
    hist = _history()
    rw = RandomWalk(hist)
    # nowcasting 2024-07 as of 2024-07-31: last published is 2024-06 (=150)
    assert rw.predict(date(2024, 7, 1), as_of=date(2024, 7, 31)) == pytest.approx(150.0)


def test_random_walk_respects_as_of_censoring():
    hist = _history()
    rw = RandomWalk(hist)
    # as of 2024-06-10: 2024-05's print (released 2024-06-05) is the latest known
    assert rw.predict(date(2024, 6, 1), as_of=date(2024, 6, 10)) == pytest.approx(140.0)


def test_trailing_mean_of_known_prints():
    hist = _history()
    tm = TrailingMean(hist, window=3)
    # as of 2024-07-31, last 3 known prints = 130,140,150 -> mean 140
    assert tm.predict(date(2024, 7, 1), as_of=date(2024, 7, 31)) == pytest.approx(140.0)


def test_predict_none_when_no_history_available():
    hist = _history()
    rw = RandomWalk(hist)
    assert rw.predict(date(2024, 1, 1), as_of=date(2024, 1, 1)) is None


def _consensus_file(tmp_path):
    df = pl.DataFrame(
        {
            "ref_month": [date(2024, 5, 1), date(2024, 6, 1)],
            "consensus_median_change_k": [180.0, 190.0],
            "survey_date": [date(2024, 6, 5), date(2024, 7, 3)],
            "release_date": [date(2024, 6, 7), date(2024, 7, 5)],
            "source": ["bloomberg", "bloomberg"],
        }
    )
    p = tmp_path / "consensus.parquet"
    df.write_parquet(p)
    return p


def test_load_consensus_absent_returns_none(tmp_path):
    from nfp_vintages.competitors.consensus import load_consensus

    assert load_consensus(tmp_path / "missing.parquet") is None


def test_load_consensus_validates_and_reads(tmp_path):
    from nfp_vintages.competitors.consensus import load_consensus

    df = load_consensus(_consensus_file(tmp_path))
    assert df is not None
    assert df.height == 2
    assert {"ref_month", "consensus_median_change_k", "survey_date",
            "release_date", "source"}.issubset(df.columns)


def test_consensus_competitor_t1_lookup(tmp_path):
    from nfp_vintages.competitors.consensus import Consensus, load_consensus

    c = Consensus(load_consensus(_consensus_file(tmp_path)))
    # at T-1 (release_date - 1 = 2024-07-04) consensus for 2024-06 is known
    assert c.predict(date(2024, 6, 1), as_of=date(2024, 7, 4)) == pytest.approx(190.0)


def test_consensus_lookup_is_month_keyed(tmp_path):
    # The backtest harness keys targets on the model date axis (the CES ref day,
    # the 12th), but the consensus file's ref_month is month-start (day=1, per
    # specs/bloomberg_consensus.md). predict must month-bucket so the lookup hits
    # — else consensus silently scores None for every month once the file lands.
    from nfp_vintages.competitors.consensus import Consensus, load_consensus

    c = Consensus(load_consensus(_consensus_file(tmp_path)))
    assert c.predict(date(2024, 6, 12), as_of=date(2024, 7, 4)) == pytest.approx(190.0)


def test_consensus_none_when_unconfigured():
    from nfp_vintages.competitors.consensus import Consensus

    c = Consensus(None)
    assert c.predict(date(2024, 6, 1), as_of=date(2024, 7, 4)) is None
