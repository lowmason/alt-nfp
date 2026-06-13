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
