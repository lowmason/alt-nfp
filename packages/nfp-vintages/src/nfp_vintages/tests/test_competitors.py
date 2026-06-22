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
    # Mirrors the real file: BOTH Total NFP ('00') and private ('05') rows, each
    # carrying a survey mean and median; ref_date keyed on the CES ref day (12th),
    # no survey_date column (release_date only). industry_code stays a string so
    # the leading-zero codes ('00'/'05') survive the parquet round-trip.
    df = pl.DataFrame(
        {
            "ownership": ["total", "total", "private", "private"],
            "industry_type": ["total", "total", "total", "total"],
            "industry_code": ["00", "00", "05", "05"],
            "ref_date": [date(2024, 5, 12), date(2024, 6, 12),
                         date(2024, 5, 12), date(2024, 6, 12)],
            "release_date": [date(2024, 6, 7), date(2024, 7, 5),
                             date(2024, 6, 7), date(2024, 7, 5)],
            "consensus_mean": [182.0, 191.0, 170.0, 174.0],
            "consensus_median": [180.0, 190.0, 168.0, 172.0],
        },
        schema_overrides={"industry_code": pl.Utf8},
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
    assert df.height == 4
    assert {"ownership", "industry_type", "industry_code", "ref_date",
            "release_date", "consensus_mean", "consensus_median"}.issubset(df.columns)
    assert df["industry_code"].dtype == pl.Utf8  # leading zeros preserved


def test_consensus_competitor_t1_lookup_total(tmp_path):
    from nfp_vintages.competitors.consensus import Consensus, load_consensus

    # Default series = Total NFP ('00') median (what Track B scores).
    c = Consensus(load_consensus(_consensus_file(tmp_path)))
    # at T-1 (release_date - 1 = 2024-07-04) consensus for 2024-06 is known
    assert c.predict(date(2024, 6, 1), as_of=date(2024, 7, 4)) == pytest.approx(190.0)


def test_consensus_selects_private_series(tmp_path):
    from nfp_vintages.competitors.consensus import Consensus, load_consensus

    # The file also carries the private ('05') consensus — selectable.
    c = Consensus(load_consensus(_consensus_file(tmp_path)), industry_code="05")
    assert c.predict(date(2024, 6, 1), as_of=date(2024, 7, 4)) == pytest.approx(172.0)


def test_consensus_selects_mean_statistic(tmp_path):
    from nfp_vintages.competitors.consensus import Consensus, load_consensus

    # mean is selectable alongside the default median.
    c = Consensus(load_consensus(_consensus_file(tmp_path)), statistic="mean")
    assert c.predict(date(2024, 6, 1), as_of=date(2024, 7, 4)) == pytest.approx(191.0)


def test_consensus_lookup_is_month_keyed(tmp_path):
    # The backtest harness keys targets on the model date axis (the CES ref day,
    # the 12th) and normalizes to month-start (day=1); the file's ref_date is the
    # 12th. predict must month-bucket BOTH sides so the lookup hits — else
    # consensus silently scores None for every month.
    from nfp_vintages.competitors.consensus import Consensus, load_consensus

    c = Consensus(load_consensus(_consensus_file(tmp_path)))
    assert c.predict(date(2024, 6, 12), as_of=date(2024, 7, 4)) == pytest.approx(190.0)


def test_consensus_withheld_before_release_eve(tmp_path):
    # No survey_date in the file: the value locks at release-eve (release_date - 1).
    # A too-early as_of must withhold it (no lookahead).
    from nfp_vintages.competitors.consensus import Consensus, load_consensus

    c = Consensus(load_consensus(_consensus_file(tmp_path)))
    assert c.predict(date(2024, 6, 1), as_of=date(2024, 6, 30)) is None


def test_consensus_none_when_unconfigured():
    from nfp_vintages.competitors.consensus import Consensus

    c = Consensus(None)
    assert c.predict(date(2024, 6, 1), as_of=date(2024, 7, 4)) is None


def test_consensus_path_preserves_s3_uri(monkeypatch):
    monkeypatch.setenv("NFP_CONSENSUS_PATH", "s3://alt-nfp/competitors/consensus.parquet")
    from importlib import reload

    import nfp_vintages.competitors.consensus as m

    reload(m)
    assert str(m.consensus_path()).startswith("s3://alt-nfp/competitors/consensus.parquet")


def test_consensus_path_local_default(monkeypatch):
    monkeypatch.delenv("NFP_CONSENSUS_PATH", raising=False)
    from importlib import reload

    import nfp_vintages.competitors.consensus as m

    reload(m)
    p = m.consensus_path()
    assert str(p).endswith("competitors/consensus.parquet")
