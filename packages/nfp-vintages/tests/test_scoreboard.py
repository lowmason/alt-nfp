# packages/nfp-vintages/tests/test_scoreboard.py
from datetime import date

import numpy as np
from nfp_vintages.scoreboard import (
    MonthTypeConfig,
    change_draws_k,
    classify_month_types,
    crps_sample,
    interval_coverage,
    venue_for,
)
from pytest import approx as pytest_approx


def test_classify_month_types_buckets():
    # ref months: 6 normal-ish, one big revision, one Feb (benchmark window),
    # one with a sharp claims jump (turning point).
    ref = [date(2023, m, 1) for m in range(1, 13)]
    # |first->third revision| in thousands; index 6 (July) is the large one.
    revision_abs = np.array([5, 8, 6, 7, 9, 4, 120, 10, 6, 5, 8, 7], dtype=float)
    # claims 3-month momentum (level change); index 9 (Oct) spikes.
    claims_mom = np.array([1, -2, 0, 1, -1, 2, 1, 0, 1, 60, 2, -1], dtype=float)
    cfg = MonthTypeConfig(
        large_revision_pctl=90.0,
        claims_momentum_k=40.0,
        benchmark_months=(2,),
    )

    out = classify_month_types(ref, revision_abs, claims_mom, cfg)

    assert out[date(2023, 7, 1)] == "large_revision"   # 120 > p90
    assert out[date(2023, 2, 1)] == "benchmark_window"  # February
    assert out[date(2023, 10, 1)] == "turning_point"    # claims spike >= 40
    assert out[date(2023, 1, 1)] == "normal"
    # Precedence: a month that is both large-revision and benchmark is labeled
    # by the first matching rule (large_revision wins — it is the rarer signal).
    assert set(out.values()) <= {
        "normal", "large_revision", "turning_point", "benchmark_window"
    }


def test_classify_month_types_precedence_large_over_benchmark():
    ref = [date(2023, 2, 1), date(2023, 3, 1)]
    revision_abs = np.array([200.0, 5.0])
    claims_mom = np.array([0.0, 0.0])
    cfg = MonthTypeConfig(large_revision_pctl=90.0, claims_momentum_k=40.0,
                          benchmark_months=(2,))
    out = classify_month_types(ref, revision_abs, claims_mom, cfg)
    assert out[date(2023, 2, 1)] == "large_revision"


# Task 2: Calibration metrics — coverage and CRPS


def test_interval_coverage_central_interval():
    draws = np.linspace(-100.0, 100.0, 2001)  # symmetric around 0
    assert interval_coverage(draws, actual=0.0, level=0.80) is True
    assert interval_coverage(draws, actual=95.0, level=0.80) is False   # outside p10..p90
    assert interval_coverage(draws, actual=95.0, level=0.95) is True    # inside p2.5..p97.5


def test_crps_point_mass_is_absolute_error():
    # CRPS of a degenerate (point-mass) forecast == |forecast - actual|.
    draws = np.full(500, 10.0)
    assert crps_sample(draws, actual=13.0) == pytest_approx(3.0)


def test_crps_smaller_when_sharper_and_centered():
    actual = 0.0
    sharp = np.random.default_rng(0).normal(0.0, 5.0, 4000)
    wide = np.random.default_rng(0).normal(0.0, 50.0, 4000)
    assert crps_sample(sharp, actual) < crps_sample(wide, actual)


# Task 3: Predictive change-draws extraction + venue tag


def test_change_draws_linearization():
    # tiny growth draws around g; with prev_index and idx_to_level the change
    # draws must match prev_index*(exp(g)-1)*idx_to_level elementwise.
    g = np.array([0.001, 0.002, -0.0005])
    prev_index = 150_000.0
    idx_to_level = 1.0  # 1 index point == 1k jobs in this fixture
    out = change_draws_k(g, prev_index=prev_index, idx_to_level=idx_to_level)
    expected = prev_index * (np.exp(g) - 1.0) * idx_to_level
    assert np.allclose(out, expected)


def test_change_draws_flattens_chains_draws():
    g2d = np.zeros((2, 50))  # (chains, draws)
    out = change_draws_k(g2d, prev_index=150_000.0, idx_to_level=1.0)
    assert out.shape == (100,)


def test_venue_for():
    assert venue_for(providers_present=True) == "full"
    assert venue_for(providers_present=False) == "public-only"
