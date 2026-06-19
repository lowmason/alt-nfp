# packages/nfp-vintages/tests/test_scoreboard.py
from datetime import date

import numpy as np
from nfp_vintages.scoreboard import MonthTypeConfig, classify_month_types


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
