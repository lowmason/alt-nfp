# packages/nfp-ingest/tests/test_first_print.py
"""A5 first-print extractor: the within-release headline change BLS announces.

Validated against published headlines. Skips when the store is unavailable.
"""
from __future__ import annotations

import numpy as np
import pytest
from nfp_ingest.first_print import first_print_changes
from nfp_lookups.paths import VINTAGE_STORE_PATH


def _store_available() -> bool:
    try:
        return VINTAGE_STORE_PATH.exists() and (
            next(VINTAGE_STORE_PATH.glob("**/*.parquet"), None) is not None
        )
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(
        not _store_available(),
        reason="Vintage store not available",
    ),
    pytest.mark.real_store,  # reads the real vintage store; exempt from cred-blanking
]


def _change_for(df, y: int, m: int) -> float:
    row = df.filter(
        (df["ref_date"].dt.year() == y) & (df["ref_date"].dt.month() == m)
    )
    assert row.height == 1, f"expected one row for {y}-{m:02d}, got {row.height}"
    return float(row["first_print_change_k"][0])


def _change_or_none(df, y: int, m: int) -> float | None:
    """Return the change for a month, or ``None`` if the month is absent or null.

    Shutdown-skipped months are *not scorable*: they may be dropped entirely (no
    valid first print) or present with a null change (no knowable partner).
    """
    row = df.filter(
        (df["ref_date"].dt.year() == y) & (df["ref_date"].dt.month() == m)
    )
    if row.height == 0:
        return None
    assert row.height == 1, f"expected one row for {y}-{m:02d}, got {row.height}"
    v = row["first_print_change_k"][0]
    return None if v is None else float(v)


def test_ordinary_month_headline_2025_07():
    # L(Jul-25 rev0) 159,539 − L(Jun-25 rev1) 159,466 = +73k (published headline).
    # The rebuilt store stamps Jun rev1 at 2025-08-03, +2 days past Jul's first-print
    # vintage 2025-08-01 (schedule-derived stagger) — a day-granular censor drops it.
    df = first_print_changes()
    assert _change_for(df, 2025, 7) == pytest.approx(73.0, abs=1.0)


def test_benchmark_fallback_headline_2026_01():
    # Jan-2026 first print. In the rebuilt store Dec-25 rev1/bmr0 (158,497) exists,
    # so ``primary`` selects it directly; its value coincides with the Dec-25
    # (rev2,bmr1) benchmark level, so +130k holds either way.
    # L(Jan-26 rev0) 158,627 − 158,497 = +130k (published headline).
    df = first_print_changes()
    assert _change_for(df, 2026, 1) == pytest.approx(130.0, abs=1.0)


def test_tolerance_stress_dec_2025():
    # Widest schedule-derived stagger in the store: Nov-25 rev1 is stamped
    # 2026-01-16, +7 days past Dec's first-print vintage 2026-01-09. A too-tight
    # release window drops it and collapses the partner onto Nov rev0 (159,552 →
    # −26k). Convention value rev0(Dec) 159,526 − rev1(Nov) 159,476 = +50k.
    # (Code-derived: this pins the +7d tolerance. The externally-validated
    # headline anchors are 2025-07 and 2026-01.)
    df = first_print_changes()
    assert _change_for(df, 2025, 12) == pytest.approx(50.0, abs=1.0)


def test_shutdown_months_not_scorable():
    # The 2025 government shutdown skipped clean first prints for Oct & Nov 2025;
    # the rebuilt store carries a -1.0 "no print" sentinel for the missing release
    # slots (e.g. Oct-25 rev0). first_print must not emit the ±159,000k garbage that
    # sentinel arithmetic produced — these months are simply not scorable.
    df = first_print_changes()
    assert _change_or_none(df, 2025, 10) is None  # Oct rev0 was the -1 sentinel
    assert _change_or_none(df, 2025, 11) is None  # Nov's Oct partner unknowable at release


def test_no_sentinel_garbage_in_series():
    # Any |change| near the full employment level (~159,000k) is sentinel garbage,
    # not a real move. The COVID extreme (Apr-2020 ≈ −20,500k) is the largest
    # legitimate swing in the 2017+ store, so 100,000k cleanly separates them.
    df = first_print_changes().drop_nulls("first_print_change_k")
    c = df["first_print_change_k"].to_numpy()
    assert np.all(np.abs(c) < 100_000), "sentinel -1 leaked into a first-print change"


def test_growth_and_change_consistent():
    df = first_print_changes().drop_nulls("first_print_change_k")
    # change_k and growth agree in sign
    assert (df["first_print_growth"].is_not_null()).all()
    g = df["first_print_growth"].to_numpy()
    c = df["first_print_change_k"].to_numpy()
    assert np.all(np.sign(g) == np.sign(c))
