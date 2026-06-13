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


pytestmark = pytest.mark.skipif(
    not _store_available(),
    reason="Vintage store not available",
)


def _change_for(df, y: int, m: int) -> float:
    row = df.filter(
        (df["ref_date"].dt.year() == y) & (df["ref_date"].dt.month() == m)
    )
    assert row.height == 1, f"expected one row for {y}-{m:02d}, got {row.height}"
    return float(row["first_print_change_k"][0])


def test_ordinary_month_headline_2025_07():
    # L(Jul-25 rev0) 159,539 − L(Jun-25 rev1) 159,466 = +73k (published headline)
    df = first_print_changes()
    assert _change_for(df, 2025, 7) == pytest.approx(73.0, abs=1.0)


def test_benchmark_fallback_headline_2026_01():
    # Dec-25 rev1 was shadowed; fall back to Dec-25 (rev2,bmr1) 158,497.
    # L(Jan-26 rev0) 158,627 − 158,497 = +130k (published headline)
    df = first_print_changes()
    assert _change_for(df, 2026, 1) == pytest.approx(130.0, abs=1.0)


def test_growth_and_change_consistent():
    df = first_print_changes().drop_nulls("first_print_change_k")
    # change_k and growth agree in sign
    assert (df["first_print_growth"].is_not_null()).all()
    g = df["first_print_growth"].to_numpy()
    c = df["first_print_change_k"].to_numpy()
    assert np.all(np.sign(g) == np.sign(c))
