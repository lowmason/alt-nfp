"""A5 helpers: near-release as-of dates + scoring math."""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from nfp_vintages.a5 import first_friday_release, near_release_asof, score


def test_first_friday_release_basic():
    # June-2025 reference -> first Friday of July 2025 = 2025-07-04 -> shifts +7
    # (Independence Day) -> 2025-07-11
    assert first_friday_release(date(2025, 6, 1)) == date(2025, 7, 11)


def test_first_friday_release_ordinary():
    # May-2025 reference -> first Friday of June 2025 = 2025-06-06
    assert first_friday_release(date(2025, 5, 1)) == date(2025, 6, 6)


def test_near_release_asof_offsets():
    rel = date(2025, 6, 6)
    assert near_release_asof(date(2025, 5, 1), days_before=1, release=rel) == date(2025, 6, 5)
    assert near_release_asof(date(2025, 5, 1), days_before=7, release=rel) == date(2025, 5, 30)


def test_score_metrics():
    errors = np.array([10.0, -20.0, 30.0])
    m = score(errors)
    assert m["me"] == pytest.approx(20.0 / 3)
    assert m["mae"] == pytest.approx(20.0)
    assert m["rmse"] == pytest.approx(np.sqrt((100 + 400 + 900) / 3))
    assert m["n"] == 3


def test_score_empty():
    m = score(np.array([]))
    assert m["n"] == 0
    assert m["mae"] is None
