"""Tests for H-4a: all-zero / missing cyclical indicator warning.

When a CONFIGURED cyclical indicator loads as None (file missing/unreadable)
or as an all-zero array (constant raw data with std==0), the model's phi_3
block silently drops it.  A UserWarning is emitted so the footgun is loud.

The warning is added in _load_cyclical_indicators, BEFORE the censoring loop
in panel_to_model_data sets the tail to 0.0.  This means:
- None or all-zero at detection time => genuinely missing/degenerate (warn)
- Non-zero at detection time => real data, censored tail added later (no warn)

The condition is: val is None or not np.any(val)
Not just np.all(val == 0.0), because missing files degrade to None, not
an all-zero array.  That is the actual code path in _load_cyclical_indicators.
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from nfp_ingest.model_data import _load_cyclical_indicators
from nfp_lookups.provider_config import CyclicalIndicator

# ---------------------------------------------------------------------------
# Fixtures: CyclicalIndicator specs for testing
# ---------------------------------------------------------------------------

MONTHLY_SPEC = CyclicalIndicator(name="test_ind", fred_id="FAKE_ID", freq="monthly", pub_lag=1)
WEEKLY_SPEC = CyclicalIndicator(name="test_weekly", fred_id="FAKE_WEEKLY", freq="weekly", pub_lag=1)

# Model dates: 12 months of 2023
MODEL_DATES = [date(2023, m, 1) for m in range(1, 13)]
T = len(MODEL_DATES)


def _write_monthly_parquet(tmp_path, name: str, values: list[float]) -> None:
    """Write a monthly parquet for the given indicator name into tmp_path."""
    assert len(values) == len(MODEL_DATES), "values must match MODEL_DATES length"
    df = pl.DataFrame(
        {
            "ref_date": MODEL_DATES,
            "value": values,
        }
    )
    df.write_parquet(tmp_path / f"{name}.parquet")


def _write_weekly_parquet(tmp_path, name: str, values_by_month: list[float]) -> None:
    """Write a weekly parquet (4 obs/month) for the given indicator name."""
    rows = []
    for d, val in zip(MODEL_DATES, values_by_month, strict=True):
        for week in range(4):
            rows.append({"ref_date": d + timedelta(weeks=week), "value": val + week * 0.01})
    df = pl.DataFrame(rows).with_columns(pl.col("ref_date").cast(pl.Date))
    df.write_parquet(tmp_path / f"{name}.parquet")


# ---------------------------------------------------------------------------
# Positive tests: warning fires when indicator is missing or all-zero
# ---------------------------------------------------------------------------


class TestAllZeroIndicatorWarning:
    """Warning emitted for missing or degenerate configured indicators."""

    def test_warning_fires_when_file_missing(self, tmp_path):
        """Empty indicators dir: all configured indicators yield None -> warn for each."""
        # No parquets written; tmp_path is empty.
        with pytest.warns(UserWarning, match="test_ind"):
            _load_cyclical_indicators(MODEL_DATES, T, [MONTHLY_SPEC], tmp_path)

    def test_warning_message_names_indicator(self, tmp_path):
        """Warning text contains the indicator name and phi_3."""
        with pytest.warns(UserWarning) as record:
            _load_cyclical_indicators(MODEL_DATES, T, [MONTHLY_SPEC], tmp_path)

        messages = [str(w.message) for w in record]
        assert any("test_ind" in m for m in messages), f"indicator name missing from: {messages}"
        assert any("phi_3" in m for m in messages), f"phi_3 missing from: {messages}"

    def test_warning_fires_for_constant_data(self, tmp_path):
        """Constant raw values (std==0) center to all-zero -> warn."""
        # All same value: after centering arr - mean => 0.0 everywhere.
        _write_monthly_parquet(tmp_path, "test_ind", [5.0] * T)

        with pytest.warns(UserWarning, match="test_ind"):
            result = _load_cyclical_indicators(MODEL_DATES, T, [MONTHLY_SPEC], tmp_path)

        # Value should be all-zero (not None) — centering of constant series
        arr = result["test_ind_c"]
        assert arr is not None, "constant series should produce all-zero array, not None"
        assert not np.any(arr), "constant series should be all-zero after centering"

    def test_warning_fires_for_each_missing_indicator(self, tmp_path):
        """Multiple missing indicators each generate their own warning."""
        specs = [
            CyclicalIndicator(name="alpha", fred_id="A", freq="monthly", pub_lag=1),
            CyclicalIndicator(name="beta", fred_id="B", freq="monthly", pub_lag=1),
        ]
        with pytest.warns(UserWarning) as record:
            _load_cyclical_indicators(MODEL_DATES, T, specs, tmp_path)

        messages = [str(w.message) for w in record]
        assert any("alpha" in m for m in messages), "alpha warning missing"
        assert any("beta" in m for m in messages), "beta warning missing"

    def test_missing_file_result_is_none(self, tmp_path):
        """Missing file path degrades to None in result dict (not all-zero array)."""
        with pytest.warns(UserWarning):
            result = _load_cyclical_indicators(MODEL_DATES, T, [MONTHLY_SPEC], tmp_path)

        assert result["test_ind_c"] is None, "missing indicator should be None, not an array"


# ---------------------------------------------------------------------------
# Negative tests: no warning for real non-zero data
# ---------------------------------------------------------------------------


class TestNoWarningForRealData:
    """No warning emitted when indicator loads with non-zero variation."""

    def test_no_warning_for_valid_monthly_indicator(self, tmp_path):
        """Valid monthly parquet with varying values should NOT warn."""
        values = [float(i) for i in range(T)]  # 0..11, std > 0
        _write_monthly_parquet(tmp_path, "test_ind", values)

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = _load_cyclical_indicators(MODEL_DATES, T, [MONTHLY_SPEC], tmp_path)

        arr = result["test_ind_c"]
        assert arr is not None
        assert np.any(arr), "centered values should be non-zero"

    def test_no_warning_for_valid_weekly_indicator(self, tmp_path):
        """Valid weekly parquet aggregated to monthly should NOT warn."""
        values = [float(i) for i in range(T)]  # 0..11 monthly means
        _write_weekly_parquet(tmp_path, "test_weekly", values)

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = _load_cyclical_indicators(MODEL_DATES, T, [WEEKLY_SPEC], tmp_path)

        arr = result["test_weekly_c"]
        assert arr is not None
        assert np.any(arr), "centered values from weekly should be non-zero"

    def test_no_warning_when_only_tail_censored(self, tmp_path):
        """Presence of 0.0 in tail (censoring) should not warn -- test detection is pre-censoring."""
        # This test documents the detection point: _load_cyclical_indicators
        # is called BEFORE the censoring loop.  At detection time, the array
        # has real non-zero values; 0.0 tail injection happens afterward.
        # We simulate by checking the loaded array is non-zero (no warn), then
        # manually zeroing the tail and confirming np.any() would be False only
        # for a fully-zeroed array, not a partially-zeroed one.
        values = [float(i + 1) for i in range(T)]  # 1..12, all non-zero
        _write_monthly_parquet(tmp_path, "test_ind", values)

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = _load_cyclical_indicators(MODEL_DATES, T, [MONTHLY_SPEC], tmp_path)

        arr = result["test_ind_c"]
        assert arr is not None
        # Simulate what censoring does: zero the last 3 months
        arr[-3:] = 0.0
        # With partial zeroing np.any(arr) is still True -> no warning condition
        assert np.any(arr), "partial zero (censored tail) should not trigger warning condition"

    def test_no_warning_for_unconfigured_indicator(self, tmp_path):
        """Indicator not in the configured list should not warn even if missing."""
        # Only MONTHLY_SPEC is configured; "other_ind" is not in the list.
        # No parquet for "test_ind" exists, so test_ind warns.
        # But if we pass an empty list, no warning fires at all.
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = _load_cyclical_indicators(MODEL_DATES, T, [], tmp_path)

        assert result == {}, "empty indicators list should produce empty result"
