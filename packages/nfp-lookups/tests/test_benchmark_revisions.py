"""Tests for nfp_lookups.benchmark_revisions."""

from __future__ import annotations

import pytest
from nfp_lookups import BENCHMARK_REVISIONS, get_benchmark_revision


class TestBenchmarkRevisions:
    """Tests for the historical benchmark revision lookup table."""

    def test_all_years_present(self):
        """All years 2009–2025 are present in the dict."""
        expected = set(range(2009, 2026))
        assert set(BENCHMARK_REVISIONS.keys()) == expected

    def test_covid_years_are_none(self):
        """COVID years (2020, 2021) have None values."""
        assert BENCHMARK_REVISIONS[2020] is None
        assert BENCHMARK_REVISIONS[2021] is None

    def test_non_covid_years_are_float(self):
        """All non-COVID years have numeric (float/int) values."""
        for year, val in BENCHMARK_REVISIONS.items():
            if year in (2020, 2021):
                continue
            assert isinstance(
                val, (int, float)
            ), f"Year {year} has value {val!r} of type {type(val).__name__}"

    def test_known_values(self):
        """Spot-check a few known benchmark revisions."""
        assert BENCHMARK_REVISIONS[2009] == -902
        assert BENCHMARK_REVISIONS[2025] == -862
        assert BENCHMARK_REVISIONS[2022] == 462


class TestGetBenchmarkRevision:
    """Tests for the get_benchmark_revision() lookup helper."""

    def test_valid_year(self):
        """Returns the correct revision for a valid year."""
        assert get_benchmark_revision(2024) == -598

    def test_covid_year_returns_none(self):
        """COVID years return None (not an error)."""
        assert get_benchmark_revision(2020) is None

    def test_invalid_year_raises_keyerror(self):
        """Years outside the table raise KeyError."""
        with pytest.raises(KeyError, match="2000"):
            get_benchmark_revision(2000)

    def test_future_year_raises_keyerror(self):
        """Future years not yet in the table raise KeyError."""
        with pytest.raises(KeyError, match="2030"):
            get_benchmark_revision(2030)
