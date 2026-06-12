"""Tests for revision schedule vintage date functions and noise multiplier lookups."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from nfp_lookups.revision_schedules import (
    PublicationCalendar,
    get_ces_vintage_date,
    get_noise_multiplier,
    get_qcew_vintage_date,
)

# ---------------------------------------------------------------------------
# Empty calendar → lag-based fallback
# ---------------------------------------------------------------------------


class TestQcewVintageDateFallback:
    """get_qcew_vintage_date with empty calendar uses lag-based approximation."""

    @pytest.fixture()
    def empty_cal(self):
        return PublicationCalendar(
            ces_release_dates=pl.DataFrame(
                schema={"ref_month": pl.Date, "revision_number": pl.Int32, "pub_date": pl.Date}
            ),
            qcew_release_dates=pl.DataFrame(
                schema={
                    "ref_quarter": pl.Utf8,
                    "ref_year": pl.Int32,
                    "revision_number": pl.Int32,
                    "pub_date": pl.Date,
                }
            ),
        )

    def test_q1_rev0(self, empty_cal):
        d = get_qcew_vintage_date("Q1", 2024, 0, calendar=empty_cal)
        # Q1 end = March, lag = 5 → August 2024
        assert d == date(2024, 8, 1)

    def test_q4_rev0(self, empty_cal):
        d = get_qcew_vintage_date("Q4", 2023, 0, calendar=empty_cal)
        # Q4 end = December, lag = 8 → August 2024
        assert d == date(2024, 8, 1)

    def test_q2_rev1(self, empty_cal):
        d = get_qcew_vintage_date("Q2", 2024, 1, calendar=empty_cal)
        # Q2 end = June, lag = 11 → May 2025
        assert d == date(2025, 5, 1)

    def test_invalid_quarter_raises(self, empty_cal):
        with pytest.raises(ValueError, match="Invalid ref_quarter"):
            get_qcew_vintage_date("Q5", 2024, 0, calendar=empty_cal)

    def test_revision_exceeds_max_raises(self, empty_cal):
        with pytest.raises(ValueError, match="not available"):
            # Q4 only has revisions 0 and 1
            get_qcew_vintage_date("Q4", 2024, 2, calendar=empty_cal)

    def test_all_quarters_rev0_valid(self, empty_cal):
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            d = get_qcew_vintage_date(q, 2024, 0, calendar=empty_cal)
            assert isinstance(d, date)


class TestCesVintageDateFallback:
    """get_ces_vintage_date with empty calendar uses lag-based approximation."""

    @pytest.fixture()
    def empty_cal(self):
        return PublicationCalendar(
            ces_release_dates=pl.DataFrame(
                schema={"ref_month": pl.Date, "revision_number": pl.Int32, "pub_date": pl.Date}
            ),
            qcew_release_dates=pl.DataFrame(
                schema={
                    "ref_quarter": pl.Utf8,
                    "ref_year": pl.Int32,
                    "revision_number": pl.Int32,
                    "pub_date": pl.Date,
                }
            ),
        )

    def test_rev0_lag1(self, empty_cal):
        d = get_ces_vintage_date(date(2024, 1, 1), 0, calendar=empty_cal)
        assert d == date(2024, 2, 1)

    def test_rev2_lag3(self, empty_cal):
        d = get_ces_vintage_date(date(2024, 6, 1), 2, calendar=empty_cal)
        assert d == date(2024, 9, 1)

    def test_benchmark_lag13(self, empty_cal):
        d = get_ces_vintage_date(date(2024, 3, 1), -1, calendar=empty_cal)
        # March + 13 = April next year
        assert d == date(2025, 4, 1)

    def test_year_rollover(self, empty_cal):
        d = get_ces_vintage_date(date(2024, 11, 1), 2, calendar=empty_cal)
        # November + 3 = February next year
        assert d == date(2025, 2, 1)

    def test_invalid_revision_raises(self, empty_cal):
        with pytest.raises(ValueError, match="not in CES schedule"):
            get_ces_vintage_date(date(2024, 1, 1), 5, calendar=empty_cal)


# ---------------------------------------------------------------------------
# Exact calendar lookup takes precedence
# ---------------------------------------------------------------------------


class TestExactCalendarLookup:
    """When calendar has exact dates, they take precedence over lag-based."""

    def test_ces_exact_date(self):
        cal = PublicationCalendar(
            ces_release_dates=pl.DataFrame({
                "ref_month": [date(2024, 1, 1)],
                "revision_number": [0],
                "pub_date": [date(2024, 2, 7)],
            }),
            qcew_release_dates=pl.DataFrame(
                schema={
                    "ref_quarter": pl.Utf8,
                    "ref_year": pl.Int32,
                    "revision_number": pl.Int32,
                    "pub_date": pl.Date,
                }
            ),
        )
        d = get_ces_vintage_date(date(2024, 1, 1), 0, calendar=cal)
        assert d == date(2024, 2, 7)  # exact, not lag-based Feb 1

    def test_qcew_exact_date(self):
        cal = PublicationCalendar(
            ces_release_dates=pl.DataFrame(
                schema={"ref_month": pl.Date, "revision_number": pl.Int32, "pub_date": pl.Date}
            ),
            qcew_release_dates=pl.DataFrame({
                "ref_quarter": ["Q1"],
                "ref_year": [2024],
                "revision_number": [0],
                "pub_date": [date(2024, 8, 21)],
            }),
        )
        d = get_qcew_vintage_date("Q1", 2024, 0, calendar=cal)
        assert d == date(2024, 8, 21)


# ---------------------------------------------------------------------------
# Noise multiplier lookups
# ---------------------------------------------------------------------------


class TestGetNoiseMultiplier:
    """get_noise_multiplier returns correct values for all source types."""

    def test_ces_rev0(self):
        assert get_noise_multiplier("ces", 0) == 3.0

    def test_ces_rev1(self):
        assert get_noise_multiplier("ces", 1) == 2.0

    def test_ces_rev2(self):
        assert get_noise_multiplier("ces", 2) == 1.5

    def test_ces_benchmark(self):
        assert get_noise_multiplier("ces", -1) == 1.0

    def test_ces_unknown_rev_returns_1(self):
        assert get_noise_multiplier("ces", 99) == 1.0

    def test_qcew_q1_rev0(self):
        assert get_noise_multiplier("qcew_Q1", 0) == 25.0

    def test_qcew_q1_final(self):
        assert get_noise_multiplier("qcew_Q1", 4) == 1.0

    def test_qcew_q4_rev0(self):
        assert get_noise_multiplier("qcew_Q4", 0) == 17.0

    def test_qcew_unknown_rev_returns_1(self):
        assert get_noise_multiplier("qcew_Q2", 99) == 1.0

    def test_provider_returns_1(self):
        assert get_noise_multiplier("provider_G", 0) == 1.0

    def test_unknown_source_returns_1(self):
        assert get_noise_multiplier("something_else", 0) == 1.0
