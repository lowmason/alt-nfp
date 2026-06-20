"""Integration tests for BLS download functions.

These tests require network access and are marked with @pytest.mark.network
so they can be skipped in CI with ``pytest -m 'not network'``.
"""

import pytest
from nfp_download.bls.ces_national import CES_SERIES_MAP, fetch_ces_national
from nfp_download.bls.ces_state import build_state_series_ids, fetch_ces_state
from nfp_download.bls.qcew import QCEW_INDUSTRY_CODES, fetch_qcew


@pytest.mark.network
class TestFetchQcew:
    """Integration tests for QCEW downloads."""

    def test_single_quarter(self):
        """Fetch one quarter of QCEW data and verify expected columns."""
        df = fetch_qcew(
            years=[2023],
            quarters=[1],
            industries=['1022'],  # Manufacturing
        )
        assert len(df) > 0
        assert 'area_fips' in df.columns
        assert 'own_code' in df.columns
        assert 'industry_code' in df.columns
        assert 'year' in df.columns
        assert 'qtr' in df.columns

    def test_monthly_employment_columns(self):
        """Verify monthly employment level columns are present."""
        df = fetch_qcew(
            years=[2023],
            quarters=[1],
            industries=['10'],  # Total all industries
        )
        if len(df) > 0:
            for col in ['month1_emplvl', 'month2_emplvl', 'month3_emplvl']:
                assert col in df.columns


@pytest.mark.network
class TestFetchCesNational:
    """Integration tests for CES National downloads."""

    def test_total_nonfarm(self):
        """Fetch total nonfarm employment and verify data shape."""
        df = fetch_ces_national(
            start_year=2023,
            end_year=2023,
            supersectors=['00'],
        )
        assert len(df) > 0
        assert 'series_id' in df.columns
        assert 'date' in df.columns
        assert 'value' in df.columns
        assert 'supersector_code' in df.columns
        assert 'is_seasonally_adjusted' in df.columns

    def test_series_map_completeness(self):
        """Verify CES_SERIES_MAP has entries for all supersectors."""
        for code in ['00', '05', '10', '20', '30', '40', '50', '55', '60', '65', '70', '80']:
            assert code in CES_SERIES_MAP
            assert 'sa' in CES_SERIES_MAP[code]
            assert 'nsa' in CES_SERIES_MAP[code]


@pytest.mark.network
class TestFetchCesState:
    """Integration tests for CES State downloads."""

    def test_single_state(self):
        """Fetch one state and verify series ID parsing."""
        df = fetch_ces_state(
            states=['36'],  # New York
            start_year=2023,
            end_year=2023,
            supersectors=['00000000'],  # Total nonfarm
            include_nsa=False,
        )
        assert len(df) > 0
        assert 'state_fips' in df.columns
        assert 'area_code' in df.columns
        assert 'supersector_code' in df.columns
        assert all(df['state_fips'] == '36')


class TestBuildStateSeriesIds:
    """Unit tests for build_state_series_ids (no network)."""

    def test_single_state_single_supersector(self):
        ids = build_state_series_ids(
            states=['36'],
            supersectors=['00000000'],
            seasonal='S',
        )
        assert len(ids) == 1
        assert ids[0].startswith('SMS36')
        assert len(ids[0]) == 20

    def test_multiple_states(self):
        ids = build_state_series_ids(
            states=['36', '06'],
            supersectors=['00000000'],
        )
        assert len(ids) == 2

    def test_default_supersectors(self):
        ids = build_state_series_ids(states=['36'])
        # 12 default supersectors
        assert len(ids) == 12


class TestQcewIndustryCodes:
    """Unit tests for QCEW_INDUSTRY_CODES constant."""

    def test_has_total(self):
        assert '10' in QCEW_INDUSTRY_CODES

    def test_has_manufacturing(self):
        assert '1022' in QCEW_INDUSTRY_CODES

    def test_has_all_private_sectors(self):
        expected = [
            '1012', '1013', '1021', '1022', '1023', '1024', '1025',
            '1026', '1027', '1028', '1029', '102A', '102B', '102C',
            '102D', '102E', '102F', '102G',
        ]
        for code in expected:
            assert code in QCEW_INDUSTRY_CODES
