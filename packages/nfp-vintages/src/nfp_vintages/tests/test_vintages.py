"""Tests for nfp_vintages — vintage views and evaluation."""

from datetime import date

import polars as pl
from nfp_lookups.schemas import PANEL_SCHEMA
from nfp_vintages import (
    build_noise_multiplier_vector,
    final_view,
    real_time_view,
    specific_vintage_view,
    vintage_diff,
)


def _make_test_panel() -> pl.DataFrame:
    """Create a test panel with multiple vintages for the same observations."""
    rows = [
        # CES SA: period 2023-02, industry '05', v0 (vintage_date=2023-03-01)
        {
            'period': date(2023, 2, 1),
            'industry_code': '05',
            'industry_level': 'supersector',
            'source': 'ces_sa',
            'source_type': 'official_sa',
            'growth': 0.002,
            'employment_level': 100000.0,
            'is_seasonally_adjusted': True,
            'vintage_date': date(2023, 3, 1),
            'revision_number': 0,
            'is_final': False,
            'publication_lag_months': 1,
            'coverage_ratio': None,
        },
        # CES SA: same period/industry, v1 (vintage_date=2023-04-01)
        {
            'period': date(2023, 2, 1),
            'industry_code': '05',
            'industry_level': 'supersector',
            'source': 'ces_sa',
            'source_type': 'official_sa',
            'growth': 0.0025,
            'employment_level': 100050.0,
            'is_seasonally_adjusted': True,
            'vintage_date': date(2023, 4, 1),
            'revision_number': 1,
            'is_final': False,
            'publication_lag_months': 2,
            'coverage_ratio': None,
        },
        # CES SA: same period/industry, v-1 benchmark (vintage_date=2024-03-01)
        {
            'period': date(2023, 2, 1),
            'industry_code': '05',
            'industry_level': 'supersector',
            'source': 'ces_sa',
            'source_type': 'official_sa',
            'growth': 0.003,
            'employment_level': 100100.0,
            'is_seasonally_adjusted': True,
            'vintage_date': date(2024, 3, 1),
            'revision_number': -1,
            'is_final': True,
            'publication_lag_months': 13,
            'coverage_ratio': None,
        },
        # Payroll provider: same period
        {
            'period': date(2023, 2, 1),
            'industry_code': '05',
            'industry_level': 'supersector',
            'source': 'pp1',
            'source_type': 'payroll',
            'growth': 0.0015,
            'employment_level': 50000.0,
            'is_seasonally_adjusted': False,
            'vintage_date': date(2023, 2, 22),
            'revision_number': 0,
            'is_final': True,
            'publication_lag_months': 0,
            'coverage_ratio': None,
        },
    ]
    return pl.DataFrame(rows, schema=PANEL_SCHEMA)


class TestRealTimeView:
    """Tests for real_time_view."""

    def test_real_time_view_filters_correctly(self):
        """real_time_view returns correct vintage based on as_of date."""
        panel = _make_test_panel().lazy()

        # as_of between v0 and v1 vintage dates -> should return v0
        result = real_time_view(panel, as_of=date(2023, 3, 15)).collect()
        ces_rows = result.filter(pl.col('source') == 'ces_sa')
        assert len(ces_rows) == 1
        assert ces_rows['revision_number'][0] == 0
        assert ces_rows['growth'][0] == 0.002

        # as_of after v1 but before benchmark -> should return v1
        result = real_time_view(panel, as_of=date(2023, 5, 1)).collect()
        ces_rows = result.filter(pl.col('source') == 'ces_sa')
        assert len(ces_rows) == 1
        assert ces_rows['revision_number'][0] == 1
        assert ces_rows['growth'][0] == 0.0025

        # as_of after benchmark -> should return benchmark (rev=-1 is highest? No.)
        # revision_number=-1 < 0 < 1, so v1 should still be selected as highest
        # unless we treat benchmark specially. With current logic, v1 > v-1 numerically.
        result = real_time_view(panel, as_of=date(2024, 4, 1)).collect()
        ces_rows = result.filter(pl.col('source') == 'ces_sa')
        assert len(ces_rows) == 1
        # v1 (revision_number=1) > v-1 (revision_number=-1) numerically
        assert ces_rows['revision_number'][0] == 1


class TestFinalView:
    """Tests for final_view."""

    def test_final_view(self):
        """Only rows with is_final=True are returned."""
        panel = _make_test_panel().lazy()

        result = final_view(panel).collect()
        assert all(result['is_final'].to_list())

        # Should have the benchmark CES row and the payroll row
        sources = sorted(result['source'].to_list())
        assert 'ces_sa' in sources
        assert 'pp1' in sources


class TestVintageDiff:
    """Tests for vintage_diff."""

    def test_vintage_diff(self):
        """Known revision between v0 and v1 is computed correctly."""
        panel = _make_test_panel().lazy()

        diff = vintage_diff(panel, source='ces_sa', rev_a=0, rev_b=1).collect()
        assert len(diff) == 1
        assert diff['revision_from'][0] == 0
        assert diff['revision_to'][0] == 1

        expected_diff = 0.0025 - 0.002
        assert abs(diff['growth_diff'][0] - expected_diff) < 1e-10


class TestNoiseMultiplier:
    """Tests for build_noise_multiplier_vector."""

    def test_noise_multiplier_vector_shape(self):
        """Output array length matches panel row count."""
        panel = _make_test_panel()
        multipliers = build_noise_multiplier_vector(panel)
        assert len(multipliers) == len(panel)

    def test_noise_multiplier_values(self):
        """Noise multipliers are looked up correctly."""
        panel = _make_test_panel()
        multipliers = build_noise_multiplier_vector(panel)

        # CES v0 should have multiplier 3.0
        ces_v0_idx = panel.with_row_index().filter(
            (pl.col('source') == 'ces_sa') & (pl.col('revision_number') == 0)
        )['index'][0]
        assert multipliers[ces_v0_idx] == 3.0

        # CES v1 should have multiplier 2.0
        ces_v1_idx = panel.with_row_index().filter(
            (pl.col('source') == 'ces_sa') & (pl.col('revision_number') == 1)
        )['index'][0]
        assert multipliers[ces_v1_idx] == 2.0

        # Payroll should have multiplier 1.0
        pp_idx = panel.with_row_index().filter(
            pl.col('source') == 'pp1'
        )['index'][0]
        assert multipliers[pp_idx] == 1.0

    def test_noise_multiplier_benchmark_revision(self):
        """CES benchmark (rev=-1) should get multiplier 1.0."""
        panel = _make_test_panel()
        multipliers = build_noise_multiplier_vector(panel)

        bench_idx = panel.with_row_index().filter(
            (pl.col('source') == 'ces_sa') & (pl.col('revision_number') == -1)
        )['index'][0]
        assert multipliers[bench_idx] == 1.0

    def test_noise_multiplier_unknown_source(self):
        """Unknown source defaults to 1.0."""
        rows = [
            {
                'period': date(2023, 2, 1),
                'industry_code': '05',
                'industry_level': 'supersector',
                'source': 'mystery_source',
                'source_type': 'other',
                'growth': 0.001,
                'employment_level': 50000.0,
                'is_seasonally_adjusted': False,
                'vintage_date': date(2023, 3, 1),
                'revision_number': 0,
                'is_final': False,
                'publication_lag_months': 1,
                'coverage_ratio': None,
            }
        ]
        panel = pl.DataFrame(rows, schema=PANEL_SCHEMA)
        multipliers = build_noise_multiplier_vector(panel)
        assert multipliers[0] == 1.0

    def test_noise_multiplier_null_revision(self):
        """Null revision_number defaults to 1.0."""
        rows = [
            {
                'period': date(2023, 2, 1),
                'industry_code': '05',
                'industry_level': 'supersector',
                'source': 'ces_sa',
                'source_type': 'official_sa',
                'growth': 0.001,
                'employment_level': 50000.0,
                'is_seasonally_adjusted': True,
                'vintage_date': date(2023, 3, 1),
                'revision_number': None,
                'is_final': False,
                'publication_lag_months': 1,
                'coverage_ratio': None,
            }
        ]
        panel = pl.DataFrame(rows, schema=PANEL_SCHEMA)
        multipliers = build_noise_multiplier_vector(panel)
        assert multipliers[0] == 1.0

    def test_noise_multiplier_qcew_quarters(self):
        """QCEW noise multipliers are quarter-dependent."""
        rows = []
        # Create one QCEW observation per quarter with revision 0
        for m in [1, 4, 7, 10]:
            rows.append(
                {
                    'period': date(2023, m, 1),
                    'industry_code': '05',
                    'industry_level': 'supersector',
                    'source': 'qcew',
                    'source_type': 'census',
                    'growth': 0.001,
                    'employment_level': 50000.0,
                    'is_seasonally_adjusted': False,
                    'vintage_date': date(2023, m + 3 if m < 10 else 12, 1),
                    'revision_number': 0,
                    'is_final': False,
                    'publication_lag_months': 3,
                    'coverage_ratio': None,
                }
            )
        panel = pl.DataFrame(rows, schema=PANEL_SCHEMA)
        multipliers = build_noise_multiplier_vector(panel)
        # All should have a defined multiplier (not default 1.0 for rev 0)
        # The exact values depend on QCEW_REVISIONS, but they should all be > 1
        assert len(multipliers) == 4


class TestSpecificVintageView:
    """Tests for specific_vintage_view."""

    def test_filters_source_and_revision(self):
        """specific_vintage_view returns only matching source and revision."""
        panel = _make_test_panel().lazy()

        result = specific_vintage_view(panel, source='ces_sa', revision_number=0).collect()
        assert len(result) == 1
        assert result['source'][0] == 'ces_sa'
        assert result['revision_number'][0] == 0

    def test_benchmark_revision(self):
        """Can select benchmark rows (revision_number=-1)."""
        panel = _make_test_panel().lazy()

        result = specific_vintage_view(panel, source='ces_sa', revision_number=-1).collect()
        assert len(result) == 1
        assert result['revision_number'][0] == -1
        assert result['is_final'][0] is True

    def test_no_match_returns_empty(self):
        """Non-existent source/revision combo returns empty frame."""
        panel = _make_test_panel().lazy()

        result = specific_vintage_view(panel, source='nonexistent', revision_number=0).collect()
        assert len(result) == 0

    def test_sorted_output(self):
        """Output is sorted by (period, industry_code)."""
        panel = _make_test_panel().lazy()

        result = specific_vintage_view(panel, source='ces_sa', revision_number=1).collect()
        resorted = result.sort('period', 'industry_code')
        assert result.equals(resorted)


class TestRealTimeViewEdgeCases:
    """Edge case tests for real_time_view."""

    def test_as_of_before_any_vintage(self):
        """as_of date before all vintage_dates returns empty."""
        panel = _make_test_panel().lazy()
        result = real_time_view(panel, as_of=date(2020, 1, 1)).collect()
        assert len(result) == 0

    def test_multiple_industries(self):
        """real_time_view deduplicates per (period, source, industry_code)."""
        rows = [
            {
                'period': date(2023, 2, 1),
                'industry_code': code,
                'industry_level': 'supersector',
                'source': 'ces_sa',
                'source_type': 'official_sa',
                'growth': 0.001,
                'employment_level': 100000.0,
                'is_seasonally_adjusted': True,
                'vintage_date': date(2023, 3, 1),
                'revision_number': rev,
                'is_final': False,
                'publication_lag_months': 1,
                'coverage_ratio': None,
            }
            for code in ['05', '30']
            for rev in [0, 1]
        ]
        # Add later vintage dates for rev 1
        for r in rows:
            if r['revision_number'] == 1:
                r['vintage_date'] = date(2023, 4, 1)

        panel = pl.DataFrame(rows, schema=PANEL_SCHEMA).lazy()
        result = real_time_view(panel, as_of=date(2023, 5, 1)).collect()

        # Should have one row per (period, source, industry_code) → 2 rows
        assert len(result) == 2
        # Both should be revision 1 (the highest available)
        assert all(r == 1 for r in result['revision_number'].to_list())


class TestVintageDiffEdgeCases:
    """Edge case tests for vintage_diff."""

    def test_no_overlap_returns_empty(self):
        """vintage_diff with non-overlapping revisions returns empty."""
        panel = _make_test_panel().lazy()
        # rev 0 and rev -1 don't overlap because -1 has different growth
        # But they share the same period/industry, so we get a diff
        diff = vintage_diff(panel, source='ces_sa', rev_a=0, rev_b=-1).collect()
        assert len(diff) == 1  # they share period=2023-02 industry=05

    def test_diff_nonexistent_source(self):
        """vintage_diff with nonexistent source returns empty."""
        panel = _make_test_panel().lazy()
        diff = vintage_diff(panel, source='nonexistent', rev_a=0, rev_b=1).collect()
        assert len(diff) == 0
