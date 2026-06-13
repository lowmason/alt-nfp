"""Tests for new ingest modules: aggregate, tagger, releases, state-level QCEW."""

from datetime import date
from pathlib import Path

import polars as pl
import pytest
from nfp_ingest import releases
from nfp_ingest.aggregate import aggregate_geo
from nfp_ingest.releases import COMBINED_COLUMNS, combine_estimates
from nfp_ingest.tagger import latest_vintage_lookup, tag_estimates
from nfp_lookups.schemas import PANEL_SCHEMA


class TestAggregateGeo:
    """Tests for geographic aggregation."""

    @pytest.fixture()
    def state_df(self) -> pl.DataFrame:
        """Create a test DataFrame with national + state rows."""
        return pl.DataFrame({
            'source': ['qcew'] * 5,
            'seasonally_adjusted': [False] * 5,
            'geographic_type': ['national', 'state', 'state', 'state', 'state'],
            'geographic_code': ['US', '36', '34', '42', '06'],
            'industry_type': ['supersector'] * 5,
            'industry_code': ['30'] * 5,
            'ref_date': [date(2020, 1, 12)] * 5,
            'employment': [1000.0, 200.0, 150.0, 180.0, 400.0],
        })

    def test_adds_region_rows(self, state_df: pl.DataFrame):
        result = aggregate_geo(state_df)
        regions = result.filter(pl.col('geographic_type') == 'region')
        assert regions.height > 0

    def test_adds_division_rows(self, state_df: pl.DataFrame):
        result = aggregate_geo(state_df)
        divisions = result.filter(pl.col('geographic_type') == 'division')
        assert divisions.height > 0

    def test_preserves_national(self, state_df: pl.DataFrame):
        result = aggregate_geo(state_df)
        national = result.filter(pl.col('geographic_type') == 'national')
        assert national.height == 1
        assert national['employment'][0] == 1000.0

    def test_preserves_state(self, state_df: pl.DataFrame):
        result = aggregate_geo(state_df)
        states = result.filter(pl.col('geographic_type') == 'state')
        assert states.height == 4

    def test_northeast_region_sums(self, state_df: pl.DataFrame):
        # NY (36), NJ (34), PA (42) are all Region 1 (Northeast)
        result = aggregate_geo(state_df)
        ne_region = result.filter(
            (pl.col('geographic_type') == 'region')
            & (pl.col('geographic_code') == '1')
        )
        assert ne_region.height == 1
        # 200 (NY) + 150 (NJ) + 180 (PA)
        assert ne_region['employment'][0] == 530.0

    def test_pacific_division(self, state_df: pl.DataFrame):
        # CA (06) is Division 09 (Pacific)
        result = aggregate_geo(state_df)
        pacific = result.filter(
            (pl.col('geographic_type') == 'division')
            & (pl.col('geographic_code') == '09')
        )
        assert pacific.height == 1
        assert pacific['employment'][0] == 400.0

    def test_empty_state_df(self):
        df = pl.DataFrame({
            'source': ['qcew'],
            'geographic_type': ['national'],
            'geographic_code': ['US'],
            'industry_type': ['supersector'],
            'industry_code': ['30'],
            'ref_date': [date(2020, 1, 12)],
            'employment': [1000.0],
        })
        result = aggregate_geo(df)
        # Only national row, no regions or divisions
        assert result.filter(pl.col('geographic_type') == 'region').height == 0
        assert result.filter(pl.col('geographic_type') == 'division').height == 0


class TestLatestVintageLookup:
    """Tests for vintage lookup aggregation."""

    def test_max_per_ref_date(self):
        vintage_df = pl.DataFrame({
            'publication': ['ces', 'ces', 'ces'],
            'ref_date': [date(2020, 1, 12)] * 3,
            'vintage_date': [date(2020, 2, 7), date(2020, 3, 6), date(2020, 4, 3)],
            'revision': [0, 1, 2],
            'benchmark_revision': [0, 0, 0],
        })
        result = latest_vintage_lookup(vintage_df, 'ces')
        assert result.height == 1
        assert result['revision'][0] == 2
        assert result['vintage_date'][0] == date(2020, 4, 3)

    def test_filters_by_publication(self):
        vintage_df = pl.DataFrame({
            'publication': ['ces', 'qcew'],
            'ref_date': [date(2020, 1, 12)] * 2,
            'vintage_date': [date(2020, 2, 7), date(2020, 8, 19)],
            'revision': [0, 0],
            'benchmark_revision': [0, 0],
        })
        result = latest_vintage_lookup(vintage_df, 'ces')
        assert result.height == 1
        assert result['vintage_date'][0] == date(2020, 2, 7)

    def test_benchmark_day_emits_both_prints(self):
        """On a benchmark release day a December ref_date carries both an
        ordinary second print (rev-1, bmr-0) and a benchmark reprint
        (rev-2, bmr-1). Both rows must survive — the rev-1 level is never
        republished, so a lost capture is unrecoverable (the store append
        anti-joins on (revision, benchmark_revision)).

        Regression for the column-wise-max shadowing documented in
        specs/ces_growth_convention.md §4(c): independent maxes fused these
        into a single (rev-2, bmr-1) tag, dropping the rev-1 row entirely.
        """
        # Dec-2025 as the CES calendar holds it on the 2026-02-11 benchmark
        # release: rev-0 first print, rev-1 ordinary second print (one month
        # later by the scheduled offset), and the benchmark reprint stamped at
        # the Jan-2026 release. Jan-2026's own first print rides along.
        vintage_df = pl.DataFrame({
            'publication': ['ces'] * 4,
            'ref_date': [date(2025, 12, 12)] * 3 + [date(2026, 1, 12)],
            'vintage_date': [
                date(2026, 1, 9),   # Dec rev-0 first print
                date(2026, 2, 9),   # Dec rev-1 ordinary second print
                date(2026, 2, 11),  # Dec benchmark reprint
                date(2026, 2, 11),  # Jan rev-0 first print
            ],
            'revision': [0, 1, 2, 0],
            'benchmark_revision': [0, 0, 1, 0],
        })
        result = latest_vintage_lookup(vintage_df, 'ces')

        dec = result.filter(pl.col('ref_date') == date(2025, 12, 12)).sort(
            'benchmark_revision'
        )
        # Both the ordinary print and the benchmark reprint survive.
        assert dec.height == 2
        # ...as coherent (vintage_date, revision, benchmark_revision) pairs,
        # not a column-wise (max vintage, max rev, max bmr) fusion.
        ordinary = dec.row(0, named=True)
        benchmark = dec.row(1, named=True)
        assert (ordinary['revision'], ordinary['benchmark_revision']) == (1, 0)
        assert ordinary['vintage_date'] == date(2026, 2, 9)
        assert (benchmark['revision'], benchmark['benchmark_revision']) == (2, 1)
        assert benchmark['vintage_date'] == date(2026, 2, 11)
        # The fused (max vintage, max rev) = (2026-02-11, rev-1) pair the old
        # code could synthesize never appears.
        assert (
            result.filter(
                (pl.col('ref_date') == date(2025, 12, 12))
                & (pl.col('revision') == 1)
                & (pl.col('vintage_date') == date(2026, 2, 11))
            ).height
            == 0
        )

        # Jan-2026 (no benchmark in its own year yet) stays a single row.
        jan = result.filter(pl.col('ref_date') == date(2026, 1, 12))
        assert jan.height == 1
        assert (jan['revision'][0], jan['benchmark_revision'][0]) == (0, 0)


class TestLatestCesVintageDates:
    """Tests for ``releases._latest_ces_vintage_dates`` (the build_releases path).

    Same per-benchmark-track contract as ``latest_vintage_lookup``: keyed only
    on ``ref_date`` it drops the rev-1 ordinary print on benchmark day. This
    feeds the live-capture ``alt-nfp current`` path, which has no triangular
    coverage, so the loss is unrecoverable there.
    """

    def _write_vintage_dates(self, tmp_path: Path, df: pl.DataFrame, monkeypatch) -> None:
        path = tmp_path / 'vintage_dates.parquet'
        df.write_parquet(path)
        monkeypatch.setattr(releases, 'VINTAGE_DATES_PATH', path)

    def test_max_per_ref_date(self, tmp_path: Path, monkeypatch):
        self._write_vintage_dates(
            tmp_path,
            pl.DataFrame({
                'publication': ['ces', 'ces', 'ces'],
                'ref_date': [date(2020, 1, 12)] * 3,
                'vintage_date': [date(2020, 2, 7), date(2020, 3, 6), date(2020, 4, 3)],
                'revision': [0, 1, 2],
                'benchmark_revision': [0, 0, 0],
            }),
            monkeypatch,
        )
        result = releases._latest_ces_vintage_dates()
        assert result.height == 1
        assert result['revision'][0] == 2
        assert result['vintage_date'][0] == date(2020, 4, 3)

    def test_filters_by_publication(self, tmp_path: Path, monkeypatch):
        self._write_vintage_dates(
            tmp_path,
            pl.DataFrame({
                'publication': ['ces', 'qcew'],
                'ref_date': [date(2020, 1, 12)] * 2,
                'vintage_date': [date(2020, 2, 7), date(2020, 8, 19)],
                'revision': [0, 0],
                'benchmark_revision': [0, 0],
            }),
            monkeypatch,
        )
        result = releases._latest_ces_vintage_dates()
        assert result.height == 1
        assert result['vintage_date'][0] == date(2020, 2, 7)

    def test_missing_file_returns_empty(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(releases, 'VINTAGE_DATES_PATH', tmp_path / 'nonexistent.parquet')
        result = releases._latest_ces_vintage_dates()
        assert result.is_empty()
        assert result.schema['benchmark_revision'] == pl.UInt8

    def test_benchmark_day_emits_both_prints(self, tmp_path: Path, monkeypatch):
        """On a benchmark release day a December ref_date carries both an
        ordinary second print (rev-1, bmr-0) and a benchmark reprint
        (rev-2, bmr-1). Both rows must survive so both ``(revision,
        benchmark_revision)`` keys reach the store via the build_releases path.

        Regression for the same column-wise shadowing fixed in
        ``tagger.latest_vintage_lookup`` and documented in
        ``specs/ces_growth_convention.md`` §4(c): keying only on ``ref_date``
        let the later-stamped benchmark row drop the rev-1 print.
        """
        self._write_vintage_dates(
            tmp_path,
            pl.DataFrame({
                'publication': ['ces'] * 4,
                'ref_date': [date(2025, 12, 12)] * 3 + [date(2026, 1, 12)],
                'vintage_date': [
                    date(2026, 1, 9),   # Dec rev-0 first print
                    date(2026, 2, 9),   # Dec rev-1 ordinary second print
                    date(2026, 2, 11),  # Dec benchmark reprint
                    date(2026, 2, 11),  # Jan rev-0 first print
                ],
                'revision': [0, 1, 2, 0],
                'benchmark_revision': [0, 0, 1, 0],
            }),
            monkeypatch,
        )
        result = releases._latest_ces_vintage_dates()

        dec = result.filter(pl.col('ref_date') == date(2025, 12, 12)).sort(
            'benchmark_revision'
        )
        # Both the ordinary print and the benchmark reprint survive...
        assert dec.height == 2
        # ...as coherent (vintage_date, revision, benchmark_revision) triples.
        ordinary = dec.row(0, named=True)
        benchmark = dec.row(1, named=True)
        assert (ordinary['revision'], ordinary['benchmark_revision']) == (1, 0)
        assert ordinary['vintage_date'] == date(2026, 2, 9)
        assert (benchmark['revision'], benchmark['benchmark_revision']) == (2, 1)
        assert benchmark['vintage_date'] == date(2026, 2, 11)
        # The fused (max vintage, max rev) = (2026-02-11, rev-1) pair the old
        # ref_date-only keying could synthesize never appears.
        assert (
            result.filter(
                (pl.col('ref_date') == date(2025, 12, 12))
                & (pl.col('revision') == 1)
                & (pl.col('vintage_date') == date(2026, 2, 11))
            ).height
            == 0
        )

        # Jan-2026 (no benchmark in its own year yet) stays a single row.
        jan = result.filter(pl.col('ref_date') == date(2026, 1, 12))
        assert jan.height == 1
        assert (jan['revision'][0], jan['benchmark_revision'][0]) == (0, 0)


class TestTagEstimates:
    """Tests for tagging estimates with vintage info."""

    @pytest.fixture()
    def vintage_dates_path(self, tmp_path: Path) -> Path:
        vintage_df = pl.DataFrame({
            'publication': ['ces', 'ces', 'ces', 'ces'],
            'ref_date': [
                date(2020, 1, 12), date(2020, 1, 12),
                date(2020, 2, 12), date(2020, 2, 12),
            ],
            'vintage_date': [
                date(2020, 2, 7), date(2020, 3, 6),
                date(2020, 3, 6), date(2020, 4, 3),
            ],
            'revision': [0, 1, 0, 1],
            'benchmark_revision': [0, 0, 0, 0],
        })
        path = tmp_path / 'vintage_dates.parquet'
        vintage_df.write_parquet(path)
        return path

    def test_tags_are_joined(self, vintage_dates_path: Path):
        estimates = pl.DataFrame({
            'ref_date': [date(2020, 1, 12), date(2020, 2, 12)],
            'employment': [100000.0, 100200.0],
        })
        result = tag_estimates(estimates, 'ces', vintage_dates_path)
        assert 'vintage_date' in result.columns
        assert 'revision' in result.columns
        assert 'benchmark_revision' in result.columns
        assert result['revision'].null_count() == 0

    def test_replaces_existing_columns(self, vintage_dates_path: Path):
        estimates = pl.DataFrame({
            'ref_date': [date(2020, 1, 12)],
            'employment': [100000.0],
            'vintage_date': [date(1900, 1, 1)],
            'revision': [99],
            'benchmark_revision': [99],
        })
        result = tag_estimates(estimates, 'ces', vintage_dates_path)
        assert result['revision'][0] != 99  # Should be replaced

    def test_missing_file_raises(self, tmp_path: Path):
        estimates = pl.DataFrame({'ref_date': [date(2020, 1, 12)]})
        with pytest.raises(FileNotFoundError):
            tag_estimates(estimates, 'ces', tmp_path / 'nonexistent.parquet')


class TestCombineEstimates:
    """Tests for combined releases output."""

    @pytest.fixture()
    def estimate_files(self, tmp_path: Path) -> tuple[Path, Path]:
        qcew = pl.DataFrame({
            'source': ['qcew'],
            'seasonally_adjusted': [False],
            'geographic_type': ['national'],
            'geographic_code': ['US'],
            'industry_type': ['supersector'],
            'industry_code': ['30'],
            'ref_date': [date(2020, 3, 12)],
            'employment': [12000000.0],
        })
        ces = pl.DataFrame({
            'source': ['ces'],
            'seasonally_adjusted': [True],
            'geographic_type': ['national'],
            'geographic_code': ['US'],
            'industry_type': ['supersector'],
            'industry_code': ['30'],
            'ref_date': [date(2020, 3, 12)],
            'vintage_date': [date(2020, 4, 3)],
            'revision': [0],
            'benchmark_revision': [0],
            'employment': [12100000.0],
        })
        qcew_path = tmp_path / 'qcew_estimates.parquet'
        ces_path = tmp_path / 'ces_estimates.parquet'
        qcew.write_parquet(qcew_path)
        ces.write_parquet(ces_path)
        return qcew_path, ces_path

    def test_combines_files(self, estimate_files: tuple[Path, Path], tmp_path: Path):
        qcew_path, ces_path = estimate_files
        out = tmp_path / 'releases.parquet'
        result = combine_estimates(qcew_path, ces_path, out_path=out)
        assert result.height == 2
        assert set(result.columns) == set(COMBINED_COLUMNS)

    def test_writes_parquet(self, estimate_files: tuple[Path, Path], tmp_path: Path):
        qcew_path, ces_path = estimate_files
        out = tmp_path / 'releases.parquet'
        combine_estimates(qcew_path, ces_path, out_path=out)
        assert out.exists()
        reloaded = pl.read_parquet(out)
        assert reloaded.height == 2

    def test_skips_missing_files(self, tmp_path: Path):
        out = tmp_path / 'releases.parquet'
        result = combine_estimates(
            tmp_path / 'nonexistent.parquet',
            out_path=out,
        )
        assert result.height == 0

    def test_fills_missing_vintage_columns(
        self, estimate_files: tuple[Path, Path], tmp_path: Path,
    ):
        qcew_path, ces_path = estimate_files
        out = tmp_path / 'releases.parquet'
        result = combine_estimates(qcew_path, ces_path, out_path=out)
        # QCEW file didn't have vintage columns; they should be null
        qcew_row = result.filter(pl.col('source') == 'qcew')
        assert qcew_row['vintage_date'][0] is None
        # CES file had vintage columns; they should be preserved
        ces_row = result.filter(pl.col('source') == 'ces')
        assert ces_row['vintage_date'][0] == date(2020, 4, 3)


class TestPanelSchemaGeography:
    """Tests that PANEL_SCHEMA includes geographic columns."""

    def test_schema_has_geographic_columns(self):
        assert 'geographic_type' in PANEL_SCHEMA
        assert 'geographic_code' in PANEL_SCHEMA
        assert PANEL_SCHEMA['geographic_type'] == pl.Utf8
        assert PANEL_SCHEMA['geographic_code'] == pl.Utf8

    def test_empty_panel_has_geographic_columns(self):
        df = pl.DataFrame(schema=PANEL_SCHEMA)
        assert 'geographic_type' in df.columns
        assert 'geographic_code' in df.columns

    def test_national_panel_validates(self):
        from nfp_lookups.schemas import validate_panel

        rows = [
            {
                'period': date(2023, m, 1),
                'geographic_type': 'national',
                'geographic_code': 'US',
                'industry_code': '05',
                'industry_level': 'supersector',
                'source': 'ces_sa',
                'source_type': 'official_sa',
                'growth': 0.001 * m,
                'employment_level': 100000.0 + m * 100,
                'is_seasonally_adjusted': True,
                'vintage_date': date(2023, m + 1 if m < 12 else 1, 1),
                'revision_number': 0,
                'is_final': False,
                'publication_lag_months': 1,
                'coverage_ratio': None,
            }
            for m in range(1, 4)
        ]
        df = pl.DataFrame(rows, schema=PANEL_SCHEMA)
        result = validate_panel(df)
        assert len(result) == 3

    def test_state_level_rows_validate(self):
        """State-level rows with different geographic_code don't conflict."""
        from nfp_lookups.schemas import validate_panel

        rows = [
            {
                'period': date(2023, 1, 1),
                'geographic_type': 'state',
                'geographic_code': fips,
                'industry_code': '31',
                'industry_level': 'sector',
                'source': 'qcew',
                'source_type': 'census',
                'growth': 0.002,
                'employment_level': 50000.0,
                'is_seasonally_adjusted': False,
                'vintage_date': date(2023, 6, 1),
                'revision_number': 0,
                'is_final': False,
                'publication_lag_months': 5,
                'coverage_ratio': None,
            }
            for fips in ['36', '34', '42']
        ]
        df = pl.DataFrame(rows, schema=PANEL_SCHEMA)
        result = validate_panel(df)
        assert len(result) == 3

    def test_mixed_geography_validates(self):
        """National + state rows for same period/industry don't conflict."""
        from nfp_lookups.schemas import validate_panel

        rows = [
            {
                'period': date(2023, 1, 1),
                'geographic_type': geo_type,
                'geographic_code': geo_code,
                'industry_code': '31',
                'industry_level': 'sector',
                'source': 'qcew',
                'source_type': 'census',
                'growth': 0.002,
                'employment_level': 50000.0,
                'is_seasonally_adjusted': False,
                'vintage_date': date(2023, 6, 1),
                'revision_number': 0,
                'is_final': False,
                'publication_lag_months': 5,
                'coverage_ratio': None,
            }
            for geo_type, geo_code in [
                ('national', 'US'),
                ('state', '36'),
                ('state', '06'),
            ]
        ]
        df = pl.DataFrame(rows, schema=PANEL_SCHEMA)
        result = validate_panel(df)
        assert len(result) == 3


class TestFetchQcewWithGeography:
    """Tests for state-level QCEW panel transformation (unit tests, no network)."""

    def test_import(self):
        """fetch_qcew_current_with_geography is importable."""
        from nfp_ingest.qcew import fetch_qcew_current_with_geography
        assert callable(fetch_qcew_current_with_geography)

    def test_ingest_qcew_accepts_include_states(self):
        """ingest_qcew accepts include_states parameter."""
        import inspect

        from nfp_ingest.qcew import ingest_qcew
        sig = inspect.signature(ingest_qcew)
        assert 'include_states' in sig.parameters
        assert 'state_fips_list' in sig.parameters

    def test_bls_fetch_qcew_with_geography_importable(self):
        """fetch_qcew_with_geography is exported from bls layer."""
        from nfp_download.bls import fetch_qcew_with_geography
        assert callable(fetch_qcew_with_geography)

    def test_qcew_csv_supports_area_slice(self):
        """BLSHttpClient.get_qcew_csv accepts slice_type='area'."""
        import inspect

        from nfp_download.bls import BLSHttpClient
        sig = inspect.signature(BLSHttpClient.get_qcew_csv)
        assert 'slice_type' in sig.parameters
