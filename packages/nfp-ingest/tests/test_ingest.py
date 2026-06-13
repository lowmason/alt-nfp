"""Tests for nfp_ingest — panel validation and schema conformance."""

from datetime import date

import polars as pl
import pytest
from nfp_ingest.panel import build_panel
from nfp_lookups.schemas import (
    CES_VINTAGE_SCHEMA,
    PANEL_SCHEMA,
    QCEW_VINTAGE_SCHEMA,
    validate_panel,
)


def _make_panel_rows(n: int = 5, **overrides) -> list[dict]:
    """Generate n valid panel rows."""
    rows = []
    base_date = date(2023, 1, 1)
    for i in range(n):
        month = base_date.month + i
        year = base_date.year + (month - 1) // 12
        month = ((month - 1) % 12) + 1
        period = date(year, month, 1)

        row = {
            'period': period,
            'geographic_type': 'national',
            'geographic_code': 'US',
            'industry_code': '05',
            'industry_level': 'supersector',
            'source': 'ces_sa',
            'source_type': 'official_sa',
            'growth': 0.001 * (i + 1),
            'employment_level': 100000.0 + i * 100,
            'is_seasonally_adjusted': True,
            'vintage_date': date(year, month + 1 if month < 12 else 1, 1),
            'revision_number': 0,
            'is_final': False,
            'publication_lag_months': 1,
            'coverage_ratio': None,
        }
        row.update(overrides)
        rows.append(row)
    return rows


def _make_panel_df(n: int = 5, **overrides) -> pl.DataFrame:
    """Generate a valid panel DataFrame."""
    return pl.DataFrame(_make_panel_rows(n, **overrides), schema=PANEL_SCHEMA)


class TestValidatePanel:
    """Tests for validate_panel."""

    def test_validate_panel_good(self):
        """A well-formed mini DataFrame passes validation."""
        df = _make_panel_df(5)
        result = validate_panel(df)
        assert len(result) == 5

    def test_validate_panel_duplicate(self):
        """Duplicate (period, source, industry_code, revision_number) raises ValueError."""
        rows = _make_panel_rows(3)
        # Create a duplicate of the first row
        dup_row = rows[0].copy()
        rows.append(dup_row)
        df = pl.DataFrame(rows, schema=PANEL_SCHEMA)

        with pytest.raises(ValueError, match='duplicate'):
            validate_panel(df)

    def test_validate_panel_missing_column(self):
        """Missing required column raises ValueError."""
        df = _make_panel_df(3)
        df = df.drop('growth')

        with pytest.raises(ValueError, match='Missing required columns'):
            validate_panel(df)

    def test_qcew_vintage_parquet_schema(self, tmp_path):
        """A sample QCEW vintage parquet with correct schema loads correctly."""
        # Create a minimal conforming parquet
        data = {
            'ref_year': [2023, 2023, 2023],
            'ref_quarter': [1, 1, 1],
            'ref_month': [1, 2, 3],
            'area_fips': ['US000', 'US000', 'US000'],
            'industry_code': ['31', '31', '31'],
            'own_code': [5, 5, 5],
            'employment': [12000000, 12010000, 12020000],
            'revision_number': [0, 0, 0],
            'vintage_date': [date(2023, 6, 1), date(2023, 6, 1), date(2023, 6, 1)],
        }
        df = pl.DataFrame(data, schema=QCEW_VINTAGE_SCHEMA)
        parquet_path = tmp_path / 'qcew_vintages.parquet'
        df.write_parquet(parquet_path)

        # Load and verify
        from nfp_ingest.qcew import load_qcew_vintages

        result = load_qcew_vintages(parquet_path)
        assert len(result) > 0
        assert set(PANEL_SCHEMA.keys()).issubset(set(result.columns))

    def test_ces_vintage_parquet_schema(self, tmp_path):
        """A sample CES vintage parquet with correct schema loads correctly."""
        data = {
            'ref_date': [date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)],
            'supersector_code': ['30', '30', '30'],
            'seasonal_adjustment': ['S', 'S', 'S'],
            'employment': [12800.0, 12810.0, 12820.0],
            'revision_number': [0, 0, 0],
            'vintage_date': [date(2023, 2, 1), date(2023, 3, 1), date(2023, 4, 1)],
            'bls_series_id': ['CES3000000001', 'CES3000000001', 'CES3000000001'],
        }
        df = pl.DataFrame(data, schema=CES_VINTAGE_SCHEMA)
        parquet_path = tmp_path / 'ces_vintages.parquet'
        df.write_parquet(parquet_path)

        from nfp_ingest.ces_national import load_ces_vintages

        result = load_ces_vintages(parquet_path)
        assert len(result) > 0
        assert set(PANEL_SCHEMA.keys()).issubset(set(result.columns))

    def test_validate_panel_non_finite_growth(self):
        """Non-finite growth values raise ValueError."""
        rows = _make_panel_rows(3)
        rows[1]['growth'] = float('inf')
        df = pl.DataFrame(rows, schema=PANEL_SCHEMA)

        with pytest.raises(ValueError, match='non-finite'):
            validate_panel(df)

    def test_validate_panel_nan_growth(self):
        """NaN growth values raise ValueError."""
        rows = _make_panel_rows(3)
        rows[1]['growth'] = float('nan')
        df = pl.DataFrame(rows, schema=PANEL_SCHEMA)

        with pytest.raises(ValueError, match='non-finite'):
            validate_panel(df)

    def test_validate_panel_null_growth_ok(self):
        """Null growth values pass validation (only non-finite is rejected)."""
        rows = _make_panel_rows(3)
        rows[1]['growth'] = None
        df = pl.DataFrame(rows, schema=PANEL_SCHEMA)
        result = validate_panel(df)
        assert len(result) == 3

    def test_validate_panel_wrong_dtype(self):
        """Wrong dtype raises ValueError."""
        df = _make_panel_df(3)
        df = df.with_columns(pl.col('growth').cast(pl.Int32))

        with pytest.raises(ValueError, match='dtype'):
            validate_panel(df)

    def test_validate_panel_payroll_bad_revision(self):
        """Payroll rows with revision_number != 0 raise ValueError."""
        rows = _make_panel_rows(2, source='pp1', source_type='payroll', revision_number=1)
        df = pl.DataFrame(rows, schema=PANEL_SCHEMA)

        with pytest.raises(ValueError, match='payroll'):
            validate_panel(df)

    @pytest.mark.real_store  # build_panel() reads the real vintage store; exempt from cred blanking
    def test_panel_builds_from_vintage_store(self):
        """build_panel() produces a valid panel from the vintage store."""
        from nfp_ingest.vintage_store import VINTAGE_STORE_PATH

        try:
            store_present = VINTAGE_STORE_PATH.exists()
        except Exception:
            store_present = False
        if not store_present:
            pytest.skip('Vintage store not present')

        panel = build_panel()
        assert len(panel) > 0
        assert set(PANEL_SCHEMA.keys()).issubset(set(panel.columns))


class TestSaveLoadPanel:
    """Tests for save_panel / load_panel round-trip."""

    def test_save_load_roundtrip(self, tmp_path):
        """Panel survives save → load with identical data."""
        from nfp_ingest.panel import load_panel, save_panel

        df = _make_panel_df(5)
        save_panel(df, tmp_path)

        loaded = load_panel(tmp_path)
        assert loaded.equals(df)

    def test_save_creates_manifest(self, tmp_path):
        """save_panel writes a panel_manifest.json with expected keys."""
        import json

        from nfp_ingest.panel import save_panel

        df = _make_panel_df(5)
        save_panel(df, tmp_path)

        manifest_path = tmp_path / 'panel_manifest.json'
        assert manifest_path.exists()
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest['row_count'] == 5
        assert 'source_counts' in manifest
        assert 'ces_sa' in manifest['source_counts']
        assert 'date_range' in manifest
        assert 'build_timestamp' in manifest

    def test_load_missing_raises(self, tmp_path):
        """load_panel on a directory without parquet raises FileNotFoundError."""
        from nfp_ingest.panel import load_panel

        with pytest.raises(FileNotFoundError):
            load_panel(tmp_path)

    def test_save_creates_directory(self, tmp_path):
        """save_panel creates output directory if it doesn't exist."""
        from nfp_ingest.panel import save_panel

        out = tmp_path / 'nested' / 'dir'
        save_panel(_make_panel_df(3), out)
        assert (out / 'observation_panel.parquet').exists()

    def test_save_load_empty_panel(self, tmp_path):
        """Empty panel round-trips correctly."""
        from nfp_ingest.panel import load_panel, save_panel

        df = pl.DataFrame(schema=PANEL_SCHEMA)
        save_panel(df, tmp_path)

        loaded = load_panel(tmp_path)
        assert len(loaded) == 0
        assert set(loaded.columns) == set(PANEL_SCHEMA.keys())
