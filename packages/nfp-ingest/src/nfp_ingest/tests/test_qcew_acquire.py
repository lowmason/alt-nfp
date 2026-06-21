"""Tests for the relocated QCEW acquire layer (nfp_ingest.qcew_acquire).

Phase 1 of the CLI production workflow (specs/cli_production_workflow.md §5.2, §14
step 1) relocates the two QCEW acquire entry points from nfp_vintages/rebuild_store.py
into this PUBLIC nfp_ingest module so capture.py (also in nfp-ingest) can call them
without an illegal upward import of private names.

Unit tests (no network) cover the pure-transform helpers and import-legality.
A @pytest.mark.network test fetches one real BLS area slice.

CRITICAL SAFETY: no test here writes to any store; no test calls the full network
fetch loops acquire_qcew_levels()/acquire_qcew_size_native() in CI.
"""

from __future__ import annotations

import polars as pl
import pytest


def _area_raw(rows: list[dict]) -> pl.DataFrame:
    """Build an all-string raw area frame (as _fetch_qcew_csv returns)."""
    cols = list(rows[0].keys())
    schema = dict.fromkeys(cols, pl.Utf8)
    return pl.DataFrame({c: [str(r[c]) for r in rows] for c in cols}, schema=schema)


def _area_row(
    *,
    own_code: str = "5",
    industry_code: str = "1013",
    agglvl_code: str = "13",
    month1: int = 10_000,
    month2: int = 10_100,
    month3: int = 10_200,
) -> dict:
    return {
        "area_fips": "US000",
        "own_code": own_code,
        "industry_code": industry_code,
        "agglvl_code": agglvl_code,
        "year": "2024",
        "qtr": "1",
        "month1_emplvl": str(month1),
        "month2_emplvl": str(month2),
        "month3_emplvl": str(month3),
        "disclosure_code": "",
        "total_qtrly_wages": "9999999",
    }


class TestPublicSymbolsExist:
    """The two acquire entry points must exist PUBLIC on nfp_ingest.qcew_acquire."""

    def test_acquire_qcew_levels_is_public_callable(self):
        from nfp_ingest.qcew_acquire import acquire_qcew_levels

        assert callable(acquire_qcew_levels)

    def test_acquire_qcew_size_native_is_public_callable(self):
        from nfp_ingest.qcew_acquire import acquire_qcew_size_native

        assert callable(acquire_qcew_size_native)


class TestImportLegality:
    """The module must NOT import nfp_vintages (it sits above nfp-ingest)."""

    def test_module_has_no_nfp_vintages_import(self):
        import inspect

        import nfp_ingest.qcew_acquire as mod

        src = inspect.getsource(mod)
        assert "nfp_vintages" not in src, (
            "qcew_acquire.py must not import nfp_vintages (illegal upward import)"
        )


class TestPrepAreaRaw:
    """_prep_area_raw moved verbatim — pure transform, no network."""

    def _prep(self, rows: list[dict]) -> pl.DataFrame:
        from nfp_ingest.qcew_acquire import _prep_area_raw

        return _prep_area_raw(_area_raw(rows))

    def test_filter_keeps_private_and_total(self):
        rows = [
            _area_row(own_code="5"),
            _area_row(own_code="0"),
            _area_row(own_code="1"),
            _area_row(own_code="2"),
            _area_row(own_code="3"),
        ]
        result = self._prep(rows)
        assert result.height == 2
        assert set(result["own_code"].to_list()) == {"5", "0"}

    def test_required_columns_exact(self):
        from nfp_ingest.qcew_acquire import _QCEW_LEVELS_REQUIRED

        result = self._prep([_area_row()])
        assert set(result.columns) == set(_QCEW_LEVELS_REQUIRED)

    def test_emplvl_cast_to_int64_and_revision_zero(self):
        result = self._prep([_area_row(month1=12345, month2=23456, month3=34567)])
        assert result["month1_emplvl"].dtype == pl.Int64
        assert result["revision"][0] == 0

    def test_hyphenated_industry_code_preserved(self):
        result = self._prep([_area_row(industry_code="44-45", agglvl_code="14")])
        assert result["industry_code"][0] == "44-45"


@pytest.mark.network
class TestAcquireLevelsNetwork:
    """Fetch ONE real QCEW area slice through the relocated helper (maintainer-run)."""

    def test_single_slice_non_empty(self):
        from nfp_download.client import create_client
        from nfp_ingest.qcew_acquire import _fetch_qcew_csv

        url = "https://data.bls.gov/cew/data/api/2024/1/area/US000.csv"
        with create_client() as session:
            raw = _fetch_qcew_csv(session, url)
        assert raw is not None
        assert raw.height > 0
