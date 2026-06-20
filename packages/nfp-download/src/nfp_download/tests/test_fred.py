"""Tests for the FRED API client and indicator store."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import httpx
import polars as pl
import pytest
from nfp_download.fred import fetch_fred_series

# ---------------------------------------------------------------------------
# FRED client — mocked
# ---------------------------------------------------------------------------


MOCK_FRED_RESPONSE = {
    "observations": [
        {"date": "2024-01-01", "value": "100.5"},
        {"date": "2024-02-01", "value": "101.2"},
        {"date": "2024-03-01", "value": "."},
        {"date": "2024-04-01", "value": "102.0"},
        {"date": "2024-05-01", "value": "invalid"},
    ]
}


class TestFetchFredSeries:
    """Test fetch_fred_series with mocked HTTP responses."""

    def _mock_response(self, payload: dict) -> MagicMock:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = payload
        resp.raise_for_status = MagicMock()
        return resp

    @patch("nfp_download.fred.httpx.Client")
    def test_parses_observations(self, mock_client_cls):
        ctx = MagicMock()
        ctx.get.return_value = self._mock_response(MOCK_FRED_RESPONSE)
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=ctx)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        df = fetch_fred_series("TEST", api_key="fake-key")

        assert set(df.columns) == {"ref_date", "value"}
        assert df.schema["ref_date"] == pl.Date
        assert df.schema["value"] == pl.Float64
        # "." and "invalid" should be dropped → 3 valid rows
        assert len(df) == 3

    @patch("nfp_download.fred.httpx.Client")
    def test_drops_missing_sentinel(self, mock_client_cls):
        ctx = MagicMock()
        ctx.get.return_value = self._mock_response(MOCK_FRED_RESPONSE)
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=ctx)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        df = fetch_fred_series("TEST", api_key="fake-key")
        dates = df["ref_date"].to_list()
        assert date(2024, 3, 1) not in dates

    @patch("nfp_download.fred.httpx.Client")
    def test_empty_observations(self, mock_client_cls):
        ctx = MagicMock()
        ctx.get.return_value = self._mock_response({"observations": []})
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=ctx)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        df = fetch_fred_series("EMPTY", api_key="fake-key")
        assert len(df) == 0
        assert set(df.columns) == {"ref_date", "value"}

    @patch("nfp_download.fred.httpx.Client")
    def test_values_are_correct(self, mock_client_cls):
        ctx = MagicMock()
        ctx.get.return_value = self._mock_response(MOCK_FRED_RESPONSE)
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=ctx)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        df = fetch_fred_series("TEST", api_key="fake-key")
        values = df["value"].to_list()
        assert values == [100.5, 101.2, 102.0]

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="FRED_API_KEY"):
            fetch_fred_series("TEST")


# ---------------------------------------------------------------------------
# Indicator store — read_indicator / download_indicators
# ---------------------------------------------------------------------------


class TestReadIndicator:
    """Test read_indicator with temp parquet files."""

    def test_reads_existing_parquet(self, tmp_path):
        from nfp_ingest.indicators import read_indicator

        df = pl.DataFrame({
            "ref_date": [date(2024, 1, 1), date(2024, 2, 1)],
            "value": [100.0, 101.0],
        })
        (tmp_path / "claims.parquet").parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(tmp_path / "claims.parquet")

        result = read_indicator("claims", store_dir=tmp_path)
        assert result is not None
        assert len(result) == 2
        assert set(result.columns) == {"ref_date", "value"}

    def test_missing_file_returns_none(self, tmp_path):
        from nfp_ingest.indicators import read_indicator

        result = read_indicator("nonexistent", store_dir=tmp_path)
        assert result is None


class TestDownloadIndicators:
    """Test download_indicators with mocked FRED client."""

    @patch("nfp_ingest.indicators.fetch_fred_series")
    def test_writes_parquet_files(self, mock_fetch, tmp_path):
        from nfp_ingest.indicators import download_indicators

        mock_fetch.return_value = pl.DataFrame({
            "ref_date": [date(2024, 1, 1), date(2024, 2, 1)],
            "value": [100.0, 101.0],
        })

        results = download_indicators(store_dir=tmp_path, api_key="fake")

        assert len(results) == 2
        for spec_name in ("claims", "jolts"):
            assert results[spec_name] == 2
            assert (tmp_path / f"{spec_name}.parquet").exists()

    @patch("nfp_ingest.indicators.fetch_fred_series")
    def test_handles_download_failure(self, mock_fetch, tmp_path):
        from nfp_ingest.indicators import download_indicators

        mock_fetch.side_effect = httpx.ConnectError("API down")

        results = download_indicators(store_dir=tmp_path, api_key="fake")

        for name in results:
            assert results[name] == 0


# ---------------------------------------------------------------------------
# Integration (requires network + FRED_API_KEY)
# ---------------------------------------------------------------------------


@pytest.mark.network
class TestFredIntegration:
    """Smoke-test against the real FRED API."""

    def test_fetch_jolts(self):
        df = fetch_fred_series("JTSJOL", start_date="2024-01-01")
        assert len(df) > 0
        assert "ref_date" in df.columns
        assert "value" in df.columns
