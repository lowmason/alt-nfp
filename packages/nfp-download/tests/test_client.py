"""Tests for nfp_download.client — HTTP client with retry logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
from nfp_download.client import (
    DEFAULT_HEADERS,
    DEFAULT_TIMEOUT,
    MAX_RETRIES,
    USER_AGENT,
    create_client,
    get_with_retry,
)


class TestCreateClient:
    """create_client() builds an httpx.Client with expected config."""

    def test_returns_client(self):
        client = create_client()
        try:
            assert isinstance(client, httpx.Client)
        finally:
            client.close()

    def test_default_headers_applied(self):
        client = create_client()
        try:
            for key, val in DEFAULT_HEADERS.items():
                assert client.headers[key] == val
        finally:
            client.close()

    def test_custom_headers_merged(self):
        client = create_client(headers={"X-Custom": "test"})
        try:
            assert client.headers["X-Custom"] == "test"
            assert client.headers["User-Agent"] == USER_AGENT
        finally:
            client.close()

    def test_custom_timeout(self):
        client = create_client(timeout=30.0)
        try:
            assert client.timeout.connect == 30.0
        finally:
            client.close()


class TestGetWithRetry:
    """get_with_retry() handles success, retries, and errors."""

    def _mock_client(self, responses):
        """Create a mock client that returns responses in sequence."""
        client = MagicMock(spec=httpx.Client)
        mock_responses = []
        for status_code, text in responses:
            r = MagicMock(spec=httpx.Response)
            r.status_code = status_code
            r.text = text
            if status_code >= 400:
                r.raise_for_status.side_effect = httpx.HTTPStatusError(
                    f"{status_code}", request=MagicMock(), response=r,
                )
            else:
                r.raise_for_status.return_value = None
            mock_responses.append(r)
        client.get.side_effect = mock_responses
        return client

    @patch("nfp_download.client.time.sleep")
    def test_success_on_first_try(self, mock_sleep):
        client = self._mock_client([(200, "ok")])
        r = get_with_retry(client, "https://example.com/data")
        assert r.status_code == 200
        mock_sleep.assert_not_called()

    @patch("nfp_download.client.time.sleep")
    def test_retry_on_429(self, mock_sleep):
        client = self._mock_client([(429, "rate limited"), (200, "ok")])
        r = get_with_retry(client, "https://example.com/data")
        assert r.status_code == 200
        mock_sleep.assert_called_once_with(1)  # 2^0 = 1

    @patch("nfp_download.client.time.sleep")
    def test_retry_on_500(self, mock_sleep):
        client = self._mock_client([(500, "error"), (200, "ok")])
        r = get_with_retry(client, "https://example.com/data")
        assert r.status_code == 200

    @patch("nfp_download.client.time.sleep")
    def test_non_retryable_error_raises_immediately(self, mock_sleep):
        client = self._mock_client([(403, "forbidden")])
        with pytest.raises(httpx.HTTPStatusError):
            get_with_retry(client, "https://example.com/data")
        mock_sleep.assert_not_called()

    @patch("nfp_download.client.time.sleep")
    def test_exponential_backoff(self, mock_sleep):
        client = self._mock_client([
            (500, "error"), (500, "error"), (500, "error"), (200, "ok"),
        ])
        get_with_retry(client, "https://example.com/data")
        waits = [call.args[0] for call in mock_sleep.call_args_list]
        assert waits == [1, 2, 4]  # 2^0, 2^1, 2^2

    @patch("nfp_download.client.time.sleep")
    def test_backoff_capped_at_120(self, mock_sleep):
        # 2^7 = 128, capped to 120
        responses = [(500, "error")] * 8 + [(200, "ok")]
        client = self._mock_client(responses)
        get_with_retry(client, "https://example.com/data", max_retries=9)
        waits = [call.args[0] for call in mock_sleep.call_args_list]
        assert waits[-1] == 120  # capped

    @patch("nfp_download.client._bls_api_key", return_value="TESTKEY")
    @patch("nfp_download.client.time.sleep")
    def test_bls_api_key_appended(self, mock_sleep, mock_key):
        client = self._mock_client([(200, "ok")])
        get_with_retry(client, "https://api.bls.gov/data")
        _, kwargs = client.get.call_args
        assert kwargs["params"]["registrationkey"] == "TESTKEY"

    @patch("nfp_download.client._bls_api_key", return_value="")
    @patch("nfp_download.client.time.sleep")
    def test_no_api_key_no_param(self, mock_sleep, mock_key):
        client = self._mock_client([(200, "ok")])
        get_with_retry(client, "https://api.bls.gov/data")
        _, kwargs = client.get.call_args
        assert "registrationkey" not in kwargs["params"]

    @patch("nfp_download.client._bls_api_key", return_value="TESTKEY")
    @patch("nfp_download.client.time.sleep")
    def test_non_bls_url_no_api_key(self, mock_sleep, mock_key):
        client = self._mock_client([(200, "ok")])
        get_with_retry(client, "https://example.com/data")
        _, kwargs = client.get.call_args
        assert "registrationkey" not in kwargs["params"]


class TestConstants:
    """Module-level constants have expected values."""

    def test_user_agent(self):
        assert "alt-nfp" in USER_AGENT

    def test_default_timeout(self):
        assert DEFAULT_TIMEOUT == 60.0

    def test_max_retries(self):
        assert MAX_RETRIES == 8
