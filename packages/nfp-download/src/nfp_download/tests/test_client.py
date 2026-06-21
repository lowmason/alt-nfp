"""Tests for nfp_download.client — HTTP client with retry logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
from curl_cffi import requests as curl_requests
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


class TestGetWithRetryTransportExceptions:
    """get_with_retry() retries on transport-level exceptions, not just HTTP status codes."""

    def _make_200(self):
        """Build a mock 200 response."""
        r = MagicMock(spec=httpx.Response)
        r.status_code = 200
        r.text = "ok"
        r.raise_for_status.return_value = None
        return r

    @pytest.mark.parametrize(
        "exc",
        [
            httpx.ConnectError("connection refused"),
            httpx.ConnectTimeout("timed out"),
            httpx.ReadTimeout("read timed out"),
            httpx.RemoteProtocolError("connection reset"),
            curl_requests.exceptions.ConnectionError("curl connect failed"),
            curl_requests.exceptions.Timeout("curl timed out"),
        ],
        ids=[
            "httpx_ConnectError",
            "httpx_ConnectTimeout",
            "httpx_ReadTimeout",
            "httpx_RemoteProtocolError",
            "curl_ConnectionError",
            "curl_Timeout",
        ],
    )
    @patch("nfp_download.client.time.sleep")
    def test_transport_exception_retries_on_first_failure(self, mock_sleep, exc):
        """Transport exception on attempt 0 is retried; 200 on attempt 1 is returned."""
        client = MagicMock(spec=httpx.Client)
        ok = self._make_200()
        client.get.side_effect = [exc, ok]

        r = get_with_retry(client, "https://example.com/data", max_retries=3)

        assert r.status_code == 200
        assert client.get.call_count == 2
        mock_sleep.assert_called_once_with(1)  # 2^0 = 1

    @pytest.mark.parametrize(
        "exc_factory",
        [
            lambda: httpx.ConnectError("connection refused"),
            lambda: curl_requests.exceptions.ConnectionError("curl connect failed"),
        ],
        ids=["httpx", "curl_cffi"],
    )
    @patch("nfp_download.client.time.sleep")
    def test_transport_exception_gives_up_after_max_retries(self, mock_sleep, exc_factory):
        """After max_retries consecutive transport failures, the last exception is re-raised."""
        max_retries = 3
        client = MagicMock(spec=httpx.Client)
        exc = exc_factory()
        client.get.side_effect = [exc_factory() for _ in range(max_retries)]

        with pytest.raises(type(exc)):
            get_with_retry(client, "https://example.com/data", max_retries=max_retries)

        assert client.get.call_count == max_retries
        # Back-off on all attempts except the last (which raises immediately)
        assert mock_sleep.call_count == max_retries - 1

    @patch("nfp_download.client.time.sleep")
    def test_transport_exception_backoff_uses_exponential_wait(self, mock_sleep):
        """Back-off on transport exceptions uses the same 2^attempt formula as HTTP errors."""
        client = MagicMock(spec=httpx.Client)
        ok = self._make_200()
        client.get.side_effect = [
            httpx.ConnectError("refused"),
            httpx.ConnectError("refused"),
            httpx.ConnectError("refused"),
            ok,
        ]

        r = get_with_retry(client, "https://example.com/data", max_retries=5)

        assert r.status_code == 200
        waits = [call.args[0] for call in mock_sleep.call_args_list]
        assert waits == [1, 2, 4]  # 2^0, 2^1, 2^2

    @patch("nfp_download.client.time.sleep")
    def test_httpstatus_error_not_swallowed_by_transport_catch(self, mock_sleep):
        """A curl_cffi HTTPError from raise_for_status() is NOT swallowed by the transport catch.

        curl_cffi.requests.exceptions.HTTPError is a subclass of RequestException.
        This test guards against the try/except being too wide (wrapping raise_for_status).
        """
        client = MagicMock(spec=curl_requests.Session)
        r = MagicMock()
        r.status_code = 403
        r.raise_for_status.side_effect = curl_requests.exceptions.HTTPError("403 Forbidden")
        client.get.return_value = r

        with pytest.raises(curl_requests.exceptions.HTTPError):
            get_with_retry(client, "https://www.bls.gov/data", max_retries=3)

        # Must NOT retry — 403 is not a transport error or a retryable status code
        assert client.get.call_count == 1
        mock_sleep.assert_not_called()


class TestConstants:
    """Module-level constants have expected values."""

    def test_user_agent(self):
        assert "alt-nfp" in USER_AGENT

    def test_default_timeout(self):
        assert DEFAULT_TIMEOUT == 60.0

    def test_max_retries(self):
        assert MAX_RETRIES == 8
