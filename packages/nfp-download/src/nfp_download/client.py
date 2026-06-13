"""Shared HTTP clients with retry logic for vintage pipeline requests.

Provides a pre-configured :class:`httpx.Client` with HTTP/2, browser-like
headers, and exponential back-off on 429 / transient 5xx errors, plus a
Chrome-impersonating :class:`curl_cffi.requests.Session` for www.bls.gov
fetches — Akamai bot management there fingerprints the TLS handshake, so
plain httpx gets 403 regardless of headers (``release_dates/scraper.py``
holds the async counterpart). Other hosts stay on httpx.

If ``BLS_API_KEY`` is set in the environment, it is appended as a
``registrationkey`` query parameter on requests to ``bls.gov`` domains.
"""

from __future__ import annotations

import logging
import os
import time

import httpx
from curl_cffi import requests as curl_requests

logger = logging.getLogger(__name__)
USER_AGENT = 'Mozilla/5.0 (compatible; alt-nfp/0.1.0)'
DEFAULT_HEADERS = {
    'User-Agent': USER_AGENT,
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-us,en;q=0.5',
}
DEFAULT_TIMEOUT = 60.0
MAX_RETRIES = 8


def _bls_api_key() -> str:
    """Return ``BLS_API_KEY`` from the environment, or an empty string."""
    return os.environ.get('BLS_API_KEY', '')


def create_client(
    *,
    http2: bool = True,
    headers: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> httpx.Client:
    """Build an :class:`httpx.Client` with HTTP/2 and BLS-friendly headers.

    Parameters
    ----------
    http2 : bool
        Enable HTTP/2 negotiation (default ``True``).
    headers : dict or None
        Extra headers merged on top of :data:`DEFAULT_HEADERS`.
    timeout : float
        Per-request timeout in seconds.

    Returns
    -------
    httpx.Client
        Caller is responsible for closing it.
    """
    merged = {**DEFAULT_HEADERS}
    if headers:
        merged.update(headers)
    return httpx.Client(http2=http2, headers=merged, timeout=timeout)


def create_impersonating_session(
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> curl_requests.Session:
    """Sync HTTP session that passes www.bls.gov's TLS-fingerprint bot detection.

    Sync counterpart of ``release_dates.scraper.create_session``.
    ``impersonate='chrome'`` tracks the newest Chrome handshake curl_cffi
    supports and supplies matching default headers; spoofing our own
    User-Agent here would contradict the fingerprint, so we send none.

    Parameters
    ----------
    timeout : float
        Default per-request timeout in seconds.

    Returns
    -------
    curl_cffi.requests.Session
        Caller is responsible for closing it.
    """
    return curl_requests.Session(
        impersonate='chrome',
        allow_redirects=True,
        timeout=timeout,
    )


def get_with_retry(
    client: httpx.Client | curl_requests.Session,
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = MAX_RETRIES,
) -> httpx.Response | curl_requests.Response:
    """GET *url* with exponential back-off on 429, transient 5xx, and transport errors.

    Retries on three failure classes: HTTP 429 (rate-limit), HTTP 5xx
    (transient server error), and transport-level exceptions
    (``httpx.RequestError`` / ``curl_cffi.requests.exceptions.RequestException``
    — e.g. connection refused, timeout, TLS reset).  Non-retryable HTTP
    errors (4xx other than 429) raise immediately.

    If ``BLS_API_KEY`` is set and the URL contains ``bls.gov``, the key is
    appended as a ``registrationkey`` query parameter.

    Parameters
    ----------
    client : httpx.Client or curl_cffi.requests.Session
        An open client from :func:`create_client` or
        :func:`create_impersonating_session` (both expose the same
        ``get(url, timeout=..., params=...)`` surface).
    url : str
        Absolute URL to fetch.
    timeout : float
        Per-request timeout in seconds.
    max_retries : int
        Maximum retry attempts (across all failure classes).

    Returns
    -------
    httpx.Response or curl_cffi.requests.Response
        The successful response.

    Raises
    ------
    httpx.HTTPStatusError or curl_cffi.requests.exceptions.HTTPError
        After exhausting retries on HTTP errors, or immediately on a
        non-retryable status code.
    httpx.RequestError or curl_cffi.requests.exceptions.RequestException
        When all retries are exhausted on transport-level failures.
    """
    params: dict[str, str] = {}
    api_key = _bls_api_key()
    if api_key and 'bls.gov' in url:
        params['registrationkey'] = api_key

    _transport_errors = (
        httpx.RequestError,
        curl_requests.exceptions.RequestException,
    )

    for attempt in range(max_retries):
        try:
            r = client.get(url, timeout=timeout, params=params)
        except _transport_errors as exc:
            wait = min(2**attempt, 120)
            if attempt < max_retries - 1:
                logger.warning("    [transport error] retrying in %ss ... (%s)", wait, exc)
                time.sleep(wait)
                continue
            raise
        if r.status_code == 429 or r.status_code >= 500:
            wait = min(2**attempt, 120)
            logger.warning("    [%s] retrying in %ss ...", r.status_code, wait)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r
