"""Shared HTTP client with retry logic for vintage pipeline requests.

Provides a pre-configured :class:`httpx.Client` with HTTP/2, browser-like
headers, and exponential back-off on 429 / transient 5xx errors.

If ``BLS_API_KEY`` is set in the environment, it is appended as a
``registrationkey`` query parameter on requests to ``bls.gov`` domains.
"""

from __future__ import annotations

import logging
import os
import time

import httpx

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


def get_with_retry(
    client: httpx.Client,
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = MAX_RETRIES,
) -> httpx.Response:
    """GET *url* with exponential back-off on 429 and transient 5xx errors.

    If ``BLS_API_KEY`` is set and the URL contains ``bls.gov``, the key is
    appended as a ``registrationkey`` query parameter.

    Parameters
    ----------
    client : httpx.Client
        An open ``httpx.Client``.
    url : str
        Absolute URL to fetch.
    timeout : float
        Per-request timeout in seconds.
    max_retries : int
        Maximum retry attempts.

    Returns
    -------
    httpx.Response
        The successful response.

    Raises
    ------
    httpx.HTTPStatusError
        After exhausting retries or on a non-retryable error.
    """
    params: dict[str, str] = {}
    api_key = _bls_api_key()
    if api_key and 'bls.gov' in url:
        params['registrationkey'] = api_key

    for attempt in range(max_retries):
        r = client.get(url, timeout=timeout, params=params)
        if r.status_code == 429 or r.status_code >= 500:
            wait = min(2**attempt, 120)
            logger.warning("    [%s] retrying in %ss ...", r.status_code, wait)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r
