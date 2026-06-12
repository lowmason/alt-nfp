"""FRED API client for downloading economic time series.

Provides a single ``fetch_fred_series`` function that retrieves observation
data from the FRED JSON API (``api.stlouisfed.org``).  Retry logic mirrors
the pattern in :mod:`alt_nfp.vintages.processing.sae_states`.

Requires ``FRED_API_KEY`` in the environment (loaded via ``python-dotenv``).
"""

from __future__ import annotations

import logging
import os
import time

import httpx
import polars as pl

logger = logging.getLogger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred"


def _get_api_key(api_key: str | None = None) -> str:
    """Resolve FRED API key from argument or environment."""
    key = api_key or os.environ.get("FRED_API_KEY", "")
    if not key:
        raise RuntimeError(
            "FRED_API_KEY is required. Set it in the environment or pass api_key=."
        )
    return key


def _request_with_retry(
    client: httpx.Client,
    url: str,
    params: dict,
    timeout: float = 30.0,
    max_retries: int = 6,
) -> httpx.Response:
    """GET with exponential backoff on 429, transient 5xx, and timeouts."""
    r: httpx.Response | None = None
    for attempt in range(max_retries):
        try:
            r = client.get(url, params=params, timeout=timeout)
        except (
            httpx.TimeoutException,
            httpx.RemoteProtocolError,
            httpx.NetworkError,
        ) as exc:
            wait = min(2**attempt, 60)
            logger.warning("[%s] retrying in %ds ...", type(exc).__name__, wait)
            time.sleep(wait)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            wait = min(2**attempt, 60)
            logger.warning("[%d] retrying in %ds ...", r.status_code, wait)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r
    if r is not None:
        r.raise_for_status()
        return r
    raise httpx.ConnectError("All retries exhausted with no response")


def fetch_fred_series(
    series_id: str,
    *,
    start_date: str = "2000-01-01",
    api_key: str | None = None,
) -> pl.DataFrame:
    """Fetch a single FRED series and return ``(ref_date, value)``.

    Parameters
    ----------
    series_id : str
        FRED series identifier (e.g. ``'ICNSA'``, ``'JTSJOL'``).
    start_date : str
        Observation start date in ``YYYY-MM-DD`` format.
    api_key : str or None
        FRED API key.  Falls back to ``FRED_API_KEY`` env var.

    Returns
    -------
    pl.DataFrame
        Two-column frame with ``ref_date`` (Date) and ``value`` (Float64),
        sorted by date ascending.  Rows with missing values (FRED's ``"."``
        sentinel) are excluded.
    """
    key = _get_api_key(api_key)

    with httpx.Client(http2=True, follow_redirects=True) as client:
        r = _request_with_retry(
            client,
            f"{FRED_BASE}/series/observations",
            params={
                "series_id": series_id,
                "api_key": key,
                "file_type": "json",
                "observation_start": start_date,
            },
        )

    payload = r.json()
    observations = payload.get("observations", [])
    if not observations:
        logger.warning("No observations returned for %s", series_id)
        return pl.DataFrame(schema={"ref_date": pl.Date, "value": pl.Float64})

    rows = []
    for obs in observations:
        val_str = obs.get("value", ".")
        if val_str == ".":
            continue
        try:
            val = float(val_str)
        except (ValueError, TypeError):
            continue
        rows.append({"ref_date": obs["date"], "value": val})

    if not rows:
        return pl.DataFrame(schema={"ref_date": pl.Date, "value": pl.Float64})

    df = (
        pl.DataFrame(rows)
        .with_columns(pl.col("ref_date").str.to_date("%Y-%m-%d"))
        .sort("ref_date")
    )
    return df
