"""FRED/ALFRED real-time vintage access for national CES series.

Resolution tables map each vintage-store ``(industry_type, industry_code)``
key (NAICS-coded sectors) to its ALFRED series id. SA aggregates have NO
systematic ``CES…01`` archive, so they resolve to FRED friendly aliases
(PAYEMS, USPRIV, MANEMP, …); NSA resolves to systematic ``CEU…01`` except
``00`` (``PAYNSA``). All ids verified live 2026-06-27/28.

This module imports no ``nfp_*`` package (download-layer boundary).
"""
from __future__ import annotations

import logging
import time

import httpx
import polars as pl

logger = logging.getLogger(__name__)

ALFRED_BASE = "https://api.stlouisfed.org/fred"

# (industry_type, industry_code NAICS) -> ALFRED SA series id.
CES_SERIES_SA: dict[tuple[str, str], str] = {
    ("total", "00"): "PAYEMS",
    ("total", "05"): "USPRIV",
    ("domain", "06"): "USGOOD",
    ("domain", "08"): "CES0800000001",
    ("supersector", "10"): "USMINE",
    ("supersector", "20"): "USCONS",
    ("supersector", "30"): "MANEMP",
    ("supersector", "40"): "USTPU",
    ("supersector", "50"): "USINFO",
    ("supersector", "55"): "USFIRE",
    ("supersector", "60"): "USPBS",
    ("supersector", "65"): "USEHS",
    ("supersector", "70"): "USLAH",
    ("supersector", "80"): "USSERV",
    ("sector", "21"): "CES1021000001",
    ("sector", "22"): "CES4422000001",
    ("sector", "31"): "DMANEMP",
    ("sector", "32"): "NDMANEMP",
    ("sector", "42"): "USWTRADE",
    ("sector", "44"): "USTRADE",
    ("sector", "48"): "CES4300000001",
    ("sector", "52"): "CES5552000001",
    ("sector", "53"): "CES5553000001",
    ("sector", "54"): "CES6054000001",
    ("sector", "55"): "CES6055000001",
    ("sector", "56"): "CES6056000001",
    ("sector", "61"): "CES6561000001",
    ("sector", "62"): "CES6562000001",
    ("sector", "71"): "CES7071000001",
    ("sector", "72"): "CES7072000001",
}

# NSA: 29/30 systematic CEU{8digit}01; only 00 -> PAYNSA.
CES_SERIES_NSA: dict[tuple[str, str], str] = {
    ("total", "00"): "PAYNSA",
    ("total", "05"): "CEU0500000001",
    ("domain", "06"): "CEU0600000001",
    ("domain", "08"): "CEU0800000001",
    ("supersector", "10"): "CEU1000000001",
    ("supersector", "20"): "CEU2000000001",
    ("supersector", "30"): "CEU3000000001",
    ("supersector", "40"): "CEU4000000001",
    ("supersector", "50"): "CEU5000000001",
    ("supersector", "55"): "CEU5500000001",
    ("supersector", "60"): "CEU6000000001",
    ("supersector", "65"): "CEU6500000001",
    ("supersector", "70"): "CEU7000000001",
    ("supersector", "80"): "CEU8000000001",
    ("sector", "21"): "CEU1021000001",
    ("sector", "22"): "CEU4422000001",
    ("sector", "31"): "CEU3100000001",
    ("sector", "32"): "CEU3200000001",
    ("sector", "42"): "CEU4142000001",
    ("sector", "44"): "CEU4200000001",
    ("sector", "48"): "CEU4300000001",
    ("sector", "52"): "CEU5552000001",
    ("sector", "53"): "CEU5553000001",
    ("sector", "54"): "CEU6054000001",
    ("sector", "55"): "CEU6055000001",
    ("sector", "56"): "CEU6056000001",
    ("sector", "61"): "CEU6561000001",
    ("sector", "62"): "CEU6562000001",
    ("sector", "71"): "CEU7071000001",
    ("sector", "72"): "CEU7072000001",
}


# Expected /series title substring per key — the title-verify guard that caught
# the USSERV(=Other Services)->08 mis-map during design.
_EXPECTED_TITLE_SUBSTR: dict[tuple[str, str], str] = {
    ("total", "00"): "Total Nonfarm",
    ("total", "05"): "Total Private",
    ("domain", "06"): "Goods-Producing",
    ("domain", "08"): "Private Service-Providing",
    ("supersector", "10"): "Mining and Logging",
    ("supersector", "20"): "Construction",
    ("supersector", "30"): "Manufacturing",
    ("supersector", "40"): "Trade, Transportation",
    ("supersector", "50"): "Information",
    ("supersector", "55"): "Financial Activities",
    ("supersector", "60"): "Professional and Business",
    ("supersector", "65"): "Education and Health",
    ("supersector", "70"): "Leisure and Hospitality",
    ("supersector", "80"): "Other Services",
    ("sector", "21"): "Mining",
    ("sector", "22"): "Utilities",
    ("sector", "31"): "Durable Goods",
    ("sector", "32"): "Nondurable Goods",
    ("sector", "42"): "Wholesale Trade",
    ("sector", "44"): "Retail Trade",
    ("sector", "48"): "Transportation and Warehousing",
    ("sector", "52"): "Finance and Insurance",
    ("sector", "53"): "Real Estate",
    ("sector", "54"): "Professional, Scientific",
    ("sector", "55"): "Management of Companies",
    ("sector", "56"): "Administrative",
    ("sector", "61"): "Educational Services",
    ("sector", "62"): "Health Care",
    ("sector", "71"): "Arts, Entertainment",
    ("sector", "72"): "Accommodation and Food",
}


def resolve_series_id(industry_type: str, industry_code: str, *, sa: bool) -> str:
    """Return the ALFRED series id for a vintage-store industry key.

    Parameters
    ----------
    industry_type : str
        One of ``'total'``, ``'domain'``, ``'supersector'``, ``'sector'``.
    industry_code : str
        Store NAICS-coded industry code (e.g. ``'42'`` for wholesale).
    sa : bool
        ``True`` for seasonally adjusted (CES/alias), ``False`` for NSA (CEU/PAYNSA).

    Returns
    -------
    str
        The resolved ALFRED series id.

    Raises
    ------
    KeyError
        If ``(industry_type, industry_code)`` is not a stored CES key.
    """
    table = CES_SERIES_SA if sa else CES_SERIES_NSA
    return table[(industry_type, industry_code)]


def _request_json(client: httpx.Client, path: str, params: dict, max_retries: int = 6) -> dict:
    """GET a FRED JSON endpoint with exponential backoff on 429/5xx/timeouts."""
    params = {**params, "file_type": "json"}
    last: httpx.Response | None = None
    for attempt in range(max_retries):
        try:
            last = client.get(f"{ALFRED_BASE}/{path}", params=params, timeout=60.0)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            time.sleep(min(2**attempt, 60))
            logger.warning("[%s] retrying ...", type(exc).__name__)
            continue
        if last.status_code == 429 or last.status_code >= 500:
            time.sleep(min(2**attempt, 60))
            continue
        last.raise_for_status()
        return last.json()
    assert last is not None
    last.raise_for_status()
    return last.json()


def get_vintage_dates(
    client: httpx.Client, series_id: str, *, api_key: str, start: str | None = None
) -> list[str]:
    """Return ALFRED vintage date strings for *series_id* (optionally from *start*).

    Parameters
    ----------
    client : httpx.Client
        Shared HTTP client.
    series_id : str
        FRED/ALFRED series identifier.
    api_key : str
        FRED API key.
    start : str, optional
        If provided, filter vintage dates to those >= *start* (inclusive).

    Returns
    -------
    list[str]
        Vintage date strings in YYYY-MM-DD format.
    """
    payload = _request_json(
        client, "series/vintagedates", {"series_id": series_id, "api_key": api_key}
    )
    vds = payload.get("vintage_dates", [])
    return [v for v in vds if start is None or v >= start]


def _matrix_from_observations(observations: list[dict]) -> pl.DataFrame:
    """Reshape ``output_type=2`` wide observations to long ``(ref_date, vintage_date, value)``.

    Wide columns are named ``{SERIES_ID}_{YYYYMMDD}``; the ``"."`` sentinel and
    empty strings are dropped (a ref month is absent from vintages before its
    first release).

    Parameters
    ----------
    observations : list[dict]
        List of observation dicts from FRED API ``output_type=2`` response.

    Returns
    -------
    pl.DataFrame
        Long frame with columns ``(ref_date, vintage_date, value)`` (Date, Date, Float64).
    """
    if not observations:
        return pl.DataFrame(
            schema={"ref_date": pl.Date, "vintage_date": pl.Date, "value": pl.Float64}
        )
    wide = pl.from_dicts(observations)
    vcols = [c for c in wide.columns if c != "date"]
    return (
        wide.unpivot(index="date", on=vcols, variable_name="vcol", value_name="value")
        .with_columns(pl.col("value").cast(pl.Utf8).str.strip_chars())
        .filter((pl.col("value") != ".") & (pl.col("value") != "")
                & pl.col("value").is_not_null())
        .with_columns(
            pl.col("date").str.to_date().alias("ref_date"),
            pl.col("vcol").str.extract(r"(\d{8})$").str.to_date("%Y%m%d").alias("vintage_date"),
            pl.col("value").cast(pl.Float64),
        )
        .select("ref_date", "vintage_date", "value")
    )


def fetch_vintage_matrix(
    client: httpx.Client,
    series_id: str,
    *,
    api_key: str,
    vintage_dates: list[str],
    observation_start: str,
    chunk_size: int = 100,
) -> pl.DataFrame:
    """Fetch ``output_type=2`` observations and return a long vintage matrix.

    Chunks ``vintage_dates`` (the API caps the request size), concatenates the
    per-chunk long frames.

    Parameters
    ----------
    client : httpx.Client
        Shared HTTP client.
    series_id : str
        FRED/ALFRED series identifier.
    api_key : str
        FRED API key.
    vintage_dates : list[str]
        Vintage dates to request (YYYY-MM-DD format).
    observation_start : str
        Earliest reference date to fetch (YYYY-MM-DD format).
    chunk_size : int, optional
        Number of vintage dates per request (default 100).

    Returns
    -------
    pl.DataFrame
        Long frame with columns ``(ref_date, vintage_date, value)`` (Date, Date, Float64).
    """
    frames: list[pl.DataFrame] = []
    for i in range(0, len(vintage_dates), chunk_size):
        chunk = vintage_dates[i : i + chunk_size]
        payload = _request_json(
            client,
            "series/observations",
            {
                "series_id": series_id,
                "api_key": api_key,
                "output_type": 2,
                "vintage_dates": ",".join(chunk),
                "observation_start": observation_start,
            },
        )
        obs = payload.get("observations", [])
        if obs:
            frames.append(_matrix_from_observations(obs))
    if not frames:
        return pl.DataFrame(
            schema={"ref_date": pl.Date, "vintage_date": pl.Date, "value": pl.Float64}
        )
    return pl.concat(frames, how="vertical")


def verify_series_concept(
    client: httpx.Client, series_id: str, *, api_key: str, sa: bool
) -> tuple[str, bool]:
    """Return ``(title, seasonal_ok)`` from the FRED ``/series`` metadata.

    ``seasonal_ok`` is ``True`` iff the series' ``seasonal_adjustment_short``
    matches the requested *sa* flag. The caller cross-checks *title* against the
    expected concept (the title-verify gate).

    Parameters
    ----------
    client : httpx.Client
        Shared HTTP client.
    series_id : str
        FRED/ALFRED series identifier.
    api_key : str
        FRED API key.
    sa : bool
        Expected seasonal-adjustment flag (True = SA, False = NSA).

    Returns
    -------
    tuple[str, bool]
        ``(title, seasonal_ok)``: title from metadata, and whether the
        seasonal_adjustment_short matches the requested *sa* flag.
    """
    payload = _request_json(client, "series", {"series_id": series_id, "api_key": api_key})
    meta = payload["seriess"][0]
    seasonal_ok = meta["seasonal_adjustment_short"] == ("SA" if sa else "NSA")
    return meta["title"], seasonal_ok


def _title_matches(industry_type: str, industry_code: str, title: str) -> bool:
    """True iff *title* contains the expected concept substring for the key.

    Parameters
    ----------
    industry_type : str
        One of ``'total'``, ``'domain'``, ``'supersector'``, ``'sector'``.
    industry_code : str
        Store NAICS-coded industry code (e.g. ``'42'`` for wholesale).
    title : str
        FRED series title to check.

    Returns
    -------
    bool
        True iff the title contains the expected concept substring for this key.
    """
    return _EXPECTED_TITLE_SUBSTR[(industry_type, industry_code)] in title


def verify_ces_series(
    client: httpx.Client,
    industry_type: str,
    industry_code: str,
    *,
    sa: bool,
    api_key: str,
) -> tuple[str, bool]:
    """Title-verify the resolved CES series for a store key (public boundary fn).

    Resolves ``(industry_type, industry_code, sa)`` to its ALFRED id, fetches the
    ``/series`` metadata, and returns ``(title, ok)`` where *ok* is ``True`` iff
    BOTH the seasonal-adjustment flag matches and the title contains the expected
    concept substring. Keeps all title-verify logic in the download layer so
    ``nfp_ingest`` never imports a download-private name.

    Parameters
    ----------
    client : httpx.Client
        Shared HTTP client.
    industry_type, industry_code : str
        Vintage-store key (NAICS-coded sectors).
    sa : bool
        Seasonal-adjustment flag.
    api_key : str
        FRED API key.

    Returns
    -------
    tuple[str, bool]
        ``(title, ok)``.
    """
    series_id = resolve_series_id(industry_type, industry_code, sa=sa)
    title, seasonal_ok = verify_series_concept(client, series_id, api_key=api_key, sa=sa)
    return title, seasonal_ok and _title_matches(industry_type, industry_code, title)
