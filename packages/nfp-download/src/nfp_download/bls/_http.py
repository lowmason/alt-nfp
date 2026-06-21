'''
Shared HTTP infrastructure for BLS data downloads.

Provides a unified client for:
- LABSTAT flat file downloads (tab-delimited)
- BLS Public Data API (JSON, v1/v2)
- QCEW CSV API

Ported from eco-stats (https://github.com/lowmason/eco-stats),
adapted for alt-nfp with added QCEW CSV support.
'''

from __future__ import annotations

import atexit
import csv
import io
import logging
import os
import shutil
import tempfile
import time
from datetime import date
from typing import Any

import httpx
import polars as pl

logger = logging.getLogger(__name__)

_UA_PRODUCT = 'alt-nfp/0.1.0'
# Non-personal placeholder (this repo is public); set BLS_CONTACT_EMAIL in a
# gitignored .env for real acquisition so BLS has a live contact.  Must not
# contain 'github.com' — Akamai bot management on download.bls.gov 403s any
# User-Agent carrying that token (URL or email domain alike).
_DEFAULT_CONTACT_EMAIL = 'alt-nfp@example.com'


def _build_user_agent() -> str:
    '''Build the User-Agent for ``download.bls.gov`` LABSTAT flat-file fetches.

    ``download.bls.gov`` sits behind Akamai bot management with two rules that
    a self-identifying UA trips (both verified empirically against the live
    endpoint):

    1. A User-Agent containing the ``github.com`` token — whether as a URL
       (``+https://github.com/...``) or an email domain
       (``foo@users.noreply.github.com``) — is 403'd on every file.  Other
       domains (``mac.com``, ``example.com``) pass.
    2. The heavy bulk-data files (``ce.data.0.AllCESSeries``, ~47 MB) require a
       contact email; a bare product token gets a 403 on those even though it
       succeeds on small overview files.

    So the UA is ``"alt-nfp/0.1.0 (<email>)"`` — product token plus a contact
    email, no ``github.com``.  The email defaults to a non-personal placeholder
    and can be overridden with the ``BLS_CONTACT_EMAIL`` environment variable.

    (This is unrelated to ``www.bls.gov``, which defends with TLS-handshake
    fingerprinting and is handled by
    :func:`nfp_download.client.create_impersonating_session`.)
    '''
    email = os.environ.get('BLS_CONTACT_EMAIL', '').strip() or _DEFAULT_CONTACT_EMAIL
    return f'{_UA_PRODUCT} ({email})'

# Programs whose survey reference period is the pay period including
# the 12th of the month.  Dates for these programs use day=12;
# all others default to day=1.
_REFERENCE_DAY_12_PROGRAMS = frozenset({'CE', 'EN'})


def _reference_day(program_prefix: str) -> int:
    '''Return the reference day-of-month for a BLS program prefix.'''
    return 12 if program_prefix.upper() in _REFERENCE_DAY_12_PROGRAMS else 1


def _period_to_month(period: str) -> int | None:
    '''
    Convert a BLS period code to a month number.

    Monthly periods ``M01``-``M12`` map to months 1-12.
    ``M13`` (annual average) returns ``None``.  Quarterly
    ``Q01``-``Q04`` map to the first month of the quarter.
    Semi-annual ``S01``-``S02`` map to months 1 and 7.
    Annual ``A01`` maps to month 1.

    Parameters
    ----------
    period : str
        BLS period string (e.g., ``'M01'``, ``'Q03'``).

    Returns
    -------
    int or None
        Month number (1-12) or ``None`` if not mappable.
    '''
    if not period or len(period) < 2:
        return None
    code = period[0].upper()
    try:
        num = int(period[1:])
    except (ValueError, TypeError):
        return None

    if code == 'M' and 1 <= num <= 12:
        return num
    if code == 'Q' and 1 <= num <= 4:
        return (num - 1) * 3 + 1
    if code == 'S' and 1 <= num <= 2:
        return (num - 1) * 6 + 1
    if code == 'A':
        return 1
    return None


class BLSHttpClient:
    '''
    Unified HTTP client for BLS data access.

    Supports three access patterns:
    - Flat file downloads from ``download.bls.gov/pub/time.series/``
    - JSON API queries to ``api.bls.gov/publicAPI/``
    - QCEW CSV API requests to ``data.bls.gov/cew/data/api/``

    Parameters
    ----------
    api_key : str or None
        BLS API registration key. Enables v2 API with higher rate limits
        and 20-year windows. Register at https://data.bls.gov/registrationEngine/
    cache_dir : str or None
        Local directory for cached downloads. ``None`` (the default) selects a
        per-process temp dir under the system tempdir, auto-removed at
        interpreter exit — container-safe on Bloomberg, which forbids writes
        outside ``/tmp``/S3 (plans/15 Tier C; the cache is rebuild scratch, so
        per-process is fine — re-download is accepted). Pass an explicit path to
        persist the cache across runs (e.g. for local dev iteration).
    cache_ttl : int
        Cache time-to-live in seconds. Defaults to 86400 (24 hours).
    '''

    FLAT_FILE_BASE = 'https://download.bls.gov/pub/time.series'
    API_BASE_V2 = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'
    API_BASE_V1 = 'https://api.bls.gov/publicAPI/v1/timeseries/data/'
    QCEW_CSV_BASE = 'https://data.bls.gov/cew/data/api'

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str | None = None,
        cache_ttl: int = 86_400,
    ) -> None:
        self.api_key = api_key
        if cache_dir is None:
            # Container-safe default: no CWD/./data writes on Bloomberg.  Use a
            # per-process temp dir, cleaned up at interpreter exit.
            cache_dir = tempfile.mkdtemp(prefix='altnfp-httpcache-')
            atexit.register(shutil.rmtree, cache_dir, ignore_errors=True)
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl
        self.session = httpx.Client(headers={'User-Agent': _build_user_agent()}, timeout=60.0)
        self.base_url = self.API_BASE_V2 if api_key else self.API_BASE_V1

    # ------------------------------------------------------------------
    # Flat file methods
    # ------------------------------------------------------------------

    def get_flat_file(
        self,
        prefix: str,
        filename: str,
    ) -> list[dict[str, str]]:
        '''
        Download and parse a tab-delimited flat file from BLS LABSTAT.

        Parameters
        ----------
        prefix : str
            Two-letter program code (e.g., ``'CE'``).
        filename : str
            Full filename (e.g., ``'ce.data.0.AllCESSeries'``).

        Returns
        -------
        list[dict[str, str]]
            List of dicts, one per row, keyed by column header.
        '''
        url = f'{self.FLAT_FILE_BASE}/{prefix.lower()}/{filename}'
        return self._download_and_parse_tsv(url, filename)

    def get_mapping(
        self,
        prefix: str,
        mapping_name: str,
    ) -> list[dict[str, str]]:
        '''
        Download and parse a mapping/lookup file.

        Parameters
        ----------
        prefix : str
            Two-letter program code (e.g., ``'CE'``).
        mapping_name : str
            Mapping file name without the prefix dot (e.g., ``'area'``).

        Returns
        -------
        list[dict[str, str]]
            List of dicts, one per row, keyed by column header.
        '''
        filename = f'{prefix.lower()}.{mapping_name}'
        url = f'{self.FLAT_FILE_BASE}/{prefix.lower()}/{filename}'
        return self._download_and_parse_tsv(url, filename)

    def get_data(
        self,
        prefix: str,
        file_suffix: str = '0.Current',
    ) -> list[dict[str, str]]:
        '''
        Download and parse a data file.

        BLS data files are named like ``ce.data.0.AllCESSeries`` or
        ``sm.data.0.Current``.

        Parameters
        ----------
        prefix : str
            Two-letter program code.
        file_suffix : str
            The portion of the filename after ``xx.data.``
            (e.g., ``'0.Current'``, ``'0.AllCESSeries'``).

        Returns
        -------
        list[dict[str, str]]
            List of dicts with keys like ``series_id``, ``year``,
            ``period``, ``value``, ``footnote_codes``.
        '''
        filename = f'{prefix.lower()}.data.{file_suffix}'
        url = f'{self.FLAT_FILE_BASE}/{prefix.lower()}/{filename}'
        return self._download_and_parse_tsv(url, filename)

    def get_series_list(
        self,
        prefix: str,
        **filters: str,
    ) -> list[dict[str, str]]:
        '''
        Download and parse the master series file, optionally filtering.

        Parameters
        ----------
        prefix : str
            Two-letter program code.
        **filters : str
            Column name -> value pairs used to filter rows.

        Returns
        -------
        list[dict[str, str]]
            List of dicts, one per matching series.
        '''
        rows = self.get_mapping(prefix, 'series')
        if not filters:
            return rows
        return [
            row
            for row in rows
            if all(row.get(k, '').strip() == v for k, v in filters.items())
        ]

    # ------------------------------------------------------------------
    # JSON API method
    # ------------------------------------------------------------------

    def get_series(
        self,
        series_ids: list[str],
        start_year: str | None = None,
        end_year: str | None = None,
    ) -> pl.DataFrame:
        '''
        Get time series data via the BLS Public Data API.

        The API accepts up to 50 series per request and supports 10-year
        windows (v1) or 20-year windows (v2 with API key).

        Parameters
        ----------
        series_ids : list[str]
            List of BLS series IDs.
        start_year : str or None
            Start year (format: ``'YYYY'``).
        end_year : str or None
            End year (format: ``'YYYY'``).

        Returns
        -------
        pl.DataFrame
            DataFrame with columns ``series_id``, ``date``, ``year``,
            ``period``, ``period_name``, ``value``.
        '''
        payload: dict[str, Any] = {'seriesid': series_ids}

        if self.api_key:
            payload['registrationkey'] = self.api_key
        if start_year:
            payload['startyear'] = start_year
        if end_year:
            payload['endyear'] = end_year

        headers = {'Content-Type': 'application/json'}
        response = self.session.post(
            self.base_url,
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        return self._parse_api_response(response.json())

    # ------------------------------------------------------------------
    # QCEW CSV API method
    # ------------------------------------------------------------------

    def get_qcew_csv(
        self,
        year: int,
        quarter: int,
        slice_code: str,
        slice_type: str = 'industry',
    ) -> pl.DataFrame:
        '''
        Download QCEW data from the CSV API.

        Parameters
        ----------
        year : int
            Reference year.
        quarter : int
            Reference quarter (1-4).
        slice_code : str
            Slice code: industry code (e.g., ``'10'``, ``'1011'``), area code
            (e.g., ``'US000'``, ``'01000'``), or size code (e.g., ``'0'``).
        slice_type : str
            One of ``'industry'`` (default), ``'area'``, or ``'size'``.

        Returns
        -------
        pl.DataFrame
            Raw QCEW DataFrame parsed from the CSV response.
        '''
        if slice_type not in ('industry', 'area', 'size'):
            raise ValueError(
                f'slice_type must be industry, area, or size; got {slice_type!r}'
            )

        cache_key = f'qcew_{year}_{quarter}_{slice_type}_{slice_code}.csv'
        cache_path = self._cache_path(cache_key)

        # Force text columns to Utf8 so values like "C1010" (CSA) and "US000" parse
        qcew_text_cols = [
            'area_fips', 'own_code', 'industry_code', 'agglvl_code', 'size_code',
            'year', 'qtr', 'disclosure_code', 'lq_disclosure_code', 'oty_disclosure_code',
        ]
        schema_overrides = dict.fromkeys(qcew_text_cols, pl.Utf8)

        if self._is_cache_valid(cache_path):
            return pl.read_csv(cache_path, schema_overrides=schema_overrides)

        url = f'{self.QCEW_CSV_BASE}/{year}/{quarter}/{slice_type}/{slice_code}.csv'
        response = self.session.get(url, timeout=60)
        response.raise_for_status()

        os.makedirs(self.cache_dir, exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as fh:
            fh.write(response.text)

        return pl.read_csv(io.StringIO(response.text), schema_overrides=schema_overrides)

    # ------------------------------------------------------------------
    # Internal: caching
    # ------------------------------------------------------------------

    def _cache_path(self, filename: str) -> str:
        '''Return the local cache file path for a given filename.'''
        safe_name = filename.replace('/', '_').replace('\\', '_')
        return os.path.join(self.cache_dir, safe_name)

    def _is_cache_valid(self, path: str) -> bool:
        '''Check whether a cached file exists and is within TTL.'''
        if not os.path.exists(path):
            return False
        age = time.time() - os.path.getmtime(path)
        return age < self.cache_ttl

    # ------------------------------------------------------------------
    # Internal: download and parse
    # ------------------------------------------------------------------

    def _download_and_parse_tsv(
        self,
        url: str,
        filename: str,
    ) -> list[dict[str, str]]:
        '''
        Download a file (or load from cache), then parse as
        tab-delimited text.

        Returns
        -------
        list[dict[str, str]]
            List of dicts keyed by the header row.
        '''
        cache_path = self._cache_path(filename)

        if self._is_cache_valid(cache_path):
            with open(cache_path, encoding='utf-8') as fh:
                text = fh.read()
        else:
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            text = response.text
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as fh:
                fh.write(text)

        return self._parse_tsv(text)

    @staticmethod
    def _parse_tsv(text: str) -> list[dict[str, str]]:
        '''
        Parse tab-separated text with a header row into a list of dicts.

        BLS flat files use tab delimiters and often have trailing
        whitespace in fields, which we strip.
        '''
        reader = csv.DictReader(io.StringIO(text), delimiter='\t')
        rows: list[dict[str, str]] = []
        for row in reader:
            cleaned = {
                k.strip(): v.strip() if v else ''
                for k, v in row.items()
                if k is not None
            }
            rows.append(cleaned)
        return rows

    @staticmethod
    def _parse_api_response(raw: dict[str, Any]) -> pl.DataFrame:
        '''
        Parse a BLS JSON API response into a Polars DataFrame.

        Extracts rows from ``Results.series[*].data[*]``, adds a
        ``series_id`` column, constructs a ``date`` from year/period,
        and casts ``value`` to Float64.

        Parameters
        ----------
        raw : dict
            The full JSON dict returned by the BLS API.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns ``series_id``, ``date``, ``year``,
            ``period``, ``period_name``, ``value``.

        Raises
        ------
        ValueError
            If the API response indicates failure.
        '''
        schema = {
            'series_id': pl.Utf8,
            'date': pl.Date,
            'year': pl.Int64,
            'period': pl.Utf8,
            'period_name': pl.Utf8,
            'value': pl.Float64,
        }

        status = raw.get('status', '')
        if status != 'REQUEST_SUCCEEDED':
            message = raw.get('message', [])
            raise ValueError(
                f'BLS API request failed (status={status!r}): {message}'
            )

        rows: list[dict[str, Any]] = []
        for series in raw.get('Results', {}).get('series', []):
            sid = series.get('seriesID', '')
            day = _reference_day(sid[:2]) if len(sid) >= 2 else 1
            for obs in series.get('data', []):
                year_str = obs.get('year', '')
                period = obs.get('period', '')

                try:
                    year_int = int(year_str)
                except (ValueError, TypeError):
                    year_int = None

                month = _period_to_month(period)
                obs_date = (
                    date(year_int, month, day)
                    if year_int is not None and month is not None
                    else None
                )

                try:
                    value = float(obs.get('value', ''))
                except (ValueError, TypeError):
                    value = None

                rows.append(
                    {
                        'series_id': sid,
                        'date': obs_date,
                        'year': year_int,
                        'period': period,
                        'period_name': obs.get('periodName', ''),
                        'value': value,
                    }
                )

        return pl.DataFrame(rows, schema=schema).sort('date')

    @staticmethod
    def _add_date_column(df: pl.DataFrame, day: int = 1) -> pl.DataFrame:
        '''
        Derive a ``date`` column from ``year`` and ``period`` columns.

        Uses native Polars expressions for vectorised operation.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame with ``year`` (Int64) and ``period`` (Utf8) columns.
        day : int
            Day-of-month to use. CES and QCEW use 12; most others use 1.

        Returns
        -------
        pl.DataFrame
            Input DataFrame with an added ``date`` column.
        '''
        period_code = pl.col('period').str.slice(0, 1)
        period_num = pl.col('period').str.slice(1).cast(pl.Int64, strict=False)

        month = (
            pl.when((period_code == 'M') & period_num.is_between(1, 12))
            .then(period_num)
            .when((period_code == 'Q') & period_num.is_between(1, 4))
            .then((period_num - 1) * 3 + 1)
            .when((period_code == 'S') & period_num.is_between(1, 2))
            .then((period_num - 1) * 6 + 1)
            .when(period_code == 'A')
            .then(1)
            .otherwise(None)
        )

        return df.with_columns(pl.date(pl.col('year'), month, day).alias('date'))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        '''Close the HTTP session.'''
        self.session.close()

    def __enter__(self) -> BLSHttpClient:
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        self.close()
