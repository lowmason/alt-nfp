"""Tests for nfp_download.bls._http — HTTP client and parsing utilities."""

import pytest
from nfp_download.bls._http import (
    BLSHttpClient,
    _build_user_agent,
    _period_to_month,
    _reference_day,
)


class TestParseTsv:
    """Tests for BLSHttpClient._parse_tsv."""

    def test_basic(self):
        text = "series_id\tyear\tperiod\tvalue\nCES00\t2023\tM01\t100.0\n"
        rows = BLSHttpClient._parse_tsv(text)
        assert len(rows) == 1
        assert rows[0]['series_id'] == 'CES00'
        assert rows[0]['year'] == '2023'
        assert rows[0]['period'] == 'M01'
        assert rows[0]['value'] == '100.0'

    def test_strips_whitespace(self):
        text = "series_id \t year \t value \n  CES00  \t  2023  \t  100.0  \n"
        rows = BLSHttpClient._parse_tsv(text)
        assert len(rows) == 1
        assert rows[0]['series_id'] == 'CES00'
        assert rows[0]['year'] == '2023'
        assert rows[0]['value'] == '100.0'

    def test_empty(self):
        rows = BLSHttpClient._parse_tsv('')
        assert rows == []

    def test_header_only(self):
        rows = BLSHttpClient._parse_tsv('col1\tcol2\n')
        assert rows == []

    def test_multiple_rows(self):
        text = "id\tval\nA\t1\nB\t2\nC\t3\n"
        rows = BLSHttpClient._parse_tsv(text)
        assert len(rows) == 3
        assert rows[0]['id'] == 'A'
        assert rows[2]['val'] == '3'

    def test_none_key_filtered(self):
        """Rows with None keys (e.g., from extra tabs) are filtered."""
        text = "col1\tcol2\t\nA\tB\tC\n"
        rows = BLSHttpClient._parse_tsv(text)
        assert len(rows) == 1
        # The None key from the empty header should be filtered out
        assert None not in rows[0]

    def test_empty_values(self):
        text = "col1\tcol2\nA\t\n"
        rows = BLSHttpClient._parse_tsv(text)
        assert len(rows) == 1
        assert rows[0]['col2'] == ''


class TestCachePath:
    """Tests for cache path sanitization."""

    def test_normal_filename(self):
        client = BLSHttpClient(cache_dir='/tmp/test_cache')
        path = client._cache_path('ce.data.0.Current')
        assert path == '/tmp/test_cache/ce.data.0.Current'

    def test_slashes_sanitized(self):
        client = BLSHttpClient(cache_dir='/tmp/test_cache')
        path = client._cache_path('some/nested/file.txt')
        assert '/' not in path.split('test_cache/')[-1]
        assert 'some_nested_file.txt' in path

    def test_backslashes_sanitized(self):
        client = BLSHttpClient(cache_dir='/tmp/test_cache')
        path = client._cache_path('some\\nested\\file.txt')
        assert '\\' not in path.split('test_cache/')[-1]


class TestPeriodToMonth:
    """Tests for _period_to_month."""

    def test_monthly(self):
        assert _period_to_month('M01') == 1
        assert _period_to_month('M06') == 6
        assert _period_to_month('M12') == 12

    def test_m13_annual_average(self):
        assert _period_to_month('M13') is None

    def test_quarterly(self):
        assert _period_to_month('Q01') == 1
        assert _period_to_month('Q02') == 4
        assert _period_to_month('Q03') == 7
        assert _period_to_month('Q04') == 10

    def test_semiannual(self):
        assert _period_to_month('S01') == 1
        assert _period_to_month('S02') == 7

    def test_annual(self):
        assert _period_to_month('A01') == 1

    def test_invalid(self):
        assert _period_to_month('') is None
        assert _period_to_month('X') is None
        assert _period_to_month('M99') is None
        assert _period_to_month(None) is None


class TestReferenceDay:
    """Tests for _reference_day."""

    def test_ce_uses_12(self):
        assert _reference_day('CE') == 12

    def test_en_uses_12(self):
        assert _reference_day('EN') == 12

    def test_sm_uses_1(self):
        assert _reference_day('SM') == 1

    def test_other_uses_1(self):
        assert _reference_day('CU') == 1


class TestParseApiResponse:
    """Tests for BLSHttpClient._parse_api_response."""

    def test_successful_response(self):
        raw = {
            'status': 'REQUEST_SUCCEEDED',
            'Results': {
                'series': [
                    {
                        'seriesID': 'CES0000000001',
                        'data': [
                            {
                                'year': '2023',
                                'period': 'M01',
                                'periodName': 'January',
                                'value': '155000',
                            },
                            {
                                'year': '2023',
                                'period': 'M02',
                                'periodName': 'February',
                                'value': '155500',
                            },
                        ],
                    }
                ]
            },
        }
        df = BLSHttpClient._parse_api_response(raw)
        assert len(df) == 2
        assert 'series_id' in df.columns
        assert 'date' in df.columns
        assert 'value' in df.columns
        assert df['series_id'][0] == 'CES0000000001'

    def test_failed_response_raises(self):
        raw = {
            'status': 'REQUEST_NOT_PROCESSED',
            'message': ['Too many requests'],
        }
        with pytest.raises(ValueError, match='BLS API request failed'):
            BLSHttpClient._parse_api_response(raw)

    def test_empty_results(self):
        raw = {
            'status': 'REQUEST_SUCCEEDED',
            'Results': {'series': []},
        }
        df = BLSHttpClient._parse_api_response(raw)
        assert len(df) == 0


class TestUserAgent:
    """Tests for _build_user_agent.

    download.bls.gov's Akamai bot management 403s any User-Agent carrying the
    ``github.com`` token (URL or email domain alike), and the heavy bulk-data
    files additionally require a contact email — a bare product token gets a
    403 on them.  The UA must therefore carry an email and no ``github.com``.
    The contact defaults to a non-personal placeholder (public repo) and is
    overridable via ``BLS_CONTACT_EMAIL``.
    """

    def test_default_carries_email_and_no_blocked_token(self, monkeypatch):
        monkeypatch.delenv('BLS_CONTACT_EMAIL', raising=False)
        ua = _build_user_agent()
        assert 'alt-nfp' in ua
        assert '@' in ua  # a contact email is present
        assert 'http' not in ua.lower()  # no URL
        assert 'github' not in ua.lower()  # Akamai 403s the github.com token
        assert ua == 'alt-nfp/0.1.0 (alt-nfp@example.com)'  # exact shipped default

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv('BLS_CONTACT_EMAIL', 'me@example.org')
        ua = _build_user_agent()
        assert 'me@example.org' in ua
        assert 'http' not in ua.lower()

    def test_blank_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv('BLS_CONTACT_EMAIL', '   ')
        ua = _build_user_agent()
        assert ua == 'alt-nfp/0.1.0 (alt-nfp@example.com)'

    def test_client_session_uses_built_ua(self, monkeypatch):
        monkeypatch.setenv('BLS_CONTACT_EMAIL', 'me@example.org')
        with BLSHttpClient() as client:
            assert client.session.headers.get('user-agent') == _build_user_agent()


class TestContextManager:
    """Tests for context manager protocol."""

    def test_context_manager(self):
        with BLSHttpClient() as client:
            assert client.session is not None


class TestCacheDirDefault:
    """plans/15 Tier C: the default HTTP cache must not write under the CWD.

    Bloomberg's container forbids writes outside /tmp and S3, so an unset
    ``cache_dir`` must resolve to a per-process tempdir, never the old
    CWD-relative ``.cache/bls``.  An explicit ``cache_dir=`` still persists.
    """

    def test_default_cache_dir_is_under_tempdir(self):
        import tempfile

        with BLSHttpClient() as client:
            assert client.cache_dir.startswith(tempfile.gettempdir())
            assert '.cache/bls' not in client.cache_dir

    def test_explicit_cache_dir_is_honored(self, tmp_path):
        explicit = str(tmp_path / 'mycache')
        with BLSHttpClient(cache_dir=explicit) as client:
            assert client.cache_dir == explicit
