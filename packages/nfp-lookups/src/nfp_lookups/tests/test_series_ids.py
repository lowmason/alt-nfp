"""Tests for nfp_lookups.series_ids — program registry and series ID utils."""

import pytest
from nfp_lookups.series_ids import (
    PROGRAMS,
    BLSProgram,
    SeriesField,
    build_series_id,
    get_program,
    list_programs,
    parse_series_id,
)


class TestSeriesField:
    """Tests for SeriesField."""

    def test_length(self):
        f = SeriesField('seasonal', 3, 3)
        assert f.length == 1

    def test_extract(self):
        f = SeriesField('supersector', 4, 5)
        assert f.extract('CES0000000001') == '00'

    def test_extract_industry(self):
        f = SeriesField('industry', 6, 11)
        assert f.extract('CES0000000001') == '000000'


class TestBLSProgram:
    """Tests for BLSProgram."""

    def test_series_id_length_ce(self):
        prog = get_program('CE')
        assert prog.series_id_length == 13

    def test_series_id_length_sm(self):
        prog = get_program('SM')
        assert prog.series_id_length == 20

    def test_series_id_length_en(self):
        prog = get_program('EN')
        assert prog.series_id_length == 17

    def test_field_names_ce(self):
        prog = get_program('CE')
        assert prog.field_names() == [
            'prefix', 'seasonal', 'supersector', 'industry', 'data_type',
        ]

    def test_get_field(self):
        prog = get_program('CE')
        f = prog.get_field('seasonal')
        assert f is not None
        assert f.start == 3
        assert f.end == 3

    def test_get_field_missing(self):
        prog = get_program('CE')
        assert prog.get_field('nonexistent') is None


class TestProgramRegistry:
    """Tests for program registry."""

    def test_three_programs_registered(self):
        assert set(PROGRAMS.keys()) == {'CE', 'SM', 'EN'}

    def test_get_program_valid(self):
        prog = get_program('CE')
        assert isinstance(prog, BLSProgram)
        assert prog.prefix == 'CE'

    def test_get_program_case_insensitive(self):
        prog = get_program('ce')
        assert prog.prefix == 'CE'

    def test_get_program_unknown_raises(self):
        with pytest.raises(KeyError, match='Unknown BLS program prefix'):
            get_program('XX')

    def test_list_programs(self):
        result = list_programs()
        assert 'CE' in result
        assert 'SM' in result
        assert 'EN' in result
        assert len(result) == 3


class TestParseSeriesId:
    """Tests for parse_series_id."""

    def test_parse_ce(self):
        result = parse_series_id('CES0000000001')
        assert result['program'] == 'CE'
        assert result['prefix'] == 'CE'
        assert result['seasonal'] == 'S'
        assert result['supersector'] == '00'
        assert result['industry'] == '000000'
        assert result['data_type'] == '01'

    def test_parse_ce_nsa(self):
        result = parse_series_id('CEU0500000001')
        assert result['seasonal'] == 'U'
        assert result['supersector'] == '05'

    def test_parse_sm(self):
        result = parse_series_id('SMS36000000000000001')
        assert result['program'] == 'SM'
        assert result['seasonal'] == 'S'
        assert result['state'] == '36'
        assert result['area'] == '00000'
        assert result['data_type'] == '01'

    def test_parse_en(self):
        result = parse_series_id('ENU00000105000000')
        assert result['program'] == 'EN'
        assert result['prefix'] == 'EN'

    def test_roundtrip_ce(self):
        original = 'CES0000000001'
        parsed = parse_series_id(original)
        rebuilt = build_series_id(
            'CE',
            seasonal=parsed['seasonal'],
            supersector=parsed['supersector'],
            industry=parsed['industry'],
            data_type=parsed['data_type'],
        )
        assert rebuilt == original

    def test_roundtrip_sm(self):
        original = 'SMS3600000000000001'
        # SM is 20 chars
        original_padded = original.ljust(20, '0')
        parsed = parse_series_id(original_padded)
        rebuilt = build_series_id(
            'SM',
            seasonal=parsed['seasonal'],
            state=parsed['state'],
            area=parsed['area'],
            supersector_industry=parsed['supersector_industry'],
            data_type=parsed['data_type'],
        )
        assert rebuilt == original_padded

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match='too short'):
            parse_series_id('CES00')

    def test_single_char_raises(self):
        with pytest.raises(ValueError, match='at least 2 characters'):
            parse_series_id('C')

    def test_unknown_prefix_raises(self):
        with pytest.raises(KeyError):
            parse_series_id('XX0000000000')


class TestBuildSeriesId:
    """Tests for build_series_id."""

    def test_build_ce_sa(self):
        sid = build_series_id(
            'CE', seasonal='S', supersector='00',
            industry='000000', data_type='01',
        )
        assert sid == 'CES0000000001'

    def test_build_ce_nsa(self):
        sid = build_series_id(
            'CE', seasonal='U', supersector='05',
            industry='000000', data_type='01',
        )
        assert sid == 'CEU0500000001'

    def test_build_defaults_to_zeros(self):
        sid = build_series_id('CE', seasonal='S')
        assert sid == 'CES0000000000'
        assert len(sid) == 13

    def test_build_sm(self):
        sid = build_series_id(
            'SM', seasonal='S', state='36', area='00000',
            supersector_industry='00000000', data_type='01',
        )
        assert sid == 'SMS36000000000000001'
        assert len(sid) == 20
