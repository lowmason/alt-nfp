"""Smoke test: ``nfp_download.bls`` re-exports the series-ID grammar.

The grammar itself lives in ``nfp_lookups.series_ids`` (tested there);
this only guards the back-compat re-export surface.
"""

import nfp_lookups.series_ids as series_ids
from nfp_download.bls import (
    BLSProgram,
    SeriesField,
    build_series_id,
    get_program,
    list_programs,
    parse_series_id,
)
from nfp_download.bls._programs import PROGRAMS


def test_reexports_are_lookups_objects():
    assert PROGRAMS is series_ids.PROGRAMS
    assert BLSProgram is series_ids.BLSProgram
    assert SeriesField is series_ids.SeriesField
    assert build_series_id is series_ids.build_series_id
    assert parse_series_id is series_ids.parse_series_id
    assert get_program is series_ids.get_program
    assert list_programs is series_ids.list_programs


def test_reexport_roundtrip():
    sid = build_series_id(
        'CE', seasonal='S', supersector='00',
        industry='000000', data_type='01',
    )
    assert sid == 'CES0000000001'
    assert parse_series_id(sid)['program'] == 'CE'
