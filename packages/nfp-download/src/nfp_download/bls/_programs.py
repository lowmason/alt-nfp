'''
Back-compat shim: BLS series-ID grammar now lives in ``nfp_lookups.series_ids``.

The grammar is pure reference knowledge (no I/O), so it belongs in the
foundation package. This module re-exports it so existing
``nfp_download.bls`` call sites keep working.
'''

from __future__ import annotations

from nfp_lookups.series_ids import (
    PROGRAMS,
    BLSProgram,
    SeriesField,
    build_series_id,
    get_program,
    list_programs,
    parse_series_id,
)

__all__ = [
    'PROGRAMS',
    'BLSProgram',
    'SeriesField',
    'build_series_id',
    'get_program',
    'list_programs',
    'parse_series_id',
]
