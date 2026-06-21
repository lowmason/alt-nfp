"""Back-compat guard for the QCEW acquire relocation (spec §5.2, §14 step 1).

After Phase 1, the acquire helpers LIVE in nfp_ingest.qcew_acquire. rebuild_store.py
must re-export them (under their original private names) so the existing
test_rebuild_acquire.py / test_rebuild_gates.py imports keep resolving, AND the
re-exported objects must be the SAME objects as the nfp_ingest definitions
(proving a real move, not a stale copy).
"""

from __future__ import annotations


def test_private_aliases_resolve_to_ingest_definitions():
    from nfp_ingest.qcew_acquire import (
        acquire_qcew_levels,
        acquire_qcew_size_native,
    )
    from nfp_vintages.rebuild_store import (
        _acquire_qcew_levels,
        _acquire_qcew_size_native,
    )

    # Same object — rebuild_store re-exports, does not redefine.
    assert _acquire_qcew_levels is acquire_qcew_levels
    assert _acquire_qcew_size_native is acquire_qcew_size_native


def test_moved_private_helpers_reexported():
    from nfp_ingest import qcew_acquire
    from nfp_vintages.rebuild_store import (
        _QCEW_LEVELS_REQUIRED,
        _fetch_qcew_csv,
        _prep_area_raw,
        _size_raw_to_native,
    )

    assert _fetch_qcew_csv is qcew_acquire._fetch_qcew_csv
    assert _prep_area_raw is qcew_acquire._prep_area_raw
    assert _size_raw_to_native is qcew_acquire._size_raw_to_native
    assert _QCEW_LEVELS_REQUIRED == qcew_acquire._QCEW_LEVELS_REQUIRED


def test_acquire_defined_in_ingest_module_not_rebuild_store():
    from nfp_vintages.rebuild_store import _acquire_qcew_levels

    # The function's home module is nfp_ingest.qcew_acquire (moved, not copied).
    assert _acquire_qcew_levels.__module__ == "nfp_ingest.qcew_acquire"


def test_series_identity_key_stays_in_rebuild_store():
    # _SERIES_IDENTITY_KEY is a compose helper, NOT part of the acquire move.
    from nfp_vintages.rebuild_store import _SERIES_IDENTITY_KEY

    assert _SERIES_IDENTITY_KEY[0] == "geographic_type"
