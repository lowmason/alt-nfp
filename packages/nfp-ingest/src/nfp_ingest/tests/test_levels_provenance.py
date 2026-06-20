"""Unit tests for ``levels_provenance`` — the nowcast index-path anchor scalars.

Regression guard for the A5/A4 ``base_index`` NaN: on the rebuilt 2017+ store the
panel's first month has no growth predecessor, so ``cum_level`` leaves
``ces_sa_index[0] = NaN``. The harnesses anchored ``base_index`` on ``[0]``, which
NaN'd every model nowcast. ``levels_provenance`` anchors on the first *finite*
index, leaving the old-store (finite-leading) case byte-identical.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from nfp_ingest.model_data import levels_provenance


def _levels(idx: list[float], lvl: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ces_sa_index": np.asarray(idx, dtype=float),
            "ces_sa_level": np.asarray(lvl, dtype=float),
        }
    )


def test_leading_nan_index_anchors_on_first_finite():
    """Rebuilt-store edge: index[0] is NaN -> anchor on the first finite index.

    This is the exact bug: ``float(ces_sa_index[0])`` would return NaN here.
    """
    levels = _levels(
        [np.nan, 100.2, 101.0, 100.0, 99.0],
        [145_000.0, 145_300.0, 146_000.0, 144_800.0, 143_000.0],
    )
    base_index, idx_to_level = levels_provenance(levels)
    assert math.isfinite(base_index)
    assert base_index == 100.2  # first finite, NOT the NaN at index[0]
    # index≈100 row is position 3 (|100.0-100|=0); NaN-robust argmin skips the lead
    assert idx_to_level == 144_800.0 / 100.0


def test_finite_leading_index_unchanged_old_store_parity():
    """Old-store case (finite index[0]): base_index == float(ces_sa_index[0])."""
    levels = _levels(
        [100.195, 100.5, 100.0, 101.0],
        [133_509.0, 133_600.0, 133_509.0, 134_000.0],
    )
    base_index, idx_to_level = levels_provenance(levels)
    assert base_index == 100.195  # identical to the pre-fix value -> A4 unchanged
    assert idx_to_level == 133_509.0 / 100.0  # row 2 (index == 100.0)


def test_all_nan_index_falls_back_to_base_100():
    """Degenerate all-NaN index -> finite fallback, never NaN."""
    levels = _levels([np.nan, np.nan], [np.nan, np.nan])
    base_index, _ = levels_provenance(levels)
    assert base_index == 100.0
