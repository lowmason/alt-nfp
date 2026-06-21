"""Tests for the rebuilt vintage-store industry taxonomy (store_rebuild §3).

Covers the ``(industry_type, ownership)`` taxonomy map, the legacy→rebuilt
``remap_industry_type`` helper, the code-``55`` two-level representation, and
the new ``ownership`` schema column. See ``specs/plans/completed/10-store_rebuild.md`` T1.
"""

from __future__ import annotations

import polars as pl
import pytest
from nfp_lookups import (
    INDUSTRY_TAXONOMY,
    INDUSTRY_TYPES,
    OWNERSHIPS,
    VINTAGE_STORE_SCHEMA,
    codes_for,
    industry_types_for_code,
    ownership_for,
    remap_industry_type,
)

# ---------------------------------------------------------------------------
# Enums — 'national' retired, ownership axis added
# ---------------------------------------------------------------------------


def test_industry_types_retire_national():
    assert INDUSTRY_TYPES == ("total", "domain", "supersector", "sector")
    assert "national" not in INDUSTRY_TYPES


def test_ownerships_reserve_government():
    # 'government' is a reserved value (deferred, §11) — present in the enum
    # but never emitted by the stored taxonomy.
    assert OWNERSHIPS == ("total", "private", "government")
    assert "government" not in set(INDUSTRY_TAXONOMY.values())


# ---------------------------------------------------------------------------
# Taxonomy map: (industry_type, code) → ownership
# ---------------------------------------------------------------------------


def test_anchor_and_root_ownership():
    # 00 is the total-nonfarm anchor (stored, not modeled); 05 the private root.
    assert ownership_for("total", "00") == "total"
    assert ownership_for("total", "05") == "private"


def test_domain_ownership():
    assert ownership_for("domain", "06") == "private"
    assert ownership_for("domain", "08") == "private"


def test_supersector_codes_are_private_and_complete():
    ss = codes_for("supersector", "private")
    assert ss == ["10", "20", "30", "40", "50", "55", "60", "65", "70", "80"]
    assert all(ownership_for("supersector", c) == "private" for c in ss)


def test_sector_codes_are_private_and_complete():
    sectors = codes_for("sector", "private")
    assert sectors == [
        "11", "21", "22", "23", "31", "32", "42", "44", "48", "51",
        "52", "53", "54", "55", "56", "61", "62", "71", "72", "81",
    ]
    assert len(sectors) == 20


def test_codes_for_without_ownership_filter():
    assert codes_for("total") == ["00", "05"]


def test_ownership_for_unknown_pair_raises():
    with pytest.raises(ValueError):
        ownership_for("sector", "99")
    # Wrong level for a real code (00 is not a sector).
    with pytest.raises(ValueError):
        ownership_for("sector", "00")
    # Deferred government supersector code.
    with pytest.raises(ValueError):
        ownership_for("supersector", "90")


# ---------------------------------------------------------------------------
# Code 55 — the lone cross-level collision must stay distinct
# ---------------------------------------------------------------------------


def test_code_55_is_representable_at_two_levels():
    assert industry_types_for_code("55") == ["sector", "supersector"]
    # Both entries exist and are keyed independently on (industry_type, code).
    assert ("supersector", "55") in INDUSTRY_TAXONOMY
    assert ("sector", "55") in INDUSTRY_TAXONOMY
    assert ownership_for("supersector", "55") == "private"
    assert ownership_for("sector", "55") == "private"


def test_non_colliding_code_is_single_level():
    assert industry_types_for_code("00") == ["total"]
    assert industry_types_for_code("21") == ["sector"]


# ---------------------------------------------------------------------------
# Legacy → rebuilt remap (the ≤2023 history join, §10)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("legacy", "expected"),
    [
        (("national", "00"), ("total", "total")),
        (("domain", "05"), ("total", "private")),
        (("domain", "06"), ("domain", "private")),
        (("domain", "08"), ("domain", "private")),
        (("supersector", "10"), ("supersector", "private")),
        (("supersector", "55"), ("supersector", "private")),
        (("sector", "55"), ("sector", "private")),
        (("sector", "81"), ("sector", "private")),
    ],
)
def test_remap_industry_type(legacy, expected):
    assert remap_industry_type(*legacy) == expected


def test_remap_is_idempotent_on_rebuilt_inputs():
    # Feeding an already-rebuilt (industry_type, code) returns it unchanged.
    assert remap_industry_type("total", "00") == ("total", "total")
    assert remap_industry_type("total", "05") == ("total", "private")
    assert remap_industry_type("sector", "55") == ("sector", "private")


def test_remap_rejects_unmappable_code():
    with pytest.raises(ValueError):
        remap_industry_type("supersector", "90")  # deferred government


# ---------------------------------------------------------------------------
# Schema — ownership column added ahead of industry_type
# ---------------------------------------------------------------------------


def test_schema_has_ownership_column():
    assert VINTAGE_STORE_SCHEMA["ownership"] == pl.Utf8
    cols = list(VINTAGE_STORE_SCHEMA)
    # ownership sits between geographic_code and industry_type (spec §7 order).
    assert cols.index("ownership") == cols.index("geographic_code") + 1
    assert cols.index("ownership") == cols.index("industry_type") - 1
