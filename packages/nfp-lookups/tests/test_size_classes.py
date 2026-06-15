"""Tests for the QCEW size-class schemes (store_rebuild §8 / size_classes.md)."""

from __future__ import annotations

import pytest
from nfp_lookups import (
    NATIVE_SIZE_CODES,
    SIZE_CLASS_MEMBERS,
    SIZE_CLASS_TYPES,
    native_to_scheme,
)


def test_scheme_names_and_bucket_counts():
    assert SIZE_CLASS_TYPES == ("total", "small", "medium", "large")
    assert len(native_to_scheme("large")) == 9
    assert set(native_to_scheme("total").values()) == {"0"}


def test_large_is_identity():
    assert native_to_scheme("large") == {c: c for c in NATIVE_SIZE_CODES}


def test_small_mapping():
    m = native_to_scheme("small")
    assert [m[c] for c in NATIVE_SIZE_CODES] == ["1", "1", "1", "1", "1", "2", "2", "3", "3"]


def test_medium_mapping():
    m = native_to_scheme("medium")
    assert [m[c] for c in NATIVE_SIZE_CODES] == ["1", "1", "1", "1", "2", "3", "4", "5", "5"]


def test_every_native_covered_once_per_scheme():
    for scheme in ("small", "medium"):
        covered = [c for natives in SIZE_CLASS_MEMBERS[scheme].values() for c in natives]
        assert sorted(covered) == list(NATIVE_SIZE_CODES)
        assert len(covered) == len(set(covered))  # no native in two buckets


def test_small_is_union_of_medium_buckets():
    # Full nesting: each 'small' bucket is a union of whole 'medium' buckets.
    med = SIZE_CLASS_MEMBERS["medium"]
    expected = {
        "1": set(med["1"]) | set(med["2"]),  # < 100
        "2": set(med["3"]) | set(med["4"]),  # 100-499
        "3": set(med["5"]),                  # 500+
    }
    got = {k: set(v) for k, v in SIZE_CLASS_MEMBERS["small"].items()}
    assert got == expected


def test_invalid_scheme_raises():
    with pytest.raises(ValueError):
        native_to_scheme("national")
