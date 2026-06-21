"""Tests for nfp_ingest.capture — CES month-T capture adapter (spec §5.1)."""

from datetime import date

import polars as pl
import pytest
from nfp_ingest import capture as _cap
from nfp_ingest.capture import (
    CaptureResult,
    _detect_corrected_levels,
    _remap_ces_to_store_schema,
    capture_ces_print,
)
from nfp_ingest.releases import COMBINED_SCHEMA
from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA


def _combined_row(
    *,
    industry_type: str,
    industry_code: str,
    sa: bool = True,
    ref_date: date = date(2026, 1, 1),
    vintage_date: date = date(2026, 2, 6),
    revision: int = 0,
    benchmark_revision: int = 0,
    employment: float = 1000.0,
) -> dict:
    """One COMBINED_SCHEMA row (legacy 11-col CES release shape)."""
    return {
        "source": "ces",
        "seasonally_adjusted": sa,
        "geographic_type": "national",
        "geographic_code": "00",
        "industry_type": industry_type,
        "industry_code": industry_code,
        "ref_date": ref_date,
        "vintage_date": vintage_date,
        "revision": revision,
        "benchmark_revision": benchmark_revision,
        "employment": employment,
    }


def _combined_frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=COMBINED_SCHEMA)


def test_remap_produces_vintage_store_schema():
    df = _combined_frame([
        _combined_row(industry_type="national", industry_code="00"),
        _combined_row(industry_type="domain", industry_code="05"),
        _combined_row(industry_type="supersector", industry_code="60"),
    ])

    out = _remap_ces_to_store_schema(df)

    assert out.columns == list(VINTAGE_STORE_SCHEMA)
    assert dict(zip(out.columns, out.dtypes, strict=True)) == VINTAGE_STORE_SCHEMA


def test_remap_assigns_rebuilt_taxonomy_per_code():
    df = _combined_frame([
        _combined_row(industry_type="national", industry_code="00"),
        _combined_row(industry_type="domain", industry_code="05"),
        _combined_row(industry_type="supersector", industry_code="70"),
    ])

    out = _remap_ces_to_store_schema(df).sort("industry_code")
    got = {
        r["industry_code"]: (r["industry_type"], r["ownership"])
        for r in out.iter_rows(named=True)
    }

    assert got["00"] == ("total", "total")
    assert got["05"] == ("total", "private")
    assert got["70"] == ("supersector", "private")


def test_remap_nulls_size_class_columns():
    df = _combined_frame([_combined_row(industry_type="supersector", industry_code="40")])

    out = _remap_ces_to_store_schema(df)

    assert out["size_class_type"].null_count() == out.height
    assert out["size_class_code"].null_count() == out.height


def _store_row(
    *,
    industry_code: str = "05",
    industry_type: str = "total",
    ownership: str = "private",
    ref_date: date = date(2026, 1, 1),
    vintage_date: date = date(2026, 2, 6),
    revision: int = 0,
    benchmark_revision: int = 0,
    employment: float = 1000.0,
    sa: bool = True,
) -> dict:
    """One VINTAGE_STORE_SCHEMA row (CES headline)."""
    return {
        "geographic_type": "national",
        "geographic_code": "00",
        "ownership": ownership,
        "industry_type": industry_type,
        "industry_code": industry_code,
        "ref_date": ref_date,
        "vintage_date": vintage_date,
        "revision": revision,
        "benchmark_revision": benchmark_revision,
        "employment": employment,
        "size_class_type": None,
        "size_class_code": None,
        "source": "ces",
        "seasonally_adjusted": sa,
    }


def _seed_store(store_path, rows: list[dict]) -> None:
    """Write VINTAGE_STORE_SCHEMA rows as a Hive-partitioned store under store_path."""
    df = pl.DataFrame(rows, schema=VINTAGE_STORE_SCHEMA)
    for (source, sa), part in df.group_by(["source", "seasonally_adjusted"]):
        sa_str = str(sa).lower()
        pdir = store_path / f"source={source}" / f"seasonally_adjusted={sa_str}"
        pdir.mkdir(parents=True, exist_ok=True)
        part.drop(["source", "seasonally_adjusted"]).write_parquet(pdir / "data.parquet")


def test_detect_corrected_flags_changed_level(tmp_path):
    _seed_store(tmp_path, [_store_row(employment=1000.0)])
    incoming = pl.DataFrame(
        [_store_row(employment=1234.0)], schema=VINTAGE_STORE_SCHEMA
    )

    corrected = _detect_corrected_levels(
        incoming, tmp_path, source="ces", seasonally_adjusted=True
    )

    assert len(corrected) == 1
    cl = corrected[0]
    assert cl.ref_date == date(2026, 1, 1)
    assert cl.industry_code == "05"
    assert cl.stored_employment == 1000.0
    assert cl.incoming_employment == 1234.0


def test_detect_corrected_ignores_matching_level(tmp_path):
    _seed_store(tmp_path, [_store_row(employment=1000.0)])
    incoming = pl.DataFrame(
        [_store_row(employment=1000.0)], schema=VINTAGE_STORE_SCHEMA
    )

    corrected = _detect_corrected_levels(
        incoming, tmp_path, source="ces", seasonally_adjusted=True
    )

    assert corrected == []


def test_detect_corrected_empty_store_returns_empty(tmp_path):
    incoming = pl.DataFrame(
        [_store_row(employment=1000.0)], schema=VINTAGE_STORE_SCHEMA
    )

    corrected = _detect_corrected_levels(
        incoming, tmp_path, source="ces", seasonally_adjusted=True
    )

    assert corrected == []


def test_capture_ces_raises_without_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("BLS_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="BLS_API_KEY"):
        capture_ces_print(date(2026, 2, 6), store_path=tmp_path)


def test_capture_ces_appends_and_censors(tmp_path, monkeypatch):
    monkeypatch.setenv("BLS_API_KEY", "dummy-key")

    # Two CES prints: one knowable as of 2026-02-06, one stamped later (must be
    # censored out by vintage_date <= as_of).
    fetched = _combined_frame([
        _combined_row(
            industry_type="domain", industry_code="05",
            ref_date=date(2026, 1, 1), vintage_date=date(2026, 2, 6),
            employment=131_000.0,
        ),
        _combined_row(
            industry_type="domain", industry_code="05",
            ref_date=date(2026, 2, 1), vintage_date=date(2026, 3, 6),
            employment=131_200.0,
        ),
    ])
    monkeypatch.setattr(_cap, "_fetch_ces_releases", lambda: fetched)

    result = capture_ces_print(date(2026, 2, 6), store_path=tmp_path)

    assert isinstance(result, CaptureResult)
    assert result.appended == 1
    assert result.corrected == []

    stored = pl.read_parquet(
        tmp_path / "source=ces" / "seasonally_adjusted=true" / "*.parquet"
    )
    assert stored.height == 1
    assert stored["ref_date"].to_list() == [date(2026, 1, 1)]
    assert stored["ownership"].to_list() == ["private"]


def test_capture_ces_idempotent_second_run_appends_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("BLS_API_KEY", "dummy-key")
    fetched = _combined_frame([
        _combined_row(
            industry_type="domain", industry_code="05",
            ref_date=date(2026, 1, 1), vintage_date=date(2026, 2, 6),
            employment=131_000.0,
        ),
    ])
    monkeypatch.setattr(_cap, "_fetch_ces_releases", lambda: fetched)

    first = capture_ces_print(date(2026, 2, 6), store_path=tmp_path)
    second = capture_ces_print(date(2026, 2, 6), store_path=tmp_path)

    assert first.appended == 1
    assert second.appended == 0
    assert second.skipped == 1


def test_capture_ces_flags_corrected_level(tmp_path, monkeypatch):
    monkeypatch.setenv("BLS_API_KEY", "dummy-key")

    base = _combined_row(
        industry_type="domain", industry_code="05",
        ref_date=date(2026, 1, 1), vintage_date=date(2026, 2, 6),
        employment=131_000.0,
    )
    monkeypatch.setattr(_cap, "_fetch_ces_releases", lambda: _combined_frame([base]))
    capture_ces_print(date(2026, 2, 6), store_path=tmp_path)

    # Re-capture the same ukey with a corrected level (a later vintage_date so it
    # is still censored in, but the same (ref,rev,bmr) ukey already present).
    corrected = dict(base)
    corrected["employment"] = 131_500.0
    corrected["vintage_date"] = date(2026, 2, 6)
    monkeypatch.setattr(
        _cap, "_fetch_ces_releases", lambda: _combined_frame([corrected])
    )
    result = capture_ces_print(date(2026, 2, 6), store_path=tmp_path)

    assert result.appended == 0
    assert len(result.corrected) == 1
    assert result.corrected[0].stored_employment == 131_000.0
    assert result.corrected[0].incoming_employment == 131_500.0


@pytest.mark.network
def test_capture_ces_live_fetch(tmp_path):
    import os

    if not os.environ.get("BLS_API_KEY"):
        pytest.skip("BLS_API_KEY not set")

    result = capture_ces_print(date.today(), store_path=tmp_path)

    assert isinstance(result, CaptureResult)
    assert result.appended >= 0
