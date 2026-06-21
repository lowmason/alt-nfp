"""Tests for nfp_ingest.capture — CES month-T capture adapter (spec §5.1)."""

from datetime import date

import polars as pl
import pytest
from nfp_ingest import capture as _cap
from nfp_ingest.capture import (
    CaptureResult,
    _detect_corrected_levels,
    _knowable_qcew_quarter,
    _remap_ces_to_store_schema,
    capture_ces_print,
    capture_qcew_quarter,
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


# ===========================================================================
# QCEW conditional quarter capture (spec §5.2) — Phase 6
# ===========================================================================

def _qcew_store_row(
    ref_date: date,
    vintage_date: date,
    *,
    industry_code: str = "05",
    industry_type: str = "total",
    ownership: str = "private",
    revision: int = 0,
    employment: float = 130_000.0,
    size_class_type: str | None = None,
    size_class_code: str | None = None,
) -> dict:
    """One VINTAGE_STORE_SCHEMA-conformant QCEW row (NSA)."""
    return {
        "geographic_type": "national",
        "geographic_code": "00",
        "ownership": ownership,
        "industry_type": industry_type,
        "industry_code": industry_code,
        "ref_date": ref_date,
        "vintage_date": vintage_date,
        "revision": revision,
        "benchmark_revision": 0,
        "employment": employment,
        "size_class_type": size_class_type,
        "size_class_code": size_class_code,
        "source": "qcew",
        "seasonally_adjusted": False,
    }


def _write_qcew_partition(rows: list[dict], store_path) -> None:
    """Seed a tmp_path QCEW partition directly (a fixture, not capture output).

    This raw write is test-fixture scaffolding to a tmp_path store ONLY — never
    production code (production appends go through append_to_vintage_store).
    """
    df = pl.DataFrame(rows, schema=VINTAGE_STORE_SCHEMA)
    pdir = store_path / "source=qcew" / "seasonally_adjusted=false"
    pdir.mkdir(parents=True, exist_ok=True)
    df.drop(["source", "seasonally_adjusted"]).write_parquet(pdir / "data.parquet")


def _qcew_panel_rows(ref_year: int, qtr: int, employment: float) -> pl.DataFrame:
    """Stand-in for build_qcew_panel output: 14-of-16 cols, NO size_class_*.

    Mirrors qcew_crosswalk.build_qcew_panel's final .select (which omits
    size_class_type/size_class_code) so capture_qcew_quarter's null-fill is
    exercised. The test monkeypatches the schedule lookup, so any placeholder
    vintage_date that satisfies vintage_date <= as_of is fine here.
    """
    ref_month = (qtr - 1) * 3 + 1
    ref = date(ref_year, ref_month, 1)
    cols = [
        c for c in VINTAGE_STORE_SCHEMA
        if c not in ("size_class_type", "size_class_code")
    ]
    return pl.DataFrame(
        {
            "geographic_type": ["national"],
            "geographic_code": ["00"],
            "ownership": ["private"],
            "industry_type": ["total"],
            "industry_code": ["05"],
            "ref_date": [ref],
            "vintage_date": [date(ref_year, ref_month + 4, 1)],
            "revision": [0],
            "benchmark_revision": [0],
            "employment": [employment],
            "source": ["qcew"],
            "seasonally_adjusted": [False],
        }
    ).select(cols)


# --- _knowable_qcew_quarter ---------------------------------------------

class TestKnowableQcewQuarter:
    def test_picks_most_recent_knowable_quarter(self, monkeypatch):
        # Q1-2024 rev0 published 2024-05-01; Q2-2024 rev0 published 2024-08-01.
        def fake_vdate(ref_quarter, ref_year, revision):
            table = {
                ("Q1", 2024): date(2024, 5, 1),
                ("Q2", 2024): date(2024, 8, 1),
                ("Q3", 2024): date(2024, 11, 1),
            }
            return table.get((ref_quarter, ref_year), date(2099, 1, 1))

        monkeypatch.setattr(_cap, "get_qcew_vintage_date", fake_vdate)
        # As of 2024-06-01: Q1-2024 is knowable, Q2-2024 is not yet.
        assert _knowable_qcew_quarter(date(2024, 6, 1)) == ("Q1", 2024)

    def test_returns_none_when_no_quarter_knowable(self, monkeypatch):
        # Every candidate publishes in the far future ⇒ nothing knowable.
        monkeypatch.setattr(
            _cap,
            "get_qcew_vintage_date",
            lambda ref_quarter, ref_year, revision: date(2099, 1, 1),
        )
        assert _knowable_qcew_quarter(date(2024, 6, 1)) is None


# --- capture_qcew_quarter -----------------------------------------------

class TestCaptureQcewQuarter:
    def test_no_new_quarter_returns_skipped_no_append(self, tmp_path, monkeypatch):
        # Store already holds Q1-2024; as-of makes Q1-2024 the newest knowable.
        _write_qcew_partition(
            [_qcew_store_row(date(2024, 1, 1), date(2024, 5, 1))], tmp_path
        )

        monkeypatch.setattr(
            _cap,
            "get_qcew_vintage_date",
            lambda ref_quarter, ref_year, revision: (
                date(2024, 5, 1)
                if (ref_quarter, ref_year) == ("Q1", 2024)
                else date(2099, 1, 1)
            ),
        )

        def _boom(*a, **k):  # acquire must NOT be called on a no-op
            raise AssertionError("acquire_qcew_levels called on a no-op month")

        monkeypatch.setattr(_cap, "acquire_qcew_levels", _boom)

        result = capture_qcew_quarter(date(2024, 6, 1), store_path=tmp_path)

        assert isinstance(result, CaptureResult)
        assert result.appended == 0
        assert result.skipped == 1
        assert result.corrected == []

    def test_knowable_new_quarter_appends_rev0(self, tmp_path, monkeypatch):
        # Empty store; Q1-2024 becomes knowable as of 2024-06-01.
        monkeypatch.setattr(
            _cap,
            "get_qcew_vintage_date",
            lambda ref_quarter, ref_year, revision: (
                date(2024, 5, 1)
                if (ref_quarter, ref_year) == ("Q1", 2024)
                else date(2099, 1, 1)
            ),
        )
        # acquire returns a raw frame; build_qcew_panel / build_size_class_panel
        # are monkeypatched to the test panel (no real crosswalk / network).
        monkeypatch.setattr(
            _cap,
            "acquire_qcew_levels",
            lambda start_year, end_year=None: pl.DataFrame(
                {"year": [2024], "qtr": [1]}
            ),
        )
        monkeypatch.setattr(
            _cap,
            "acquire_qcew_size_native",
            lambda start_year, end_year=None: pl.DataFrame({"year": [2024]}),
        )
        monkeypatch.setattr(
            _cap, "build_qcew_panel", lambda raw: _qcew_panel_rows(2024, 1, 130_000.0)
        )
        # Size leg disabled for this test (return an empty Q1 size frame).
        empty_size = pl.DataFrame(schema=VINTAGE_STORE_SCHEMA).filter(pl.lit(False))
        monkeypatch.setattr(_cap, "build_size_class_panel", lambda native: empty_size)

        result = capture_qcew_quarter(date(2024, 6, 1), store_path=tmp_path)

        assert result.skipped == 0
        assert result.appended == 1
        assert result.corrected == []

        stored = pl.read_parquet(
            str(tmp_path / "source=qcew" / "seasonally_adjusted=false" / "*.parquet")
        )
        assert stored.height == 1
        assert stored["revision"].to_list() == [0]
        assert stored["industry_code"].to_list() == ["05"]
        # null-fill of the missing size cols held:
        assert stored["size_class_type"].to_list() == [None]
        assert stored["size_class_code"].to_list() == [None]
