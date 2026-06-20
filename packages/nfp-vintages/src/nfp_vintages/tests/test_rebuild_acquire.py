"""Tests for the QCEW acquire layer in nfp_vintages.rebuild_store (store_rebuild T5).

Unit tests (no network) cover the pure-transform functions:
  - _prep_area_raw        — levels path prepare helper
  - _size_raw_to_native   — size path transform (agglvl remap, disclosure, per-size_code)

Network integration tests (``@pytest.mark.network``) fetch one real year from
BLS and are **not run by CI**.  They must be triggered manually by the maintainer::

    uv run pytest packages/nfp-vintages/tests/test_rebuild_acquire.py -m network --no-cov

CRITICAL SAFETY: no test here writes to a real/remote store.  No test calls
``_acquire_qcew_levels()`` or ``_acquire_qcew_size_native()`` (the full network
fetch loops); only the pure transform helpers and single-slice network probes
are tested. The network probes use ``create_client()`` (httpx) to exercise the
**same** transport as production — ``data.bls.gov`` needs no impersonation.
"""

from __future__ import annotations

import logging

import polars as pl
import pytest
from nfp_ingest.qcew_crosswalk import build_qcew_panel
from nfp_ingest.size_class import build_size_class_panel

# ---------------------------------------------------------------------------
# Synthetic frame builders
# ---------------------------------------------------------------------------

# QCEW industry_codes used in the area/size APIs.
# agglvl 13 (supersectors) – e.g. code '1013' = Manufacturing supersector
# agglvl 14 (sectors)      – e.g. code '22' = Utilities
# agglvl 15 (3-digit mfg)  – e.g. '311'–'339'
# agglvl 16 (4-digit)      – e.g. '1133' = Logging


def _area_csv_row(
    *,
    area_fips: str = "US000",
    own_code: str = "5",
    industry_code: str = "1013",
    agglvl_code: str = "13",
    year: int = 2024,
    qtr: int = 1,
    month1: int = 10_000,
    month2: int = 10_100,
    month3: int = 10_200,
    **extra,
) -> dict:
    """One raw area-endpoint CSV row (all values as strings, mirroring infer_schema_length=0)."""
    row: dict = {
        "area_fips": area_fips,
        "own_code": own_code,
        "industry_code": industry_code,
        "agglvl_code": agglvl_code,
        "year": str(year),
        "qtr": str(qtr),
        "month1_emplvl": str(month1),
        "month2_emplvl": str(month2),
        "month3_emplvl": str(month3),
        # Extra columns that appear in real CSVs but aren't consumed
        "disclosure_code": "",
        "total_qtrly_wages": "9999999",
    }
    row.update({k: str(v) for k, v in extra.items()})
    return row


def _make_area_raw(rows: list[dict]) -> pl.DataFrame:
    """Build an all-string raw area frame (as _fetch_qcew_csv returns).

    Polars ``infer_schema_length=0`` only works on file reads, not list-of-dicts.
    We build the frame column-by-column with an explicit ``Utf8`` schema.
    """
    if not rows:
        raise ValueError("_make_area_raw: empty rows list")
    cols = list(rows[0].keys())
    schema = dict.fromkeys(cols, pl.Utf8)
    return pl.DataFrame(
        {c: [str(r[c]) for r in rows] for c in cols},
        schema=schema,
    )


def _size_csv_row(
    *,
    area_fips: str = "US001",   # deliberately NOT US000 — tests normalisation
    own_code: str = "5",
    industry_code: str = "1013",
    agglvl_code: str = "23",     # size supersector (→ 13 after −10 remap)
    disclosure_code: str = "",
    year: int = 2024,
    qtr: int = 1,
    size_code: str = "1",
    month1: int = 5_000,
    month2: int = 5_050,
    month3: int = 5_100,
    **extra,
) -> dict:
    """One raw size-endpoint CSV row (all values as strings)."""
    row: dict = {
        "area_fips": area_fips,
        "own_code": own_code,
        "industry_code": industry_code,
        "agglvl_code": agglvl_code,
        "disclosure_code": disclosure_code,
        "year": str(year),
        "qtr": str(qtr),
        "size_code": size_code,
        "month1_emplvl": str(month1),
        "month2_emplvl": str(month2),
        "month3_emplvl": str(month3),
        "total_qtrly_wages": "0",
    }
    row.update({k: str(v) for k, v in extra.items()})
    return row


def _make_size_raw(rows: list[dict]) -> pl.DataFrame:
    """Build an all-string raw size frame (as _fetch_qcew_csv returns, after size_code tag)."""
    if not rows:
        raise ValueError("_make_size_raw: empty rows list")
    cols = list(rows[0].keys())
    schema = dict.fromkeys(cols, pl.Utf8)
    return pl.DataFrame(
        {c: [str(r[c]) for r in rows] for c in cols},
        schema=schema,
    )


# ---------------------------------------------------------------------------
# Minimal synthetic area + size datasets that produce known output
# ---------------------------------------------------------------------------

# We need enough rows for build_qcew_panel to produce ≥1 CES row.
# The simplest path: agglvl-13 supersector '1013' → CES supersector '30' (Manufacturing).
# This avoids summing children (supersectors with qcew_code=None only).

_MINIMAL_AREA_ROWS = [
    # agglvl 13 supersectors
    _area_csv_row(industry_code="1013", agglvl_code="13"),  # Manufacturing ss
    _area_csv_row(industry_code="1012", agglvl_code="13"),  # Construction ss
    _area_csv_row(industry_code="1021", agglvl_code="13"),  # TTU ss
    _area_csv_row(industry_code="1022", agglvl_code="13"),  # Information ss
    _area_csv_row(industry_code="1023", agglvl_code="13"),  # Financial ss
    _area_csv_row(industry_code="1024", agglvl_code="13"),  # PBS ss
    _area_csv_row(industry_code="1025", agglvl_code="13"),  # EHS ss
    _area_csv_row(industry_code="1026", agglvl_code="13"),  # L&H ss
    _area_csv_row(industry_code="1027", agglvl_code="13"),  # Other ss
    # agglvl 14 sectors (needed for sector-level pulls)
    _area_csv_row(industry_code="21", agglvl_code="14", month1=500, month2=510, month3=520),
    _area_csv_row(industry_code="22", agglvl_code="14", month1=600, month2=610, month3=620),
    _area_csv_row(industry_code="23", agglvl_code="14", month1=700, month2=710, month3=720),
    _area_csv_row(industry_code="42", agglvl_code="14", month1=800, month2=810, month3=820),
    _area_csv_row(industry_code="44-45", agglvl_code="14", month1=1200, month2=1210, month3=1220),
    _area_csv_row(industry_code="48-49", agglvl_code="14", month1=900, month2=910, month3=920),
    _area_csv_row(industry_code="51", agglvl_code="14", month1=300, month2=310, month3=320),
    _area_csv_row(industry_code="52", agglvl_code="14", month1=400, month2=410, month3=420),
    _area_csv_row(industry_code="53", agglvl_code="14", month1=350, month2=360, month3=370),
    _area_csv_row(industry_code="54", agglvl_code="14", month1=550, month2=560, month3=570),
    _area_csv_row(industry_code="55", agglvl_code="14", month1=200, month2=210, month3=220),
    _area_csv_row(industry_code="56", agglvl_code="14", month1=450, month2=460, month3=470),
    _area_csv_row(industry_code="61", agglvl_code="14", month1=250, month2=260, month3=270),
    _area_csv_row(industry_code="62", agglvl_code="14", month1=700, month2=710, month3=720),
    _area_csv_row(industry_code="71", agglvl_code="14", month1=150, month2=160, month3=170),
    _area_csv_row(industry_code="72", agglvl_code="14", month1=600, month2=610, month3=620),
    _area_csv_row(industry_code="81", agglvl_code="14", month1=300, month2=310, month3=320),
    # agglvl 15: 3-digit mfg subsectors (durable + nondurable pulls)
    _area_csv_row(industry_code="311", agglvl_code="15", month1=100, month2=110, month3=120),
    _area_csv_row(industry_code="312", agglvl_code="15", month1=50, month2=55, month3=60),
    _area_csv_row(industry_code="321", agglvl_code="15", month1=80, month2=85, month3=90),
    _area_csv_row(industry_code="327", agglvl_code="15", month1=60, month2=65, month3=70),
    _area_csv_row(industry_code="331", agglvl_code="15", month1=90, month2=95, month3=100),
    _area_csv_row(industry_code="332", agglvl_code="15", month1=200, month2=210, month3=220),
    _area_csv_row(industry_code="333", agglvl_code="15", month1=150, month2=160, month3=170),
    _area_csv_row(industry_code="334", agglvl_code="15", month1=180, month2=190, month3=200),
    _area_csv_row(industry_code="335", agglvl_code="15", month1=70, month2=75, month3=80),
    _area_csv_row(industry_code="336", agglvl_code="15", month1=500, month2=510, month3=520),
    _area_csv_row(industry_code="337", agglvl_code="15", month1=60, month2=65, month3=70),
    _area_csv_row(industry_code="339", agglvl_code="15", month1=40, month2=45, month3=50),
    _area_csv_row(industry_code="313", agglvl_code="15", month1=20, month2=25, month3=30),
    _area_csv_row(industry_code="314", agglvl_code="15", month1=15, month2=18, month3=20),
    _area_csv_row(industry_code="315", agglvl_code="15", month1=10, month2=12, month3=14),
    _area_csv_row(industry_code="316", agglvl_code="15", month1=8, month2=10, month3=12),
    _area_csv_row(industry_code="322", agglvl_code="15", month1=50, month2=55, month3=60),
    _area_csv_row(industry_code="323", agglvl_code="15", month1=30, month2=35, month3=40),
    _area_csv_row(industry_code="324", agglvl_code="15", month1=60, month2=65, month3=70),
    _area_csv_row(industry_code="325", agglvl_code="15", month1=200, month2=210, month3=220),
    _area_csv_row(industry_code="326", agglvl_code="15", month1=80, month2=85, month3=90),
    # agglvl 16: 4-digit Logging (for sector '11')
    _area_csv_row(industry_code="1133", agglvl_code="16", month1=30, month2=32, month3=34),
]


def _make_minimal_size_rows(
    size_code: str,
    month1_base: int,
    area_fips: str = "US001",
) -> list[dict]:
    """Return the minimal set of size rows for one size_code to pass build_qcew_panel."""
    # Same industry_code structure as area but at size agglvl (23/24/25/26)
    rows = []
    # agglvl 23 = supersectors
    for ic in ["1013", "1012", "1021", "1022", "1023", "1024", "1025", "1026", "1027"]:
        rows.append(_size_csv_row(
            area_fips=area_fips,
            industry_code=ic,
            agglvl_code="23",
            size_code=size_code,
            month1=month1_base,
            month2=month1_base + 50,
            month3=month1_base + 100,
        ))
    # agglvl 24 = sectors
    for ic, m in [
        ("21", 50), ("22", 60), ("23", 70), ("42", 80),
        ("44-45", 120), ("48-49", 90), ("51", 30), ("52", 40),
        ("53", 35), ("54", 55), ("55", 20), ("56", 45),
        ("61", 25), ("62", 70), ("71", 15), ("72", 60), ("81", 30),
    ]:
        rows.append(_size_csv_row(
            area_fips=area_fips,
            industry_code=ic,
            agglvl_code="24",
            size_code=size_code,
            month1=month1_base // 10 + m,
            month2=month1_base // 10 + m + 5,
            month3=month1_base // 10 + m + 10,
        ))
    # agglvl 25 = 3-digit mfg subsectors
    for ic in [
        "311", "312", "313", "314", "315", "316", "322", "323", "324", "325", "326",
        "321", "327", "331", "332", "333", "334", "335", "336", "337", "339",
    ]:
        rows.append(_size_csv_row(
            area_fips=area_fips,
            industry_code=ic,
            agglvl_code="25",
            size_code=size_code,
            month1=month1_base // 100 + 5,
            month2=month1_base // 100 + 6,
            month3=month1_base // 100 + 7,
        ))
    # agglvl 26 = 4-digit Logging
    rows.append(_size_csv_row(
        area_fips=area_fips,
        industry_code="1133",
        agglvl_code="26",
        size_code=size_code,
        month1=3,
        month2=4,
        month3=5,
    ))
    return rows


# ---------------------------------------------------------------------------
# _prep_area_raw tests
# ---------------------------------------------------------------------------


class TestPrepAreaRaw:
    """Unit tests for _prep_area_raw (pure transform — no network)."""

    from nfp_vintages.rebuild_store import _prep_area_raw  # type: ignore[attr-defined]

    def _prep(self, rows: list[dict]) -> pl.DataFrame:
        from nfp_vintages.rebuild_store import _prep_area_raw

        return _prep_area_raw(_make_area_raw(rows))

    def test_filter_keeps_private_and_total(self):
        rows = [
            _area_csv_row(own_code="5"),   # private — keep
            _area_csv_row(own_code="0"),   # total-covered — keep (T2)
            _area_csv_row(own_code="1"),   # federal govt — drop
            _area_csv_row(own_code="2"),   # state govt — drop
            _area_csv_row(own_code="3"),   # local govt — drop
        ]
        result = self._prep(rows)
        assert result.height == 2
        assert set(result["own_code"].to_list()) == {"5", "0"}

    def test_required_columns_present(self):
        from nfp_vintages.rebuild_store import _QCEW_LEVELS_REQUIRED

        result = self._prep([_area_csv_row()])
        for col in _QCEW_LEVELS_REQUIRED:
            assert col in result.columns, f"missing column {col!r}"

    def test_no_extra_columns(self):
        """Extra CSV columns (wages, disclosure, …) must be stripped."""
        from nfp_vintages.rebuild_store import _QCEW_LEVELS_REQUIRED

        result = self._prep([_area_csv_row()])
        assert set(result.columns) == set(_QCEW_LEVELS_REQUIRED)

    def test_emplvl_cast_to_int64(self):
        result = self._prep([_area_csv_row(month1=12345, month2=23456, month3=34567)])
        assert result["month1_emplvl"].dtype == pl.Int64
        assert result["month2_emplvl"].dtype == pl.Int64
        assert result["month3_emplvl"].dtype == pl.Int64

    def test_emplvl_values_correct(self):
        result = self._prep([_area_csv_row(month1=10_000, month2=10_100, month3=10_200)])
        assert result["month1_emplvl"][0] == 10_000
        assert result["month2_emplvl"][0] == 10_100
        assert result["month3_emplvl"][0] == 10_200

    def test_revision_zero_added(self):
        result = self._prep([_area_csv_row()])
        assert "revision" in result.columns
        assert result["revision"][0] == 0

    def test_area_fips_preserved(self):
        """area_fips must survive as-is (US000 for the national area endpoint)."""
        result = self._prep([_area_csv_row(area_fips="US000")])
        assert result["area_fips"][0] == "US000"

    def test_hyphenated_industry_code_preserved(self):
        """Codes like '44-45' must not be mangled by numeric casting."""
        result = self._prep([_area_csv_row(industry_code="44-45", agglvl_code="14")])
        assert result["industry_code"][0] == "44-45"

    def test_round_trip_build_qcew_panel(self):
        """prep output → build_qcew_panel succeeds and returns non-empty frame."""
        prepped = self._prep(_MINIMAL_AREA_ROWS)
        panel = build_qcew_panel(prepped)
        assert panel.height > 0
        assert "employment" in panel.columns
        assert panel["employment"].dtype == pl.Float64


# ---------------------------------------------------------------------------
# _size_raw_to_native tests
# ---------------------------------------------------------------------------


class TestSizeRawToNative:
    """Unit tests for _size_raw_to_native (pure transform — no network)."""

    def _native(self, rows: list[dict]) -> pl.DataFrame:
        from nfp_vintages.rebuild_store import _size_raw_to_native

        return _size_raw_to_native(_make_size_raw(rows))

    def _two_size_code_rows(self) -> list[dict]:
        """Minimal rows for size_code '1' and '9' with distinct employment."""
        sc1 = _make_minimal_size_rows("1", month1_base=1_000)
        sc9 = _make_minimal_size_rows("9", month1_base=9_000)
        return sc1 + sc9

    # --- Column contract ---

    def test_required_columns_present(self):
        from nfp_ingest.size_class import _REQUIRED_COLUMNS

        native = self._native(self._two_size_code_rows())
        for col in _REQUIRED_COLUMNS:
            assert col in native.columns, f"missing required column {col!r}"

    def test_size_code_is_string(self):
        native = self._native(self._two_size_code_rows())
        assert native["size_code"].dtype == pl.Utf8
        # Both size codes present
        codes = set(native["size_code"].unique().to_list())
        assert "1" in codes
        assert "9" in codes

    # --- area_fips normalisation ---

    def test_area_fips_normalised_to_us000(self):
        """size frames carry area_fips != US000; normalisation must happen."""
        rows = _make_minimal_size_rows("1", month1_base=1_000, area_fips="US001")
        native = self._native(rows)
        # If normalisation failed, build_qcew_panel would have returned 0 rows
        # (its internal filter is area_fips == QCEW_AREA_NATIONAL).
        assert native.height > 0

    def test_non_us000_area_fips_produces_output(self):
        """Explicit check: different area_fips values in input still yield output."""
        for fips in ["US001", "99999", "USALL"]:
            rows = _make_minimal_size_rows("1", month1_base=1_000, area_fips=fips)
            native = self._native(rows)
            assert native.height > 0, f"area_fips={fips!r} yielded zero rows"

    # --- agglvl 61–64 duplicate exclusion ---

    def test_agglvl_61_64_excluded(self):
        """Rows with agglvl 61–64 must not inflate output (they duplicate 21–24)."""
        sc1_good = _make_minimal_size_rows("1", month1_base=1_000)
        # Add duplicate rows at agglvl 63 (same industry_codes, same employment)
        # If they were included, employment would double.
        dup_rows = [
            _size_csv_row(
                industry_code="1013",
                agglvl_code="63",    # duplicate family — must be dropped
                size_code="1",
                month1=1_000_000,    # huge value — would inflate if included
                month2=1_000_000,
                month3=1_000_000,
            )
        ]
        native_without_dups = self._native(sc1_good)
        native_with_dups = self._native(sc1_good + dup_rows)
        # Employment totals must be identical (dups dropped)
        assert pytest.approx(native_without_dups["employment"].sum(), rel=1e-6) == \
            native_with_dups["employment"].sum()

    # --- disclosure_code filtering ---

    def test_disclosure_n_rows_dropped(self):
        """Rows with disclosure_code='N' must be dropped, not summed as zeros.

        Discriminating design: the suppressed row uses industry_code '1027' at
        agglvl 23 with month1=999_999.  If the N-row is NOT dropped it bleeds
        into the build_qcew_panel sum and inflates total employment by ~999_999.
        The baseline run has no such industry at that value, so any bleed is
        detectable as a large employment difference.
        """
        sc1_good = _make_minimal_size_rows("1", month1_base=1_000)
        # Baseline: employment sum without any N-row
        native_baseline = self._native(sc1_good)
        baseline_emp = native_baseline["employment"].sum()

        # N-row uses a sentinel value that would inflate the sum if not filtered
        suppressed = _size_csv_row(
            industry_code="1027",   # other services supersector — present in minimal rows
            agglvl_code="23",
            size_code="1",
            disclosure_code="N",
            month1=999_999,         # sentinel: visible if bleed-through occurs
            month2=999_999,
            month3=999_999,
        )
        native_with_n = self._native(sc1_good + [suppressed])
        emp_with_n = native_with_n["employment"].sum()

        # If the N-row were included, employment would be ~999_999 higher
        assert emp_with_n == pytest.approx(baseline_emp, rel=1e-6), (
            f"N-row bleed detected: employment with N-row ({emp_with_n:.0f}) != "
            f"baseline ({baseline_emp:.0f}); disclosure_code='N' filter is broken"
        )

    def test_disclosure_empty_string_kept(self):
        """disclosure_code='' (disclosed) rows must be kept."""
        rows = _make_minimal_size_rows("1", month1_base=1_000)
        # All rows in _make_minimal_size_rows have disclosure_code='' by default.
        native = self._native(rows)
        assert native.height > 0

    def test_disclosure_null_kept(self):
        """Null disclosure_code (empty CSV field parsed by Polars) must be kept.

        ``pl.read_csv(infer_schema_length=0)`` parses an empty CSV field as
        ``null`` (not ``""``).  The filter ``~(col == "N")`` would drop null rows
        via Polars three-valued logic.  The safe filter is
        ``col.is_null() | (col != "N")``.
        """
        from nfp_vintages.rebuild_store import _size_raw_to_native

        sc1_rows = _make_minimal_size_rows("1", month1_base=1_000)
        # Build the frame normally, then overwrite disclosure_code with None
        # for all rows to simulate what Polars CSV read produces for empty fields.
        raw = _make_size_raw(sc1_rows).with_columns(
            pl.lit(None, pl.Utf8).alias("disclosure_code")
        )
        native = _size_raw_to_native(raw)
        assert native.height > 0, (
            "null disclosure_code rows were dropped — Polars null-safety bug in disclosure filter"
        )

    def test_disclosure_logging(self, caplog):
        """Disclosure distribution must be logged at INFO level."""
        rows = _make_minimal_size_rows("1", month1_base=1_000)
        with caplog.at_level(logging.INFO, logger="nfp_vintages.rebuild_store"):
            self._native(rows)
        assert any("Disclosure distribution" in rec.message for rec in caplog.records), (
            "expected 'Disclosure distribution' log entry not found"
        )

    # --- per-size_code correctness (the key anti-collapse guard) ---

    def test_two_size_codes_not_summed(self):
        """size_code='1' and '9' must remain separate — not collapsed by build_qcew_panel."""
        native = self._native(self._two_size_code_rows())
        # Both size_codes must be present in output
        codes = set(native["size_code"].unique().to_list())
        assert "1" in codes
        assert "9" in codes

        # Employment for size_code '1' and '9' must differ (they have different
        # input employment values), confirming they were not summed together.
        emp_1 = native.filter(pl.col("size_code") == "1")["employment"].sum()
        emp_9 = native.filter(pl.col("size_code") == "9")["employment"].sum()
        assert emp_1 != pytest.approx(emp_9), (
            "employment for size_code='1' and '9' must differ — they appear to have been summed"
        )

    def test_size_codes_independent(self):
        """Employment of a single-code run must equal the same code in a combined run."""
        sc1_only = _make_minimal_size_rows("1", month1_base=1_000)
        sc9_only = _make_minimal_size_rows("9", month1_base=9_000)
        combined = sc1_only + sc9_only

        native_1_only = self._native(sc1_only)
        native_combined = self._native(combined)

        emp_1_alone = native_1_only["employment"].sum()
        emp_1_in_combined = native_combined.filter(
            pl.col("size_code") == "1"
        )["employment"].sum()
        assert emp_1_alone == pytest.approx(emp_1_in_combined, rel=1e-6), (
            "single size_code run must have same employment as the same code in combined run"
        )

    # --- Q1 assertion ---

    def test_ref_dates_all_q1(self):
        """Output ref_dates must all be in months 1–3 (Q1-only endpoint)."""
        native = self._native(self._two_size_code_rows())
        months = native["ref_date"].dt.month().unique().to_list()
        assert all(m in [1, 2, 3] for m in months), (
            f"non-Q1 months in native output: {[m for m in months if m not in [1, 2, 3]]}"
        )

    # --- round-trip through build_size_class_panel ---

    def test_round_trip_build_size_class_panel(self):
        """build_size_class_panel(_size_raw_to_native(synthetic)) succeeds."""
        native = self._native(self._two_size_code_rows())
        panel = build_size_class_panel(native)
        assert panel.height > 0
        assert "size_class_type" in panel.columns
        assert "size_class_code" in panel.columns

    def test_round_trip_has_all_scheme_types(self):
        """The cross-product must contain total / small / medium / large buckets."""
        from nfp_lookups.size_classes import SIZE_CLASS_TYPES

        native = self._native(self._two_size_code_rows())
        panel = build_size_class_panel(native)
        present_types = set(panel["size_class_type"].unique().to_list())
        for scheme in SIZE_CLASS_TYPES:
            assert scheme in present_types, f"size_class_type={scheme!r} missing from panel"

    def test_round_trip_size_codes_propagated(self):
        """Each size bucket code must correspond to a real native size_code or rollup."""
        native = self._native(self._two_size_code_rows())
        panel = build_size_class_panel(native)
        # total scheme has code '0'
        assert "0" in panel.filter(
            pl.col("size_class_type") == "total"
        )["size_class_code"].unique().to_list()

    # --- fail-loud on zero rows ---

    def test_no_size_codes_after_filter_raises(self):
        """Only agglvl 61–64 (duplicate family) → all filtered → 'no size_codes' guard."""
        # agglvl 61 is dropped by the 21–28 filter, so no rows (and no size_codes)
        # survive → the aggregate guard fires (NOT the per-size_code guard).
        bad_rows = [
            _size_csv_row(industry_code="1013", agglvl_code="61", size_code="1", month1=1_000)
        ]
        from nfp_vintages.rebuild_store import _size_raw_to_native

        with pytest.raises(RuntimeError, match="no size_codes found"):
            _size_raw_to_native(_make_size_raw(bad_rows))

    def test_per_size_code_zero_rows_raises(self):
        """A size_code that survives filtering but yields no crosswalk rows must raise.

        agglvl 27 passes the 21–28 filter but remaps to 17, which
        build_qcew_panel does not pull → 0 output rows for that size_code → the
        *per-size_code* guard fires (distinct from the aggregate guard above).
        """
        # 5-digit code at agglvl 27 (→17 after −10); a real size_code survives.
        bad_rows = [
            _size_csv_row(industry_code="11111", agglvl_code="27", size_code="1", month1=1_000)
        ]
        from nfp_vintages.rebuild_store import _size_raw_to_native

        with pytest.raises(RuntimeError, match="0 rows for size_code"):
            _size_raw_to_native(_make_size_raw(bad_rows))


# ---------------------------------------------------------------------------
# Network integration tests — maintainer-run only, NOT in CI
# ---------------------------------------------------------------------------


@pytest.mark.network
class TestAcquireLevelsNetwork:
    """Fetch ONE real QCEW area slice and validate basic shape.

    Run with: ``pytest -m network packages/nfp-vintages/tests/test_rebuild_acquire.py``
    """

    def test_single_slice_non_empty(self):
        from nfp_download.client import create_client
        from nfp_vintages.rebuild_store import _fetch_qcew_csv

        url = "https://data.bls.gov/cew/data/api/2024/1/area/US000.csv"
        with create_client() as session:
            raw = _fetch_qcew_csv(session, url)
        assert raw is not None, "expected 200 for 2024/Q1 area slice"
        assert raw.height > 0

    def test_single_slice_has_expected_columns(self):
        from nfp_download.client import create_client
        from nfp_vintages.rebuild_store import _fetch_qcew_csv

        url = "https://data.bls.gov/cew/data/api/2024/1/area/US000.csv"
        with create_client() as session:
            raw = _fetch_qcew_csv(session, url)
        assert raw is not None
        for col in [
            "area_fips", "own_code", "industry_code", "agglvl_code",
            "year", "qtr", "month1_emplvl", "month2_emplvl", "month3_emplvl",
        ]:
            assert col in raw.columns, f"missing column {col!r} in real area CSV"

    def test_prep_private_and_total(self):
        from nfp_download.client import create_client
        from nfp_vintages.rebuild_store import _fetch_qcew_csv, _prep_area_raw

        url = "https://data.bls.gov/cew/data/api/2024/1/area/US000.csv"
        with create_client() as session:
            raw = _fetch_qcew_csv(session, url)
        assert raw is not None
        prepped = _prep_area_raw(raw)
        assert prepped.height > 0
        # After T2: both private (own_code='5') and total-covered (own_code='0') are kept;
        # government (own_code='1'/'2'/'3') is dropped.
        assert set(prepped["own_code"].unique().to_list()) == {"0", "5"}

    def test_expected_agglvls_present(self):
        from nfp_download.client import create_client
        from nfp_vintages.rebuild_store import _fetch_qcew_csv, _prep_area_raw

        url = "https://data.bls.gov/cew/data/api/2024/1/area/US000.csv"
        with create_client() as session:
            raw = _fetch_qcew_csv(session, url)
        assert raw is not None
        prepped = _prep_area_raw(raw)
        agglvls = set(prepped["agglvl_code"].unique().to_list())
        # Private tree uses agglvl 13–16; total-covered anchor uses agglvl 10.
        for expected in ["10", "13", "14", "15", "16"]:
            assert expected in agglvls, f"agglvl_code {expected!r} missing from real area slice"

    def test_404_on_future_year_returns_none(self):
        from nfp_download.client import create_client
        from nfp_vintages.rebuild_store import _fetch_qcew_csv

        url = "https://data.bls.gov/cew/data/api/2099/4/area/US000.csv"
        with create_client() as session:
            result = _fetch_qcew_csv(session, url)
        assert result is None, "expected None (404) for a far-future quarter"


@pytest.mark.network
class TestAcquireSizeNetwork:
    """Fetch ONE real QCEW size slice and validate basic shape + disclosure logging.

    Run with: ``pytest -m network packages/nfp-vintages/tests/test_rebuild_acquire.py``
    """

    def test_single_size_slice_non_empty(self):
        from nfp_download.client import create_client
        from nfp_vintages.rebuild_store import _fetch_qcew_csv

        url = "https://data.bls.gov/cew/data/api/2024/1/size/1.csv"
        with create_client() as session:
            raw = _fetch_qcew_csv(session, url)
        assert raw is not None, "expected 200 for 2024/Q1 size/1 slice"
        assert raw.height > 0

    def test_size_slice_has_own_code(self):
        """Verify size CSVs carry own_code so private filter works."""
        from nfp_download.client import create_client
        from nfp_vintages.rebuild_store import _fetch_qcew_csv

        url = "https://data.bls.gov/cew/data/api/2024/1/size/1.csv"
        with create_client() as session:
            raw = _fetch_qcew_csv(session, url)
        assert raw is not None
        assert "own_code" in raw.columns

    def test_size_agglvls_present(self):
        """Verify agglvl 23–26 exist in the real size CSV (the ones we remap)."""
        from nfp_download.client import create_client
        from nfp_vintages.rebuild_store import _fetch_qcew_csv

        url = "https://data.bls.gov/cew/data/api/2024/1/size/1.csv"
        with create_client() as session:
            raw = _fetch_qcew_csv(session, url)
        assert raw is not None
        agglvls = set(raw["agglvl_code"].unique().to_list())
        # At minimum, agglvl 23 (supersectors) must be present
        assert "23" in agglvls, "agglvl 23 (supersectors) not found in real size CSV"

    def test_disclosure_logging_fires(self, caplog):
        """Verify _size_raw_to_native logs the disclosure distribution for a real slice."""
        from nfp_download.client import create_client
        from nfp_vintages.rebuild_store import _fetch_qcew_csv, _size_raw_to_native

        url = "https://data.bls.gov/cew/data/api/2024/1/size/1.csv"
        with create_client() as session:
            raw = _fetch_qcew_csv(session, url)
        assert raw is not None
        raw = raw.with_columns(size_code=pl.lit("1", pl.Utf8))

        with caplog.at_level(logging.INFO, logger="nfp_vintages.rebuild_store"):
            _size_raw_to_native(raw)

        assert any("Disclosure distribution" in r.message for r in caplog.records)
