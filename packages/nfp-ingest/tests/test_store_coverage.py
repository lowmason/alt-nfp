"""Integration test: vintage store industry coverage.

Verifies that the built vintage store in ``data/store/`` contains the
expected set of (industry_type, industry_code) combinations for each
source, and that revision coverage is correct.

Skips gracefully when the store has not been built.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from nfp_ingest.vintage_store import read_vintage_store, transform_to_panel
from nfp_lookups.industry import (
    CES_SECTOR_TO_NAICS,
    INDUSTRY_MAP,
    SINGLE_SECTOR_SUPERSECTORS,
)
from nfp_lookups.paths import VINTAGE_STORE_PATH


def _store_available() -> bool:
    """True if the vintage store (local dir or S3) is reachable and non-empty."""
    try:
        return VINTAGE_STORE_PATH.exists() and (
            next(VINTAGE_STORE_PATH.glob("**/*.parquet"), None) is not None
        )
    except Exception:  # unreachable endpoint / bad credentials → skip, not error
        return False


pytestmark = [
    pytest.mark.skipif(
        not _store_available(),
        reason="Vintage store not available (no local data/store/ and no reachable NFP_STORE_URI)",
    ),
    pytest.mark.real_store,  # reads the real vintage store via read_vintage_store(); exempt from cred blanking
]

VINTAGE_CUTOFF = date(2022, 1, 1)


def _offset_ref_date(d: date, months: int) -> date:
    """Shift a date by *months* and pin to day=12 (BLS convention)."""
    total = d.year * 12 + (d.month - 1) + months
    return date(total // 12, total % 12 + 1, 12)


# ---------------------------------------------------------------------------
# Build expected industry sets programmatically
# ---------------------------------------------------------------------------

_SECTOR_RECODE = {
    ces: naics for ces, naics in CES_SECTOR_TO_NAICS.items() if ces != naics
}


def _scope_to_types(scope: str) -> set[str]:
    """Map scope name to allowed industry_type values."""
    if scope == "all":
        return {"national", "domain", "supersector", "sector"}
    if scope == "domain":
        return {"national", "domain"}
    if scope == "supersector":
        return {"supersector"}
    if scope == "sector":
        return {"sector"}
    raise ValueError(f"Unknown scope: {scope!r}")


def _build_expected_industries(scope: str = "all") -> set[tuple[str, str]]:
    """Build expected (industry_type, industry_code) pairs from INDUSTRY_MAP + recoding.

    CES processing remaps ``industry_code='00'`` to ``industry_type='national'``,
    recodes sector codes 41→42 / 42→44 / 43→48, and duplicates single-sector
    supersectors (20→23, 50→51, 80→81) as sector rows.  QCEW produces the same
    final set via its own NAICS-based pipeline.
    """
    allowed = _scope_to_types(scope)
    pairs: set[tuple[str, str]] = set()

    for entry in INDUSTRY_MAP:
        itype = entry.industry_type
        icode = entry.industry_code

        if icode == "00" and itype == "domain":
            itype = "national"

        if itype not in allowed:
            continue

        if itype == "sector" and icode in _SECTOR_RECODE:
            icode = _SECTOR_RECODE[icode]

        pairs.add((itype, icode))

    if "sector" in allowed:
        for sec_code in SINGLE_SECTOR_SUPERSECTORS.values():
            pairs.add(("sector", sec_code))

    return pairs


def _build_expected_combos(
    scope: str,
    geographies: set[tuple[str, str]],
) -> set[tuple[str, str, str, str]]:
    """Cross-product geographies with expected industries into 4-tuples."""
    industries = _build_expected_industries(scope)
    return {
        (gt, gc, it, ic)
        for gt, gc in geographies
        for it, ic in industries
    }


def _discover_geographies(df: pl.DataFrame) -> set[tuple[str, str]]:
    """Return the distinct (geographic_type, geographic_code) pairs in *df*."""
    return set(
        df.select("geographic_type", "geographic_code").unique().iter_rows()
    )


# ---------------------------------------------------------------------------
# Data loading (cached to avoid repeated I/O across parametrised tests)
# ---------------------------------------------------------------------------

_STORE_CACHE: dict[tuple[str, bool | None], pl.DataFrame] = {}


def _load_store(source: str, seasonally_adjusted: bool | None) -> pl.DataFrame:
    """Load store data for one source partition, filtered to recent vintages."""
    key = (source, seasonally_adjusted)
    if key not in _STORE_CACHE:
        _STORE_CACHE[key] = (
            read_vintage_store(source=source, seasonally_adjusted=seasonally_adjusted)
            .filter(pl.col("vintage_date") >= VINTAGE_CUTOFF)
            .collect()
        )
    return _STORE_CACHE[key]


# ---------------------------------------------------------------------------
# Presence check
# ---------------------------------------------------------------------------


_GEO_INDUSTRY_COLS = [
    "geographic_type", "geographic_code", "industry_type", "industry_code",
]


def _check_presence(
    df: pl.DataFrame,
    expected: set[tuple[str, str, str, str]],
    label: str,
) -> list[str]:
    """Return gap descriptions for missing (geo, industry) 4-tuples.

    National geography is checked strictly.  Sub-national geographies may
    legitimately lack certain industries (e.g. DC has no Mining), so those
    gaps are reported but only fail when more than 1% of expected combos
    are missing for a given geographic_type.
    """
    if df.is_empty():
        return [f"{label}: no data after {VINTAGE_CUTOFF}"]

    observed = set(df.select(_GEO_INDUSTRY_COLS).unique().iter_rows())
    missing = sorted(expected - observed)
    if not missing:
        return []

    national_missing = [m for m in missing if m[0] == "national"]
    if national_missing:
        return [f"{label}: missing {len(national_missing)} national combos: {national_missing}"]

    gaps: list[str] = []
    by_geo_type: dict[str, list] = {}
    for m in missing:
        by_geo_type.setdefault(m[0], []).append(m)
    for geo_type, items in sorted(by_geo_type.items()):
        n_expected = sum(1 for e in expected if e[0] == geo_type)
        pct = len(items) / n_expected * 100 if n_expected else 0
        if pct > 1.0:
            gaps.append(
                f"{label}: {geo_type} missing {len(items)}/{n_expected} "
                f"combos ({pct:.1f}%): {items[:10]}"
            )
    return gaps


# ---------------------------------------------------------------------------
# CES revision check
# ---------------------------------------------------------------------------


def _check_ces_revisions(df: pl.DataFrame, label: str) -> list[str]:
    """CES: for each (industry, ref_date), revisions {0, 1, 2} should all exist.

    Only checks pre-benchmark rows (``benchmark_revision == 0``) because
    annual benchmark revisions replace the original triangular data.

    Uses a conservative ref_date window to avoid two known edge effects:

    * **Start of store**: ref_dates whose early revisions were published
      before ``VINTAGE_CUTOFF`` won't have revisions 0/1 in the filtered
      data.  We skip the first 3 months.
    * **Benchmark boundaries**: ref_dates within ~14 months of a February
      benchmark release may have their revision 2 superseded by the
      benchmarked value (``benchmark_revision > 0``), which is excluded
      by the filter.  We skip the last 15 months.
    """
    if df.is_empty():
        return []

    df = df.filter(pl.col("benchmark_revision") == 0)
    if df.is_empty():
        return []

    max_vdate = df["vintage_date"].max()
    min_ref = _offset_ref_date(VINTAGE_CUTOFF, 3)
    max_ref = _offset_ref_date(max_vdate, -15)

    grouped = (
        df.filter(pl.col("ref_date").is_between(min_ref, max_ref))
        .group_by([*_GEO_INDUSTRY_COLS, "ref_date"])
        .agg(revisions=pl.col("revision").unique().sort())
    )

    if grouped.is_empty():
        return []

    gaps: list[str] = []
    for k in range(3):
        missing_k = grouped.filter(~pl.col("revisions").list.contains(k))
        if missing_k.is_empty():
            continue
        n = missing_k.height
        examples = [
            f"  {r['geographic_type']}/{r['geographic_code']} "
            f"{r['industry_type']}/{r['industry_code']} ref={r['ref_date']}"
            for r in missing_k.head(5).iter_rows(named=True)
        ]
        gaps.append(
            f"{label}: {n} groups missing revision {k}:\n"
            + "\n".join(examples)
        )

    return gaps


# ---------------------------------------------------------------------------
# CES censored-series invariant (as_of → 1 rev-0, 1 rev-1, rest rev-2)
# ---------------------------------------------------------------------------

_CES_SERIES_KEY = [
    "geographic_type", "geographic_code", "industry_type", "industry_code",
]


def _check_ces_censored_invariant(df: pl.DataFrame, as_of: date, label: str) -> list[str]:
    """Verify CES triangular diagonal after censoring at as_of.

    For each series (geo, industry), filter to vintage_date <= as_of and
    benchmark_revision == 0, keep max revision per ref_date. The result
    must have exactly 1 ref_date with revision 0, 1 with revision 1, rest
    with revision 2, and ref_dates must be unique consecutive months
    ending at the rev-0 ref_date.
    """
    if df.is_empty():
        return []

    censored = (
        df.filter(
            (pl.col("vintage_date") <= as_of)
            & (pl.col("benchmark_revision") == 0)
        )
        .sort("revision", descending=True)
        .unique(subset=[*_CES_SERIES_KEY, "ref_date"], keep="first")
    )

    if censored.is_empty():
        return []

    gaps: list[str] = []
    for key_tuple, series in censored.group_by(_CES_SERIES_KEY):
        ref_dates = series["ref_date"].sort()

        n0 = (series["revision"] == 0).sum()
        n1 = (series["revision"] == 1).sum()
        n2 = (series["revision"] == 2).sum()
        n = len(series)

        if n0 != 1 or n1 != 1 or n2 != n - 2:
            key = key_tuple if isinstance(key_tuple, tuple) else (key_tuple,)
            gaps.append(
                f"{label} as_of={as_of} {key}: revision counts "
                f"(0,1,2)=({n0},{n1},{n2}) expected (1,1,{n - 2})"
            )
            continue

        rev0_ref = series.filter(pl.col("revision") == 0)["ref_date"][0]
        if ref_dates[-1] != rev0_ref:
            key = key_tuple if isinstance(key_tuple, tuple) else (key_tuple,)
            gaps.append(
                f"{label} as_of={as_of} {key}: last ref_date {ref_dates[-1]} "
                f"!= rev-0 ref_date {rev0_ref}"
            )
            continue

        months = ref_dates.dt.year() * 12 + ref_dates.dt.month()
        diffs = months.diff().drop_nulls()
        if (diffs != 1).any():
            key = key_tuple if isinstance(key_tuple, tuple) else (key_tuple,)
            gaps.append(
                f"{label} as_of={as_of} {key}: ref_dates not consecutive monthly"
            )

    return gaps


# ---------------------------------------------------------------------------
# QCEW revision check
# ---------------------------------------------------------------------------


def _check_qcew_revisions(df: pl.DataFrame) -> list[str]:
    """QCEW structural revision check.

    - **national/00**: should have at least one revision per ref_date (the
      revisions CSV may supersede the bulk revision-0 data for some quarters,
      so we check *any* revision exists rather than requiring revision 0).
    - **All other industries**: should have *only* revision 0 (bulk data).
    """
    if df.is_empty():
        return []

    gaps: list[str] = []
    is_total = (pl.col("industry_type") == "national") & (
        pl.col("industry_code") == "00"
    )

    total = df.filter(is_total)
    if not total.is_empty():
        empty_refs = (
            total.group_by([*_GEO_INDUSTRY_COLS, "ref_date"])
            .agg(n_revisions=pl.col("revision").n_unique())
            .filter(pl.col("n_revisions") == 0)
        )
        if not empty_refs.is_empty():
            gaps.append(
                f"QCEW national/00: {empty_refs.height} ref_dates with no revisions"
            )

    other = df.filter(~is_total)
    if not other.is_empty():
        non_zero = other.filter(pl.col("revision") > 0)
        if not non_zero.is_empty():
            combos = sorted(
                non_zero.select(_GEO_INDUSTRY_COLS).unique().iter_rows()
            )
            gaps.append(
                f"QCEW: {len(combos)} non-total industries have "
                f"unexpected non-zero revisions: {combos[:10]}"
            )

    return gaps


# ---------------------------------------------------------------------------
# Tests: industry presence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scope", ["all", "domain", "supersector", "sector"])
class TestStoreCoveragePresence:
    """Every expected (geo, industry) 4-tuple is present per source."""

    def test_ces_sa(self, scope: str) -> None:
        df = _load_store("ces", seasonally_adjusted=True)
        expected = _build_expected_combos(scope, _discover_geographies(df))
        gaps = _check_presence(df, expected, "ces_sa")
        assert not gaps, "\n".join(gaps)

    def test_ces_nsa(self, scope: str) -> None:
        df = _load_store("ces", seasonally_adjusted=False)
        expected = _build_expected_combos(scope, _discover_geographies(df))
        gaps = _check_presence(df, expected, "ces_nsa")
        assert not gaps, "\n".join(gaps)

    def test_qcew(self, scope: str) -> None:
        df = _load_store("qcew", seasonally_adjusted=False)
        expected = _build_expected_combos(scope, _discover_geographies(df))
        gaps = _check_presence(df, expected, "qcew")
        assert not gaps, "\n".join(gaps)


# ---------------------------------------------------------------------------
# Tests: revision coverage
# ---------------------------------------------------------------------------


class TestCesRevisionCoverage:
    """CES revision diagonals (0, 1, 2) are complete for ref_dates with gap >= 4 months."""

    def test_ces_sa_revisions(self) -> None:
        gaps = _check_ces_revisions(
            _load_store("ces", seasonally_adjusted=True), "ces_sa"
        )
        assert not gaps, "\n".join(gaps)

    def test_ces_nsa_revisions(self) -> None:
        gaps = _check_ces_revisions(
            _load_store("ces", seasonally_adjusted=False), "ces_nsa"
        )
        assert not gaps, "\n".join(gaps)


class TestCesCensoredSeriesInvariant:
    """CES censored via combined vintage_date + rank-based selection has
    1 rev-0, 1 rev-1, rest rev-2 per series with consecutive ref_dates.
    """

    @staticmethod
    def _check_panel_diagonal(
        panel: pl.DataFrame, source: str, label: str, as_of_ref: date,
    ) -> list[str]:
        """Verify the triangular diagonal invariant on a censored panel."""
        sub = panel.filter(pl.col("source") == source)
        if sub.is_empty():
            return []

        series_key = ["geographic_type", "geographic_code", "industry_level", "industry_code"]
        gaps: list[str] = []

        for key_vals, grp in sub.group_by(series_key):
            # One row per period
            if grp["period"].n_unique() != len(grp):
                key = key_vals if isinstance(key_vals, tuple) else (key_vals,)
                gaps.append(f"{label} {as_of_ref} {key}: duplicate periods")
                continue

            rev_nums = grp["revision_number"]
            n0 = (rev_nums == 0).sum()
            n1 = (rev_nums == 1).sum()

            if n0 != 1 or n1 != 1:
                key = key_vals if isinstance(key_vals, tuple) else (key_vals,)
                gaps.append(
                    f"{label} {as_of_ref} {key}: revision_number counts "
                    f"(0,1)=({n0},{n1}) expected (1,1)"
                )
                continue

            # Consecutive months
            months = grp["period"].sort()
            yms = months.dt.year() * 12 + months.dt.month()
            diffs = yms.diff().drop_nulls()
            if (diffs != 1).any():
                key = key_vals if isinstance(key_vals, tuple) else (key_vals,)
                gaps.append(f"{label} {as_of_ref} {key}: periods not consecutive")

        return gaps

    @staticmethod
    def _sample_as_of_dates() -> list[date]:
        """Pick as_of_ref dates (YYYY-MM-12) spread across the store window."""
        # Use ref_dates from the store to find a valid range
        df = _load_store("ces", seasonally_adjusted=True)
        min_ref = _offset_ref_date(VINTAGE_CUTOFF, 12)
        max_vdate = df["vintage_date"].max()
        max_ref = _offset_ref_date(max_vdate, -4)
        if min_ref >= max_ref:
            return []

        # Build candidate dates at day=12 stepping by ~3 months
        candidates: list[date] = []
        cursor = min_ref
        while cursor <= max_ref:
            # Skip Feb/Mar benchmark window
            if cursor.month not in (2, 3):
                candidates.append(cursor)
            cursor = _offset_ref_date(cursor, 1)

        if not candidates:
            return []

        step = max(1, len(candidates) // 8)
        return candidates[::step][:8]

    def test_ces_sa(self) -> None:
        as_of_dates = self._sample_as_of_dates()
        if not as_of_dates:
            pytest.skip("Store range too short for censored invariant window")

        lf = read_vintage_store(source="ces", seasonally_adjusted=True)
        gaps: list[str] = []
        for as_of_ref in as_of_dates:
            panel = transform_to_panel(lf, geographic_scope="national", as_of_ref=as_of_ref)
            gaps.extend(self._check_panel_diagonal(panel, "ces_sa", "ces_sa", as_of_ref))
        assert not gaps, "\n".join(gaps)

    def test_ces_nsa(self) -> None:
        as_of_dates = self._sample_as_of_dates()
        if not as_of_dates:
            pytest.skip("Store range too short for censored invariant window")

        lf = read_vintage_store(source="ces", seasonally_adjusted=False)
        gaps: list[str] = []
        for as_of_ref in as_of_dates:
            panel = transform_to_panel(lf, geographic_scope="national", as_of_ref=as_of_ref)
            gaps.extend(self._check_panel_diagonal(panel, "ces_nsa", "ces_nsa", as_of_ref))
        assert not gaps, "\n".join(gaps)


class TestQcewRevisionCoverage:
    """QCEW national/00 has revisions; other industries have revision 0 only."""

    def test_qcew_revisions(self) -> None:
        gaps = _check_qcew_revisions(
            _load_store("qcew", seasonally_adjusted=False)
        )
        assert not gaps, "\n".join(gaps)
