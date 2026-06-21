"""Credential-free characterisation tests for transform_to_panel censoring and
panel_to_model_data extraction (H-2a / TestGap).

All tests use synthetic Polars frames and tmp_path only.  No store URI, no
S3 credentials, no real parquet files are needed.  The autouse conftest
blanks credentials for unmarked tests; these tests must still pass with
credentials blanked (CI condition).

Coverage
--------
Test A — CES censored diagonal:
    transform_to_panel(lf, as_of_ref=D) selects the triangular diagonal from a
    synthetic store with staggered vintage dates.  Asserts no future vintages,
    consecutive ref_dates, and the rev-0/rev-1/rev-2+ diagonal structure.

Test B — CES sigma vintage-remap ({0, 2} gap):
    panel_to_model_data with a panel where best-available CES omits revision 1.
    Asserts n_ces_vintages==2 and the correct ces_vintage_map.

Test C — QCEW post-COVID boundary multiplier:
    panel_to_model_data with two QCEW obs in Q1-2021: one M2 (Feb, interior)
    and one M3 (Mar, boundary).  Asserts the boundary month's noise multiplier
    is inflated by the era multiplier and the M2 month is not inflated.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest
from nfp_ingest.model_data import ModelDataConfig, panel_to_model_data
from nfp_ingest.vintage_store import VINTAGE_STORE_SCHEMA, transform_to_panel
from nfp_lookups.revision_schedules import get_noise_multiplier
from nfp_lookups.schemas import PANEL_SCHEMA

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def _ref(y: int, m: int) -> date:
    """BLS-convention ref_date (day=12)."""
    return date(y, m, 12)


def _make_store_df(
    *,
    geo_type: str = "national",
    geo_code: str = "00",
    ind_type: str = "national",
    ind_code: str = "00",
    source: str = "ces",
    sa: bool = True,
    ref_months: list[tuple[int, int]],
    revisions_and_lags: list[tuple[int, int, int]],
    base_emp: float = 150_000.0,
) -> pl.DataFrame:
    """Build a raw VINTAGE_STORE_SCHEMA frame (no growth column).

    Parameters
    ----------
    ref_months : list[(year, month)]
        Reference months to include.
    revisions_and_lags : list[(revision, benchmark_revision, lag_months)]
        For each combination, all ref_months get a row with vintage_date =
        ref_date shifted by lag_months.
    base_emp : float
        Base employment; each ref_month increments by 100 and each revision
        increments by 5 so all values are distinct and positive.
    """
    rows: list[dict] = []
    for i, (ry, rm) in enumerate(ref_months):
        rd = _ref(ry, rm)
        for rev, bmr, lag in revisions_and_lags:
            vm = rm + lag
            vy = ry + (vm - 1) // 12
            vm = ((vm - 1) % 12) + 1
            rows.append({
                "geographic_type": geo_type,
                "geographic_code": geo_code,
                "industry_type": ind_type,
                "industry_code": ind_code,
                "ref_date": rd,
                "vintage_date": date(vy, vm, 6),
                "revision": rev,
                "benchmark_revision": bmr,
                "employment": base_emp + i * 100.0 + rev * 5.0,
                "source": source,
                "seasonally_adjusted": sa,
            })
    return pl.DataFrame(rows, schema=VINTAGE_STORE_SCHEMA)


def _write_hive(df: pl.DataFrame, store_path: Path) -> None:
    """Write frame as Hive-partitioned vintage store under store_path."""
    for (source, sa), part in df.group_by(["source", "seasonally_adjusted"]):
        sa_str = str(sa).lower()
        pdir = store_path / f"source={source}" / f"seasonally_adjusted={sa_str}"
        pdir.mkdir(parents=True, exist_ok=True)
        part.drop(["source", "seasonally_adjusted"]).write_parquet(pdir / "data.parquet")


def _make_panel_row(
    period: date,
    source: str,
    source_type: str,
    revision_number: int,
    growth: float,
    vintage_date: date,
    is_final: bool = False,
    is_seasonally_adjusted: bool = True,
    employment_level: float = 150_000.0,
    geographic_code: str = "00",
    industry_code: str = "00",
) -> dict:
    """Return one row dict matching PANEL_SCHEMA."""
    return {
        "period": period,
        "geographic_type": "national",
        "geographic_code": geographic_code,
        "industry_code": industry_code,
        "industry_level": "national",
        "source": source,
        "source_type": source_type,
        "growth": growth,
        "employment_level": employment_level,
        "is_seasonally_adjusted": is_seasonally_adjusted,
        "vintage_date": vintage_date,
        "revision_number": revision_number,
        "is_final": is_final,
        "publication_lag_months": 1,
        "coverage_ratio": None,
    }


# ---------------------------------------------------------------------------
# Test A — CES censored diagonal via transform_to_panel
# ---------------------------------------------------------------------------


class TestCesCensoredDiagonal:
    """transform_to_panel(lf, as_of_ref=D) produces the correct triangular diagonal."""

    @pytest.fixture()
    def store_and_D(self, tmp_path):
        """Synthetic CES store with 7 ref-months, rev-0/1/2 at staggered vintages.

        as_of D = 2024-08-12 so:
          - ref-months 2024-01 through 2024-07 are all < D (eligible)
          - rev-0 vintage: lag 1 month (all <= D)
          - rev-1 vintage: lag 2 months (all <= D)
          - rev-2 vintage: lag 3 months (all <= D *except* 2024-07, which
            lands at 2024-10, well after D — excluded by vintage_date filter)

        After growth-diff each revision group loses the first ref-month, so
        survivors are 2024-02 through 2024-07 (6 unique periods available to
        rank-based selection).

        At D = 2024-08-12:
          Rank 1 (most-recent survivor) = 2024-07 → should pick rev-0
          Rank 2                         = 2024-06 → should pick rev-1
          Rank 3+                        = 2024-05..02 → should pick rev-2
        """
        D = _ref(2024, 8)

        # ref-months: 7 months 2024-01 through 2024-07
        ref_months = [(2024, m) for m in range(1, 8)]

        # rev-0: lag 1 mo; rev-1: lag 2 mo; rev-2: lag 3 mo
        # All rev-2 vintages land at lag+3 from ref, which for 2024-07 is 2024-10 > D.
        revisions_and_lags = [
            (0, 0, 1),  # rev-0 published 1 month later
            (1, 0, 2),  # rev-1 published 2 months later
            (2, 0, 3),  # rev-2 published 3 months later
        ]

        ces_sa = _make_store_df(
            ref_months=ref_months,
            revisions_and_lags=revisions_and_lags,
            source="ces",
            sa=True,
        )
        ces_nsa = _make_store_df(
            ref_months=ref_months,
            revisions_and_lags=revisions_and_lags,
            source="ces",
            sa=False,
        )
        df = pl.concat([ces_sa, ces_nsa])
        _write_hive(df, tmp_path)
        return tmp_path, D

    def test_no_future_vintages(self, store_and_D, tmp_path):
        """All selected vintage_dates are <= D (no lookahead)."""
        from nfp_ingest.vintage_store import read_vintage_store

        store_path, D = store_and_D
        lf = read_vintage_store(store_path)
        panel = transform_to_panel(lf, as_of_ref=D)

        assert panel["vintage_date"].max() <= D, (
            f"Panel contains a vintage_date after D={D}: "
            f"{panel['vintage_date'].max()}"
        )

    def test_consecutive_ref_dates(self, store_and_D):
        """Selected ref_dates within each series are consecutive months."""
        from nfp_ingest.vintage_store import read_vintage_store

        store_path, D = store_and_D
        lf = read_vintage_store(store_path)
        panel = transform_to_panel(lf, as_of_ref=D)

        # Group by series key (source + geography + industry) and check gaps
        for (src,), grp in panel.group_by(["source"]):
            periods = sorted(grp["period"].unique().to_list())
            for i in range(1, len(periods)):
                d1, d2 = periods[i - 1], periods[i]
                diff = (d2.year - d1.year) * 12 + d2.month - d1.month
                assert diff == 1, (
                    f"Source {src}: gap in period sequence {d1} → {d2}"
                )

    def test_diagonal_revision_structure(self, store_and_D):
        """Newest period → rev-0, next → rev-1, older periods → rev-2."""
        from nfp_ingest.vintage_store import read_vintage_store

        store_path, D = store_and_D
        lf = read_vintage_store(store_path)
        panel = transform_to_panel(lf, as_of_ref=D)

        # CES SA series
        ces_sa = panel.filter(pl.col("source") == "ces_sa").sort("period", descending=True)
        periods = ces_sa["period"].to_list()
        rev_nums = ces_sa["revision_number"].to_list()

        assert len(periods) >= 3, f"Expected at least 3 periods, got {len(periods)}"

        # Most recent: revision_number == 0 (first print)
        assert rev_nums[0] == 0, (
            f"Newest period {periods[0]} should be rev-0 but got rev-{rev_nums[0]}"
        )
        # Second: revision_number == 1 (second print)
        assert rev_nums[1] == 1, (
            f"Second-newest period {periods[1]} should be rev-1 but got rev-{rev_nums[1]}"
        )
        # Remaining: revision_number == 2 (third print / benchmark)
        for j in range(2, len(rev_nums)):
            assert rev_nums[j] == 2, (
                f"Period {periods[j]} (rank {j + 1}) should be rev-2 "
                f"but got rev-{rev_nums[j]}"
            )

    def test_newest_period_is_rev0(self, store_and_D):
        """The most-recent ref_date in the panel has revision_number == 0."""
        from nfp_ingest.vintage_store import read_vintage_store

        store_path, D = store_and_D
        lf = read_vintage_store(store_path)
        panel = transform_to_panel(lf, as_of_ref=D)

        ces_sa = panel.filter(pl.col("source") == "ces_sa")
        newest_row = ces_sa.sort("period", descending=True).head(1)
        assert newest_row["revision_number"][0] == 0

    def test_panel_schema_columns(self, store_and_D):
        """Output has all PANEL_SCHEMA columns with correct dtypes."""
        from nfp_ingest.vintage_store import read_vintage_store

        store_path, D = store_and_D
        lf = read_vintage_store(store_path)
        panel = transform_to_panel(lf, as_of_ref=D)

        for col, dtype in PANEL_SCHEMA.items():
            assert col in panel.columns, f"Missing column: {col}"
            assert panel.schema[col] == dtype, (
                f"Column {col!r}: expected {dtype}, got {panel.schema[col]}"
            )

    def test_ref_date_after_D_excluded(self, store_and_D):
        """No row in the panel has period >= D (ref_date < D filter)."""
        from nfp_ingest.vintage_store import read_vintage_store

        store_path, D = store_and_D
        lf = read_vintage_store(store_path)
        panel = transform_to_panel(lf, as_of_ref=D)

        bad = panel.filter(pl.col("period") >= D)
        assert len(bad) == 0, f"Found {len(bad)} rows with period >= D={D}"


# ---------------------------------------------------------------------------
# Test B — CES sigma vintage-remap when revision-1 is absent
# ---------------------------------------------------------------------------


class TestCesVintageRemap:
    """panel_to_model_data remaps vintage indices to contiguous 0-based range.

    When the best-available selection produces only vintages {0, 2} (no period
    resolves to revision-1), the remap should yield:
        n_ces_vintages == 2
        ces_vintage_map == {0: 0, 2: 1}
    """

    def _build_panel(self, D: date) -> pl.DataFrame:
        """Hand-build a panel where CES SA has rev-0 for the newest period and
        rev-2 for all older periods (no rev-1 anywhere).

        Periods: 2022-01 through 2022-05 (5 months).
        Rev-0 observation:  period=2022-05, vintage_date=D (within cutoff).
        Rev-2 observations: periods 2022-01..04, vintage_date early 2022.
        """
        rows: list[dict] = []

        # Rev-2 obs for 2022-01..04
        for m in range(1, 5):
            rows.append(_make_panel_row(
                period=date(2022, m, 1),
                source="ces_sa",
                source_type="official_sa",
                revision_number=2,
                growth=0.001 * m,
                vintage_date=date(2022, m + 4, 1),
                is_final=False,
            ))

        # Rev-0 obs for 2022-05
        rows.append(_make_panel_row(
            period=date(2022, 5, 1),
            source="ces_sa",
            source_type="official_sa",
            revision_number=0,
            growth=0.001,
            vintage_date=D,
            is_final=False,
        ))

        # NSA counterparts with identical structure (needed so CES NSA also
        # contributes to _all_vintages union; they share {0,2} → same remap).
        for m in range(1, 5):
            rows.append(_make_panel_row(
                period=date(2022, m, 1),
                source="ces_nsa",
                source_type="official_nsa",
                revision_number=2,
                growth=0.001 * m,
                vintage_date=date(2022, m + 4, 1),
                is_final=False,
                is_seasonally_adjusted=False,
            ))
        rows.append(_make_panel_row(
            period=date(2022, 5, 1),
            source="ces_nsa",
            source_type="official_nsa",
            revision_number=0,
            growth=0.001,
            vintage_date=D,
            is_final=False,
            is_seasonally_adjusted=False,
        ))

        return pl.DataFrame(rows, schema=PANEL_SCHEMA)

    def test_n_ces_vintages_matches_observed_set(self, tmp_path):
        """n_ces_vintages == 2 when only rev-0 and rev-2 are observed."""
        D = date(2022, 6, 12)
        panel = self._build_panel(D)

        md = panel_to_model_data(
            panel,
            providers=[],
            as_of=D,
            indicators_dir=tmp_path,
        )

        assert md["n_ces_vintages"] == 2, (
            f"Expected 2 distinct vintages (0, 2), got {md['n_ces_vintages']}"
        )

    def test_ces_vintage_map_gap(self, tmp_path):
        """ces_vintage_map skips revision-1 when it is absent."""
        D = date(2022, 6, 12)
        panel = self._build_panel(D)

        md = panel_to_model_data(
            panel,
            providers=[],
            as_of=D,
            indicators_dir=tmp_path,
        )

        expected = {0: 0, 2: 1}
        assert md["ces_vintage_map"] == expected, (
            f"ces_vintage_map: expected {expected}, got {md['ces_vintage_map']}"
        )

    def test_ces_sa_vintage_idx_uses_remapped_indices(self, tmp_path):
        """ces_sa_vintage_idx values are all < n_ces_vintages."""
        D = date(2022, 6, 12)
        panel = self._build_panel(D)

        md = panel_to_model_data(
            panel,
            providers=[],
            as_of=D,
            indicators_dir=tmp_path,
        )

        n = md["n_ces_vintages"]
        idx = md["ces_sa_vintage_idx"]
        assert all(0 <= v < n for v in idx.tolist()), (
            f"ces_sa_vintage_idx out of range [0, {n}): {idx.tolist()}"
        )

    def test_all_vintages_present_n_equals_three(self, tmp_path):
        """When rev-0, rev-1, rev-2 all appear, n_ces_vintages == 3."""
        D = date(2022, 8, 12)
        rows: list[dict] = []
        for rev, m in zip([0, 1, 2], [7, 6, 5], strict=True):
            rows.append(_make_panel_row(
                period=date(2022, m, 1),
                source="ces_sa",
                source_type="official_sa",
                revision_number=rev,
                growth=0.001,
                vintage_date=date(2022, m + 1, 1),
                is_final=False,
            ))
        # Add earlier months to build out a multi-period model calendar
        for m in range(1, 5):
            rows.append(_make_panel_row(
                period=date(2022, m, 1),
                source="ces_sa",
                source_type="official_sa",
                revision_number=2,
                growth=0.001,
                vintage_date=date(2022, m + 3, 1),
                is_final=False,
            ))
        panel = pl.DataFrame(rows, schema=PANEL_SCHEMA)

        md = panel_to_model_data(
            panel,
            providers=[],
            as_of=D,
            indicators_dir=tmp_path,
        )

        assert md["n_ces_vintages"] == 3


# ---------------------------------------------------------------------------
# Test C — QCEW post-COVID boundary multiplier
# ---------------------------------------------------------------------------


class TestQcewPostCovidBoundaryMultiplier:
    """panel_to_model_data inflates QCEW noise only for post-COVID M1/M3 months.

    The default ModelDataConfig uses:
      era_breaks = (date(2020, 1, 1),)          → era ≥ 1 for ref >= 2020-01-01
      qcew_post_covid_boundary_mult = {0: 5.0, 1: 3.5, 2: 2.0}
      qcew_post_covid_boundary_mult_default = 1.0

    M2 months (Feb=2, May=5, Aug=8, Nov=11) are quarter-interior → no inflation.
    M1/M3 months are boundary → multiplied by the era multiplier.

    We use Q1-2021 (Jan-2021 = M1 boundary, Feb-2021 = M2 interior, Mar-2021 = M3 boundary)
    with revision_number=0 in all three months so get_noise_multiplier("qcew_Q1", 0)
    applies everywhere, and the era multiplier (5.0 for rev=0) is the only
    differentiating factor.
    """

    def _build_panel(self, D: date) -> pl.DataFrame:
        """Hand-build a PANEL_SCHEMA frame with three QCEW obs in Q1-2021.

        Jan-2021 (M1, boundary), Feb-2021 (M2, interior), Mar-2021 (M3, boundary).
        Also include CES SA rows for each month so the model calendar covers
        the same periods (required by panel_to_model_data to build date_to_idx).
        """
        rows: list[dict] = []

        q1_months = [1, 2, 3]

        # QCEW rows (revision_number=0 throughout)
        for m in q1_months:
            rows.append(_make_panel_row(
                period=date(2021, m, 1),
                source="qcew",
                source_type="census",
                revision_number=0,
                growth=0.001 * m,
                vintage_date=date(2021, m + 3, 1),
                is_final=False,
                is_seasonally_adjusted=False,
            ))

        # CES SA rows for the same periods (needed to populate model calendar)
        for m in q1_months:
            rows.append(_make_panel_row(
                period=date(2021, m, 1),
                source="ces_sa",
                source_type="official_sa",
                revision_number=2,
                growth=0.001 * m,
                vintage_date=date(2021, m + 3, 1),
                is_final=False,
            ))

        return pl.DataFrame(rows, schema=PANEL_SCHEMA)

    def test_m2_month_not_inflated(self, tmp_path):
        """Feb-2021 (M2 interior) noise multiplier equals get_noise_multiplier without era inflation."""
        D = date(2021, 6, 12)
        panel = self._build_panel(D)
        config = ModelDataConfig()

        md = panel_to_model_data(
            panel,
            providers=[],
            as_of=D,
            config=config,
            indicators_dir=tmp_path,
        )

        dates_list = md["dates"]
        qcew_obs = md["qcew_obs"]
        qcew_noise_mult = md["qcew_noise_mult"]
        qcew_is_m2 = md["qcew_is_m2"]

        # Find the Feb-2021 obs
        m2_indices = [j for j, is_m2 in enumerate(qcew_is_m2) if is_m2
                      and dates_list[qcew_obs[j]].month == 2]
        assert len(m2_indices) == 1, (
            f"Expected exactly 1 Feb-2021 QCEW obs (M2), found {len(m2_indices)}"
        )

        j = m2_indices[0]
        expected_base = get_noise_multiplier("qcew_Q1", 0)
        assert qcew_noise_mult[j] == pytest.approx(expected_base, rel=1e-6), (
            f"Feb-2021 (M2): expected mult={expected_base} (no inflation), "
            f"got {qcew_noise_mult[j]}"
        )

    def test_m3_boundary_month_inflated(self, tmp_path):
        """Mar-2021 (M3 boundary, post-COVID) noise multiplier is base × era_mult."""
        D = date(2021, 6, 12)
        panel = self._build_panel(D)
        config = ModelDataConfig()
        era_mult = config.qcew_post_covid_boundary_mult[0]  # 5.0 for rev=0

        md = panel_to_model_data(
            panel,
            providers=[],
            as_of=D,
            config=config,
            indicators_dir=tmp_path,
        )

        dates_list = md["dates"]
        qcew_obs = md["qcew_obs"]
        qcew_noise_mult = md["qcew_noise_mult"]
        qcew_is_m2 = md["qcew_is_m2"]

        # Find the Mar-2021 obs (M3, boundary)
        m3_indices = [j for j, is_m2 in enumerate(qcew_is_m2)
                      if (not is_m2) and dates_list[qcew_obs[j]].month == 3]
        assert len(m3_indices) == 1, (
            f"Expected exactly 1 Mar-2021 QCEW obs (M3 boundary), found {len(m3_indices)}"
        )

        j = m3_indices[0]
        base = get_noise_multiplier("qcew_Q1", 0)
        expected_inflated = base * era_mult
        assert qcew_noise_mult[j] == pytest.approx(expected_inflated, rel=1e-6), (
            f"Mar-2021 (M3 boundary): expected mult={expected_inflated} "
            f"(base={base} × era_mult={era_mult}), got {qcew_noise_mult[j]}"
        )

    def test_boundary_to_interior_ratio_equals_era_multiplier(self, tmp_path):
        """Ratio mult[Mar-2021] / mult[Feb-2021] == era_mult for rev=0.

        This form avoids hard-coding absolute noise values and directly
        asserts that the boundary-only inflation is applied correctly.
        """
        D = date(2021, 6, 12)
        panel = self._build_panel(D)
        config = ModelDataConfig()
        era_mult = config.qcew_post_covid_boundary_mult[0]

        md = panel_to_model_data(
            panel,
            providers=[],
            as_of=D,
            config=config,
            indicators_dir=tmp_path,
        )

        dates_list = md["dates"]
        qcew_obs = md["qcew_obs"]
        qcew_noise_mult = md["qcew_noise_mult"]

        mult_by_month: dict[int, float] = {}
        for j in range(len(qcew_obs)):
            m = dates_list[qcew_obs[j]].month
            mult_by_month[m] = float(qcew_noise_mult[j])

        assert 2 in mult_by_month, "Feb-2021 obs not found in qcew_obs"
        assert 3 in mult_by_month, "Mar-2021 obs not found in qcew_obs"

        ratio = mult_by_month[3] / mult_by_month[2]
        assert ratio == pytest.approx(era_mult, rel=1e-6), (
            f"mult[Mar] / mult[Feb] = {ratio:.4f}, expected era_mult={era_mult}"
        )

    def test_pre_covid_month_not_inflated(self, tmp_path):
        """A pre-COVID M3 boundary month (2019-03) has no era inflation."""
        D = date(2019, 6, 12)

        rows: list[dict] = []
        for m in [1, 2, 3]:
            rows.append(_make_panel_row(
                period=date(2019, m, 1),
                source="qcew",
                source_type="census",
                revision_number=0,
                growth=0.001 * m,
                vintage_date=date(2019, m + 3, 1),
                is_final=False,
                is_seasonally_adjusted=False,
            ))
            rows.append(_make_panel_row(
                period=date(2019, m, 1),
                source="ces_sa",
                source_type="official_sa",
                revision_number=2,
                growth=0.001 * m,
                vintage_date=date(2019, m + 3, 1),
                is_final=False,
            ))
        panel = pl.DataFrame(rows, schema=PANEL_SCHEMA)

        md = panel_to_model_data(
            panel,
            providers=[],
            as_of=D,
            indicators_dir=tmp_path,
        )

        dates_list = md["dates"]
        qcew_obs = md["qcew_obs"]
        qcew_noise_mult = md["qcew_noise_mult"]

        mult_by_month: dict[int, float] = {}
        for j in range(len(qcew_obs)):
            month = dates_list[qcew_obs[j]].month
            mult_by_month[month] = float(qcew_noise_mult[j])

        # Both M1 (Jan-2019) and M3 (Mar-2019) are pre-COVID → no era inflation
        # They should equal the base Q1-rev0 multiplier, same as Feb (M2)
        base = get_noise_multiplier("qcew_Q1", 0)
        for boundary_m in [1, 3]:
            if boundary_m in mult_by_month:
                assert mult_by_month[boundary_m] == pytest.approx(base, rel=1e-6), (
                    f"Pre-COVID month {boundary_m}: expected no inflation (mult={base}), "
                    f"got {mult_by_month[boundary_m]}"
                )
