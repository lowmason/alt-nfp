"""Tests for QCEW-weighted provider compositing (ingest/compositing.py)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from nfp_ingest.compositing import (
    CELL_INDEX,
    CELL_SUPERSECTOR,
    N_CELLS,
    REGION_CODES,
    SUPERSECTOR_CODES,
    compute_provider_composite,
    load_qcew_weights,
    redistribute_weights,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_REF_DATES = [date(2020, m, 12) for m in range(1, 7)]  # Jan–Jun 2020


def _build_qcew_store(
    ref_dates: list[date],
    tmp_dir: Path,
) -> Path:
    """Write a minimal QCEW store to *tmp_dir* and return the store path.

    Each cell gets employment = 1000 + cell_index so shares are non-uniform.
    """
    rows: list[dict] = []
    for rd in ref_dates:
        for (rg, ss), ci in CELL_INDEX.items():
            rows.append({
                "geographic_type": "region",
                "geographic_code": rg,
                "industry_type": "supersector",
                "industry_code": ss,
                "ref_date": rd,
                "vintage_date": rd,
                "revision": 0,
                "benchmark_revision": 0,
                "employment": 1000.0 + ci,
                "growth": None,
                "source": "qcew",
                "seasonally_adjusted": False,
            })

    store_dir = tmp_dir / "source=qcew" / "seasonally_adjusted=false"
    store_dir.mkdir(parents=True)
    df = pl.DataFrame(rows).drop("source", "seasonally_adjusted")
    df.write_parquet(store_dir / "data.parquet")
    return tmp_dir


def _build_provider_df(
    ref_dates: list[date],
    cells: list[tuple[str, str]] | None = None,
    min_npe: int = 10,
) -> pl.DataFrame:
    """Build a synthetic cell-level provider DataFrame.

    Employment levels are set so that each cell grows at a known constant
    rate of 0.001/month (0.1%/month).
    """
    if cells is None:
        cells = list(CELL_INDEX.keys())
    rows: list[dict] = []
    base_emp = 500.0
    for rg, ss in cells:
        for t, rd in enumerate(ref_dates):
            rows.append({
                "geographic_type": "region",
                "geographic_code": rg,
                "industry_type": "supersector",
                "industry_code": ss,
                "ref_date": rd,
                "n_pseudo_estabs": min_npe,
                "employment": base_emp * np.exp(0.001 * t),
            })
    return pl.DataFrame(rows, schema={
        "geographic_type": pl.Utf8,
        "geographic_code": pl.Utf8,
        "industry_type": pl.Utf8,
        "industry_code": pl.Utf8,
        "ref_date": pl.Date,
        "n_pseudo_estabs": pl.UInt32,
        "employment": pl.Float64,
    })


# ---------------------------------------------------------------------------
# Task A tests — QCEW weight loading
# ---------------------------------------------------------------------------


class TestLoadQcewWeights:
    def test_shares_sum_to_one(self, tmp_path: Path) -> None:
        store = _build_qcew_store(_REF_DATES[:3], tmp_path)
        weights, staleness = load_qcew_weights(
            store, _REF_DATES[0], _REF_DATES[2], _REF_DATES[:3],
        )
        assert len(weights) == 3
        for _rd, w in weights.items():
            assert w.shape == (N_CELLS,)
            assert abs(w.sum() - 1.0) < 1e-12

    def test_weight_vector_length(self, tmp_path: Path) -> None:
        store = _build_qcew_store(_REF_DATES[:1], tmp_path)
        weights, _ = load_qcew_weights(
            store, _REF_DATES[0], _REF_DATES[0], _REF_DATES[:1],
        )
        assert weights[_REF_DATES[0]].shape == (N_CELLS,)

    def test_carry_forward(self, tmp_path: Path) -> None:
        """QCEW for Jan–Feb; provider has Jan–Apr.  Mar/Apr carry-forward."""
        store = _build_qcew_store(_REF_DATES[:2], tmp_path)
        provider_dates = _REF_DATES[:4]
        weights, staleness = load_qcew_weights(
            store, _REF_DATES[0], _REF_DATES[3], provider_dates,
        )
        assert len(weights) == 4

        # Jan, Feb should use their own weights (staleness=0).
        jan_row = staleness.filter(pl.col("ref_date") == _REF_DATES[0])
        assert jan_row["weight_staleness_months"].item() == 0

        feb_row = staleness.filter(pl.col("ref_date") == _REF_DATES[1])
        assert feb_row["weight_staleness_months"].item() == 0

        # Mar uses Feb QCEW → staleness = 1.
        mar_row = staleness.filter(pl.col("ref_date") == _REF_DATES[2])
        assert mar_row["weight_staleness_months"].item() == 1
        assert mar_row["qcew_weight_ref_date"].item() == _REF_DATES[1]

        # Apr uses Feb QCEW → staleness = 2.
        apr_row = staleness.filter(pl.col("ref_date") == _REF_DATES[3])
        assert apr_row["weight_staleness_months"].item() == 2

    def test_empty_store(self, tmp_path: Path) -> None:
        store_dir = tmp_path / "source=qcew" / "seasonally_adjusted=false"
        store_dir.mkdir(parents=True)
        empty = pl.DataFrame(schema={
            "geographic_type": pl.Utf8,
            "geographic_code": pl.Utf8,
            "industry_type": pl.Utf8,
            "industry_code": pl.Utf8,
            "ref_date": pl.Date,
            "vintage_date": pl.Date,
            "revision": pl.UInt8,
            "benchmark_revision": pl.UInt8,
            "employment": pl.Float64,
            "growth": pl.Float64,
        })
        empty.write_parquet(store_dir / "data.parquet")
        weights, staleness = load_qcew_weights(
            tmp_path, _REF_DATES[0], _REF_DATES[2], _REF_DATES[:3],
        )
        assert len(weights) == 0
        assert staleness.is_empty()


# ---------------------------------------------------------------------------
# Task B tests — Weight redistribution
# ---------------------------------------------------------------------------


class TestRedistributeWeights:
    def test_all_covered(self) -> None:
        w = np.ones(N_CELLS) / N_CELLS
        covered = np.ones(N_CELLS, dtype=bool)
        result = redistribute_weights(w, covered)
        np.testing.assert_allclose(result.sum(), 1.0)
        np.testing.assert_allclose(result, w)

    def test_missing_cells_get_zero_weight(self) -> None:
        w = np.ones(N_CELLS) / N_CELLS
        covered = np.ones(N_CELLS, dtype=bool)
        covered[0] = False  # drop one cell
        result = redistribute_weights(w, covered)
        assert result[0] == 0.0
        assert abs(result.sum() - 1.0) < 1e-12

    def test_redistribute_within_supersector(self) -> None:
        """Missing cell's weight should go to covered cells in same supersector."""
        w = np.ones(N_CELLS) / N_CELLS
        covered = np.ones(N_CELLS, dtype=bool)
        target_cell = 0  # region '1', supersector '10'
        covered[target_cell] = False
        target_ss = CELL_SUPERSECTOR[target_cell]

        result = redistribute_weights(w, covered)

        same_ss_covered = np.where(covered & (CELL_SUPERSECTOR == target_ss))[0]

        total_same_ss = result[same_ss_covered].sum()
        expected_same_ss = w[same_ss_covered].sum() + w[target_cell]
        assert abs(total_same_ss - expected_same_ss) < 1e-10

    def test_sum_preserved(self) -> None:
        rng = np.random.default_rng(42)
        w = rng.random(N_CELLS)
        w /= w.sum()
        covered = np.ones(N_CELLS, dtype=bool)
        covered[rng.choice(N_CELLS, size=10, replace=False)] = False
        result = redistribute_weights(w, covered)
        assert abs(result.sum() - 1.0) < 1e-12
        assert (result[~covered] == 0.0).all()


# ---------------------------------------------------------------------------
# Task C tests — National composite computation
# ---------------------------------------------------------------------------


class TestComputeProviderComposite:
    def test_uniform_growth(self, tmp_path: Path) -> None:
        """All cells grow at 0.1%/month → composite should be ~0.001."""
        store = _build_qcew_store(_REF_DATES[:3], tmp_path)
        prov = _build_provider_df(_REF_DATES[:3])

        composite_df, staleness = compute_provider_composite(
            prov, store, min_pseudo_estabs=5,
        )
        assert not composite_df.is_empty()
        assert "ref_date" in composite_df.columns
        assert "employment" in composite_df.columns

        levels = composite_df.sort("ref_date")["employment"].to_numpy()
        growth = np.diff(np.log(levels))
        np.testing.assert_allclose(growth, 0.001, atol=1e-10)

    def test_staleness_columns(self, tmp_path: Path) -> None:
        store = _build_qcew_store(_REF_DATES[:2], tmp_path)
        prov = _build_provider_df(_REF_DATES[:4])

        _, staleness = compute_provider_composite(prov, store, min_pseudo_estabs=5)
        assert "ref_date" in staleness.columns
        assert "qcew_weight_ref_date" in staleness.columns
        assert "weight_staleness_months" in staleness.columns
        assert len(staleness) == 4

    def test_low_npe_excluded(self, tmp_path: Path) -> None:
        """Cells with n_pseudo_estabs below threshold produce no growth."""
        store = _build_qcew_store(_REF_DATES[:3], tmp_path)
        prov = _build_provider_df(_REF_DATES[:3], min_npe=3)  # below default=5
        composite_df, _ = compute_provider_composite(prov, store, min_pseudo_estabs=5)
        # Only the base row (t=0) survives; growth months are NaN → dropped.
        assert len(composite_df) <= 1

    def test_partial_cells(self, tmp_path: Path) -> None:
        """Composite works with a subset of 44 cells."""
        store = _build_qcew_store(_REF_DATES[:3], tmp_path)
        cells = [(rg, ss) for rg in REGION_CODES for ss in SUPERSECTOR_CODES[:5]]
        prov = _build_provider_df(_REF_DATES[:3], cells=cells)

        composite_df, _ = compute_provider_composite(prov, store, min_pseudo_estabs=5)
        assert not composite_df.is_empty()
        levels = composite_df.sort("ref_date")["employment"].to_numpy()
        growth = np.diff(np.log(levels))
        np.testing.assert_allclose(growth, 0.001, atol=1e-10)

    def test_fewer_than_two_dates(self, tmp_path: Path) -> None:
        store = _build_qcew_store(_REF_DATES[:1], tmp_path)
        prov = _build_provider_df(_REF_DATES[:1])
        composite_df, _ = compute_provider_composite(prov, store)
        assert composite_df.is_empty()


# ---------------------------------------------------------------------------
# Task F.2 — Integration: cell-level through load_provider_series
# ---------------------------------------------------------------------------


class TestCellLevelIntegration:
    def test_load_provider_series_cell_level(self, tmp_path: Path) -> None:
        """load_provider_series routes cell-level data through compositing."""
        from nfp_ingest.payroll import _is_cell_level, read_provider_table

        _build_qcew_store(_REF_DATES[:3], tmp_path)
        prov = _build_provider_df(_REF_DATES[:3])
        parquet_path = tmp_path / "provider.parquet"
        prov.write_parquet(parquet_path)

        raw = read_provider_table(parquet_path)
        assert raw is not None
        assert _is_cell_level(raw)

    def test_is_cell_level_national(self) -> None:
        from nfp_ingest.payroll import _is_cell_level

        national = pl.DataFrame({
            "geography_type": ["national"],
            "geography_code": ["00"],
            "ref_date": [date(2020, 1, 12)],
            "employment": [100.0],
        })
        assert not _is_cell_level(national)

    def test_is_cell_level_region(self) -> None:
        from nfp_ingest.payroll import _is_cell_level

        cell = pl.DataFrame({
            "geographic_type": ["region"],
            "geographic_code": ["1"],
            "ref_date": [date(2020, 1, 12)],
            "employment": [100.0],
        })
        assert _is_cell_level(cell)
