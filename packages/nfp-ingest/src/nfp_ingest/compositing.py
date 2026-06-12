"""QCEW-weighted national compositing for cell-level provider data.

Implements the representativeness correction from ``provider_spec.md`` §5:
load QCEW employment at supersector × region level from the vintage store,
compute cell-level employment shares (with carry-forward for months beyond
the QCEW frontier), redistribute weights for missing/low-coverage cells,
and produce a single national composite growth series per provider.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical cell grid: 11 supersectors × 4 Census regions = 44 cells
# ---------------------------------------------------------------------------

SUPERSECTOR_CODES: list[str] = [
    "10", "20", "30", "40", "50", "55", "60", "65", "70", "80", "90",
]
REGION_CODES: list[str] = ["1", "2", "3", "4"]
N_CELLS: int = len(REGION_CODES) * len(SUPERSECTOR_CODES)  # 44

# Canonical cell ordering: region 1 × ss 10, region 1 × ss 20, …, region 4 × ss 90.
CELL_INDEX: dict[tuple[str, str], int] = {}
_cell_ss: list[int] = []
_cell_rg: list[int] = []
_i = 0
for _ri, _rg in enumerate(REGION_CODES):
    for _si, _ss in enumerate(SUPERSECTOR_CODES):
        CELL_INDEX[(_rg, _ss)] = _i
        _cell_ss.append(_si)
        _cell_rg.append(_ri)
        _i += 1

CELL_SUPERSECTOR: np.ndarray = np.array(_cell_ss, dtype=np.intp)
CELL_REGION: np.ndarray = np.array(_cell_rg, dtype=np.intp)


def _months_between(later: date, earlier: date) -> int:
    """Non-negative integer months from *earlier* to *later* (clamped at 0)."""
    return max(0, (later.year - earlier.year) * 12 + (later.month - earlier.month))


# ---------------------------------------------------------------------------
# Task A — QCEW weight loading with carry-forward and staleness tracking
# ---------------------------------------------------------------------------


def load_qcew_weights(
    store_path: Path,
    min_ref: date,
    max_ref: date,
    provider_ref_dates: Sequence[date],
) -> tuple[dict[date, np.ndarray], pl.DataFrame]:
    """Load QCEW cell-level employment shares, with carry-forward.

    Parameters
    ----------
    store_path
        Path to the Hive-partitioned vintage store (``data/store``).
    min_ref, max_ref
        Inclusive date range for QCEW loading.
    provider_ref_dates
        Sorted provider ``ref_date`` values that need weights.

    Returns
    -------
    weights_by_date
        ``{provider_ref_date: weight_vector_44}`` — one entry per
        provider month.  Weights sum to 1.0 over the 44 canonical cells.
    staleness_df
        Columns: ``ref_date``, ``qcew_weight_ref_date``,
        ``weight_staleness_months``.
    """
    _empty_staleness = pl.DataFrame(schema={
        "ref_date": pl.Date,
        "qcew_weight_ref_date": pl.Date,
        "weight_staleness_months": pl.Int32,
    })

    qcew = (
        pl.read_parquet(str(store_path / "**/*.parquet"), hive_partitioning=True)
        .filter(
            pl.col("source") == "qcew",
            pl.col("geographic_type") == "region",
            pl.col("industry_type") == "supersector",
            pl.col("ref_date").is_between(min_ref, max_ref),
        )
    )

    if qcew.is_empty():
        logger.warning(
            "No QCEW region×supersector data in store for %s – %s", min_ref, max_ref,
        )
        return {}, _empty_staleness

    # Keep the latest vintage per cell per ref_date.
    qcew_latest = (
        qcew
        .sort("vintage_date", descending=True)
        .group_by(["ref_date", "geographic_code", "industry_code"])
        .agg(pl.col("employment").first())
    )

    # Build weight vectors keyed by QCEW ref_date.
    shares_by_qcew_date: dict[date, np.ndarray] = {}
    for qd, grp in qcew_latest.group_by("ref_date"):
        qd_val: date = qd[0]  # type: ignore[index]
        vec = np.zeros(N_CELLS, dtype=np.float64)
        for row in grp.iter_rows(named=True):
            key = (row["geographic_code"], row["industry_code"])
            idx = CELL_INDEX.get(key)
            if idx is not None:
                vec[idx] = row["employment"]
        total = vec.sum()
        if total > 0:
            shares_by_qcew_date[qd_val] = vec / total
        else:
            logger.warning("QCEW total employment is zero for %s", qd_val)

    if not shares_by_qcew_date:
        logger.warning("All QCEW ref_dates had zero total employment")
        return {}, _empty_staleness

    sorted_qcew_dates = sorted(shares_by_qcew_date)

    # Map each provider ref_date → best available QCEW date (carry-forward).
    weights_by_date: dict[date, np.ndarray] = {}
    staleness_rows: list[dict] = []

    for prd in provider_ref_dates:
        best: date | None = None
        for qd in sorted_qcew_dates:
            if qd <= prd:
                best = qd
            else:
                break
        if best is None:
            best = sorted_qcew_dates[0]
            logger.warning(
                "No QCEW weights at or before %s; using %s (lookahead)", prd, best
            )

        weights_by_date[prd] = shares_by_qcew_date[best]
        staleness_rows.append({
            "ref_date": prd,
            "qcew_weight_ref_date": best,
            "weight_staleness_months": _months_between(prd, best),
        })

    staleness_df = pl.DataFrame(
        staleness_rows,
        schema={
            "ref_date": pl.Date,
            "qcew_weight_ref_date": pl.Date,
            "weight_staleness_months": pl.Int32,
        },
    )
    return weights_by_date, staleness_df


# ---------------------------------------------------------------------------
# Task B — Weight redistribution for missing / low-coverage cells (§5.2)
# ---------------------------------------------------------------------------


def redistribute_weights(
    weights: np.ndarray,
    covered_mask: np.ndarray,
) -> np.ndarray:
    """Redistribute weight from uncovered cells to covered cells.

    For each missing cell ``(s, r)``:

    1. Redistribute proportionally to covered cells in supersector *s*.
    2. If none, redistribute to covered cells in region *r*.
    3. If none, redistribute uniformly to all covered cells.

    The returned vector sums to 1.0 with zero weight on uncovered cells.
    """
    if covered_mask.all():
        return weights.copy()

    result = weights.copy()
    missing_indices = np.where(~covered_mask)[0]

    for ci in missing_indices:
        w = result[ci]
        if w == 0.0:
            continue
        result[ci] = 0.0

        # 1) Same supersector
        same_ss = covered_mask & (CELL_SUPERSECTOR == CELL_SUPERSECTOR[ci])
        if same_ss.any():
            denom = result[same_ss].sum()
            if denom > 0:
                result[same_ss] += w * (result[same_ss] / denom)
                continue

        # 2) Same region
        same_rg = covered_mask & (CELL_REGION == CELL_REGION[ci])
        if same_rg.any():
            denom = result[same_rg].sum()
            if denom > 0:
                result[same_rg] += w * (result[same_rg] / denom)
                continue

        # 3) Uniform fallback
        denom = result[covered_mask].sum()
        if denom > 0:
            result[covered_mask] += w * (result[covered_mask] / denom)
        else:
            n_cov = int(covered_mask.sum())
            if n_cov > 0:
                result[covered_mask] += w / n_cov

    total = result.sum()
    if total > 0:
        result /= total
    return result


# ---------------------------------------------------------------------------
# Task C — National composite computation (§5.1)
# ---------------------------------------------------------------------------


def compute_provider_composite(
    provider_cell_df: pl.DataFrame,
    store_path: Path,
    min_pseudo_estabs: int = 5,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compute QCEW-weighted national composite from cell-level provider data.

    Parameters
    ----------
    provider_cell_df
        Cell-level DataFrame with columns ``geographic_type``,
        ``geographic_code``, ``industry_type``, ``industry_code``,
        ``ref_date``, ``n_pseudo_estabs``, ``employment``.
    store_path
        Path to the Hive-partitioned vintage store.
    min_pseudo_estabs
        Minimum pseudo-establishment count for a cell to be included.

    Returns
    -------
    composite_df
        Columns ``ref_date``, ``employment`` (synthetic national level,
        base = 100, from cumulated composite growth).
    staleness_df
        Columns ``ref_date``, ``qcew_weight_ref_date``,
        ``weight_staleness_months``.
    """
    _empty_composite = pl.DataFrame(
        schema={"ref_date": pl.Date, "employment": pl.Float64},
    )
    _empty_staleness = pl.DataFrame(schema={
        "ref_date": pl.Date,
        "qcew_weight_ref_date": pl.Date,
        "weight_staleness_months": pl.Int32,
    })

    df = provider_cell_df.sort("ref_date")
    ref_dates: list[date] = sorted(df["ref_date"].unique().to_list())

    if len(ref_dates) < 2:
        logger.warning("Provider has < 2 ref_dates; cannot compute composite")
        return _empty_composite, _empty_staleness

    # Load QCEW weights (with carry-forward for recent months).
    weights_by_date, staleness_df = load_qcew_weights(
        store_path, ref_dates[0], ref_dates[-1], ref_dates,
    )
    if not weights_by_date:
        logger.warning("No QCEW weights available; cannot compute composite")
        return _empty_composite, staleness_df

    # Build cell-level matrices: employment[T, 44] and n_pseudo_estabs[T, 44].
    n_dates = len(ref_dates)
    date_idx = {d: i for i, d in enumerate(ref_dates)}

    emp = np.full((n_dates, N_CELLS), np.nan)
    npe = np.zeros((n_dates, N_CELLS), dtype=np.uint32)

    for row in df.iter_rows(named=True):
        key = (row["geographic_code"], row["industry_code"])
        ci = CELL_INDEX.get(key)
        if ci is None:
            continue
        ti = date_idx[row["ref_date"]]
        emp[ti, ci] = row["employment"]
        npe[ti, ci] = row["n_pseudo_estabs"]

    # Cell-level log-difference growth: growth[t] = ln(emp[t]) - ln(emp[t-1]).
    with np.errstate(divide="ignore", invalid="ignore"):
        log_emp = np.log(emp)
    growth = np.full((n_dates, N_CELLS), np.nan)
    growth[1:] = log_emp[1:] - log_emp[:-1]

    # Composite growth per month.
    composite_growth = np.full(n_dates, np.nan)

    for t in range(1, n_dates):
        rd = ref_dates[t]
        raw_w = weights_by_date.get(rd)
        if raw_w is None:
            continue

        finite_growth = np.isfinite(growth[t])
        enough_npe_curr = npe[t] >= min_pseudo_estabs
        enough_npe_prev = npe[t - 1] >= min_pseudo_estabs
        covered = finite_growth & enough_npe_curr & enough_npe_prev

        if not covered.any():
            logger.warning("No covered cells for %s", rd)
            continue

        adj_w = redistribute_weights(raw_w, covered)
        composite_growth[t] = float(
            np.nansum(adj_w * np.where(covered, growth[t], 0.0))
        )

    # Synthesize a national employment level (base = 100) so the existing
    # provider loading path can log-difference it to recover growth.
    level = np.full(n_dates, np.nan)
    level[0] = 100.0
    for t in range(1, n_dates):
        if np.isfinite(composite_growth[t]) and np.isfinite(level[t - 1]):
            level[t] = level[t - 1] * np.exp(composite_growth[t])

    composite_df = (
        pl.DataFrame({"ref_date": ref_dates, "employment": level})
        .filter(pl.col("employment").is_not_nan() & pl.col("employment").is_not_null())
    )
    return composite_df, staleness_df
