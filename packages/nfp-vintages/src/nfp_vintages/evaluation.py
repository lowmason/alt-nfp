"""Vintage evaluation: revision analysis and noise multiplier construction.

Provides functions to compute revision differences between vintages and
to build per-observation noise multiplier arrays for the model.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from nfp_lookups.revision_schedules import CES_REVISIONS, QCEW_REVISIONS


def vintage_diff(
    panel: pl.LazyFrame,
    source: str,
    rev_a: int,
    rev_b: int,
) -> pl.LazyFrame:
    """Compute the revision (growth difference) between two vintages.

    Parameters
    ----------
    panel : pl.LazyFrame
        Full observation panel.
    source : str
        Source identifier (e.g., 'ces_sa', 'qcew').
    rev_a : int
        First revision number (typically earlier/noisier).
    rev_b : int
        Second revision number (typically later/more accurate).

    Returns
    -------
    pl.LazyFrame
        Frame with columns: period, industry_code, revision_from, revision_to,
        growth_diff (= growth_b - growth_a).
    """
    view_a = (
        panel.filter(
            (pl.col('source') == source) & (pl.col('revision_number') == rev_a)
        )
        .select('period', 'industry_code', pl.col('growth').alias('growth_a'))
    )

    view_b = (
        panel.filter(
            (pl.col('source') == source) & (pl.col('revision_number') == rev_b)
        )
        .select('period', 'industry_code', pl.col('growth').alias('growth_b'))
    )

    return (
        view_a.join(view_b, on=['period', 'industry_code'], how='inner')
        .with_columns(
            (pl.col('growth_b') - pl.col('growth_a')).alias('growth_diff'),
            pl.lit(rev_a).cast(pl.Int32).alias('revision_from'),
            pl.lit(rev_b).cast(pl.Int32).alias('revision_to'),
        )
        .select('period', 'industry_code', 'revision_from', 'revision_to', 'growth_diff')
        .sort('period', 'industry_code')
    )


def build_noise_multiplier_vector(panel_view: pl.DataFrame) -> np.ndarray:
    """Build a per-observation noise multiplier array from a panel view.

    Looks up each row's (source, revision_number) in the revision schedule
    to determine the appropriate noise scaling factor.

    Parameters
    ----------
    panel_view : pl.DataFrame
        A materialized panel view (output of real_time_view or final_view).

    Returns
    -------
    np.ndarray
        Float array of length len(panel_view), where each entry is the
        noise multiplier for that observation. Suitable for passing to
        the model as observation-level noise scaling.
    """
    n = len(panel_view)
    multipliers = np.ones(n, dtype=np.float64)

    sources = panel_view['source'].to_list()
    rev_nums = panel_view['revision_number'].to_list()

    # Precompute lookup dicts for fast access
    ces_lookup: dict[int, float] = {
        spec.revision_number: spec.noise_multiplier for spec in CES_REVISIONS
    }
    qcew_lookup: dict[str, dict[int, float]] = {}
    for q_label, specs in QCEW_REVISIONS.items():
        qcew_lookup[q_label] = {
            spec.revision_number: spec.noise_multiplier for spec in specs
        }

    # If panel has period info, use it to determine QCEW quarter
    periods = panel_view['period'].to_list() if 'period' in panel_view.columns else [None] * n

    for i in range(n):
        src = sources[i]
        rev = rev_nums[i]

        if rev is None:
            continue

        if src in ('ces_sa', 'ces_nsa'):
            multipliers[i] = ces_lookup.get(rev, 1.0)
        elif src == 'qcew':
            # Determine quarter from period
            period = periods[i]
            if period is not None:
                quarter = (period.month - 1) // 3 + 1
                q_label = f'Q{quarter}'
                q_lookup = qcew_lookup.get(q_label, {})
                multipliers[i] = q_lookup.get(rev, 1.0)
        # Payroll providers: multiplier stays 1.0

    return multipliers
