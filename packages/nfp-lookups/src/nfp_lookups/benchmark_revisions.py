"""Historical CES benchmark revisions (NSA total nonfarm, 000s).

Each February, BLS publishes the annual benchmark revision alongside the
January Employment Situation report.  CES re-anchors March employment
levels to QCEW-derived near-census counts.  This module stores the
historical revisions and provides a typed lookup helper.

Source: BLS Employment Situation benchmark revision tables.
"""

from __future__ import annotations

# March reference year → NSA total nonfarm revision (000s).
# Negative values indicate CES overstated employment.
BENCHMARK_REVISIONS: dict[int, float | None] = {
    2009: -902,
    2010: -378,
    2011: 229,
    2012: 481,
    2013: 340,
    2014: 105,
    2015: -259,
    2016: -151,
    2017: 136,
    2018: -43,
    2019: -505,
    2020: None,  # COVID — excluded from evaluation
    2021: None,  # COVID — excluded from evaluation
    2022: 462,
    2023: -105,
    2024: -598,
    2025: -862,
}


def get_benchmark_revision(year: int) -> float | None:
    """Look up the benchmark revision for a given March reference year.

    Parameters
    ----------
    year : int
        March reference year (e.g. 2025 for the revision published Feb 2026).

    Returns
    -------
    float or None
        Revision in thousands of jobs, or ``None`` for COVID-excluded years.

    Raises
    ------
    KeyError
        If *year* is not in the lookup table.
    """
    if year not in BENCHMARK_REVISIONS:
        raise KeyError(
            f"No benchmark revision for year {year}. "
            f"Available: {min(BENCHMARK_REVISIONS)}–{max(BENCHMARK_REVISIONS)}"
        )
    return BENCHMARK_REVISIONS[year]
