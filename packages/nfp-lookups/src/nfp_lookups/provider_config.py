"""Provider and cyclical indicator configuration dataclasses.

These are shared across ingest and model packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ProviderConfig:
    """Specification for a single payroll-provider series.

    Adding a new provider requires only a new entry in ``PROVIDERS``.
    The model, diagnostics, and plots adapt automatically.

    **National providers** (default) supply a pre-aggregated national time
    series filtered by the ``geography_type`` / ``industry_type`` fields.
    Birth rates may live in the same file (``birth_rate`` column) or in a
    separate parquet pointed to by ``birth_file``.

    **Cell-level providers** supply a supersector x Census-region parquet
    (schema: ``geographic_type``, ``geographic_code``, ``industry_type``,
    ``industry_code``, ``ref_date``, ``n_pseudo_estabs``, ``employment``).
    When the loader detects ``geographic_type='region'`` in the file, it
    performs QCEW-weighted compositing automatically (see
    ``ingest/compositing.py`` and ``provider_spec.md`` S5).  Since cell-level
    files have no birth-rate column, use ``birth_file`` to supply a separate
    national-level birth-rate parquet.
    """

    name: str
    file: str
    error_model: Literal["iid", "ar1"] = "iid"
    birth_file: str | None = None
    industry_type: str = "national"
    industry_code: str = "00"
    geography_type: str = "national"
    geography_code: str = "00"


@dataclass(frozen=True)
class CyclicalIndicator:
    """Specification for a single cyclical indicator."""

    name: str
    fred_id: str
    freq: Literal["weekly", "monthly"]
    pub_lag: int = 1


# Default cyclical indicators
CYCLICAL_INDICATORS_DEFAULT: list[CyclicalIndicator] = [
    CyclicalIndicator(name="claims", fred_id="ICNSA", freq="weekly", pub_lag=1),
    CyclicalIndicator(name="jolts", fred_id="JTSJOL", freq="monthly", pub_lag=2),
]

# Compositing threshold
MIN_PSEUDO_ESTABS_PER_CELL: int = 5
