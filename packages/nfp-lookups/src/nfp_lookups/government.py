"""Government wedge reference data: known interventions + change-space shapes.

Used by the government-wedge forecast (specs/government_wedge.md). The table
carries an ``announcement_date`` axis so backtests can censor to what was
knowable at each release-eve (the lookahead guard is a date comparison).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from nfp_lookups.provider_config import CyclicalIndicator


@dataclass(frozen=True)
class GovIntervention:
    """One deterministic federal shock. Magnitudes are SIGNED, in thousands."""

    ref_month: date          # month-start the effect begins
    name: str
    shape: str               # 'pulse' (level shift) | 'box' (phased ramp) | 'tc' (census)
    magnitude_k: float       # prior MEAN, thousands (negative = job loss)
    magnitude_sd_k: float    # prior SD, thousands
    announcement_date: date  # when it became publicly knowable (the censor key)
    source_url: str
    box_months: int = 1      # 'box' width
    tc_decay: float = 0.5    # 'tc' geometric giveback rate


# PLACEHOLDER priors — the maintainer supplies the real 2025 RIF values
# (spec §8: announced permanent-separation count, honest sd, announcement_date,
# source_url). Placeholder keeps the build unblocked; it must be replaced before
# any accuracy claim.
KNOWN_INTERVENTIONS: list[GovIntervention] = [
    GovIntervention(
        ref_month=date(2025, 3, 1),
        name="federal_rif_2025",
        shape="pulse",
        magnitude_k=-50.0,
        magnitude_sd_k=25.0,
        announcement_date=date(2025, 2, 11),
        source_url="PLACEHOLDER — replace with the RIF announcement URL",
    ),
]


def get_known_interventions_as_of(as_of: date) -> list[GovIntervention]:
    """Interventions knowable on ``as_of`` (announcement_date <= as_of)."""
    return [iv for iv in KNOWN_INTERVENTIONS if iv.announcement_date <= as_of]


def intervention_column(iv: GovIntervention, ref_months: list[date]) -> np.ndarray:
    """Unit change-space basis column (length T) for one intervention.

    A sampled coefficient ``c`` (prior ``N(magnitude_k, magnitude_sd_k)``) times
    this column is the intervention's contribution to the wedge CHANGE. Shapes map
    level-space X-13 events into change-space:
      pulse -> a one-month change (level steps and stays = permanent LS),
      box   -> 1/k over k months (phased ramp),
      tc    -> +1 then geometric givebacks (census bump-and-fade).
    """
    T = len(ref_months)
    col = np.zeros(T, dtype=float)
    if iv.ref_month not in ref_months:
        return col
    t = ref_months.index(iv.ref_month)
    if iv.shape == "pulse":
        col[t] = 1.0
    elif iv.shape == "box":
        k = max(1, iv.box_months)
        for j in range(k):
            if t + j < T:
                col[t + j] = 1.0 / k
    elif iv.shape == "tc":
        col[t] = 1.0
        rho = iv.tc_decay
        j = 1
        while t + j < T:
            col[t + j] = -(1.0 - rho) * (rho ** (j - 1))
            j += 1
    else:
        raise ValueError(f"unknown intervention shape {iv.shape!r}")
    return col


# Candidate FRED ids for government CES SA series — PLAN-SIDE VERIFICATION
# required (spec §3.2): confirm fetchable before relying on them.
GOVERNMENT_INDICATORS: list[CyclicalIndicator] = [
    CyclicalIndicator(name="gov_total", fred_id="CES9000000001", freq="monthly", pub_lag=1),
    CyclicalIndicator(name="gov_federal", fred_id="CES9091000001", freq="monthly", pub_lag=1),
    CyclicalIndicator(name="gov_state", fred_id="CES9092000001", freq="monthly", pub_lag=1),
    CyclicalIndicator(name="gov_local", fred_id="CES9093000001", freq="monthly", pub_lag=1),
]

__all__ = [
    "GovIntervention", "KNOWN_INTERVENTIONS", "get_known_interventions_as_of",
    "intervention_column", "GOVERNMENT_INDICATORS",
]
