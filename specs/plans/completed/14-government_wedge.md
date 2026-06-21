# Government Wedge Forecast — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **✅ IMPLEMENTED — all 10 tasks complete (verified 2026-06-19).** Committed on `a5-rebuilt-integration` (commits `69c1200`→`458c6c4`). Checkboxes were ticked after a per-task verification that **ran each task's tests**: fast suite **770 passed / 2 skipped**, the 15 wedge unit tests green, ruff clean, and the Task-9 build gate **ran against the real store and converged** (0 divergences, finite μ, R-hat ≤ 1.01). **Build-here / validate-on-port:** the accuracy verdict vs consensus, the 2025-RIF intervention magnitudes (placeholder priors — maintainer input gate, §8 of the spec), and the Bloomberg consensus file remain for the port. Sanctioned deviations: `_wedge_diag.py` kept as a spec reference (not `git rm`'d); `cmd_total` added skip-guards beyond the plan draft; `decomposition_residual` uses `ddof=1`. The cross-cutting specs (`model_improvements.md`, `a5_real_competitors.md`, `bloomberg_consensus.md`, `plans/13`, `plans/0`) were reconciled to this built state in commit `8326621`.

**Goal:** Forecast the government wedge `g = published_00 − published_05` (first-print MoM change, SA, thousands) with a small Bayesian model, assemble it with the existing private nowcast into a Total-NFP posterior, and score that Total against the Bloomberg consensus.

**Architecture:** A standalone NumPyro change-space structural time-series (`nfp_model/wedge.py`) = constant drift + a shrunk monthly-seasonal block + a deterministic, announcement-priored intervention layer + masked iid-Normal likelihood. The wedge target and intervention basis are built data-side (`nfp_ingest/wedge_data.py`) from store first prints + a government reference table (`nfp_lookups/government.py`). A harness helper (`nfp_vintages/assembly.py`) convolves the wedge posterior with the private nowcast posterior; the A5 backtest is extended to score the assembled Total vs consensus + the Total first print.

**Tech Stack:** Python 3.12, JAX + NumPyro (float64), Polars, numpy. uv workspace.

## Global Constraints

- **Units are thousands of jobs.** CES `change_k` is already in thousands, so the wedge change is numeric O(±100), std ≈ 24. ALL priors are in thousands-units (`drift ~ Normal(0, 50)`, `τ_season ~ HalfNormal(30)`, `sigma ~ HalfNormal(30)`). Never write `30_000`.
- **Import boundary:** `nfp_model/wedge.py` imports **only** jax/numpyro/numpy — no `nfp_*` package. It reimplements the mask idiom inline via `numpyro.handlers.mask` (do **not** import `nfp_model.model._maybe_mask`).
- **No new A3 parity baseline:** the wedge is a *separate* model; it does not touch `model.py`/`nowcast.py`.
- **Mask, never delete rows:** COVID 2020–21 and the Oct-2025 no-print hole are masked to keep the calendar axis contiguous.
- **Lookahead is a DATE comparison:** interventions enter a fit only when `announcement_date ≤ as_of`. Never encode realized post-hoc shock sizes per ref-month.
- **Build gate = convergence + sane posterior, NOT accuracy.** The accuracy verdict vs consensus is deferred to the Bloomberg port.
- **Store `.env` gotcha:** ad-hoc reads must `load_dotenv('.env')` or they hit the empty local store. Store tests self-skip when the store is unavailable (mark `@pytest.mark.real_store` / network as the package does).
- Lint: ruff line length 100. Run tests with `uv run pytest -m "not network" --no-cov`.

Spec: `specs/completed/government_wedge.md`. Rationale: `docs/government_design.md`.

---

## File Structure

- `packages/nfp-lookups/src/nfp_lookups/government.py` **(create)** — `GovIntervention` dataclass, `KNOWN_INTERVENTIONS` (announcement-dated table, placeholder 2025 RIF), `get_known_interventions_as_of`, `intervention_column` (change-space shape encoders), `GOVERNMENT_INDICATORS` (FRED entries). Pure reference + numpy.
- `packages/nfp-ingest/src/nfp_ingest/wedge_data.py` **(create)** — `wedge_first_print_changes` (the `00−05` join), `build_wedge_model_data(as_of, target_month)` (the model input dict), `read_government_signal` (diagnostic reader).
- `packages/nfp-model/src/nfp_model/wedge.py` **(create)** — `wedge_model`, `WEDGE_DETERMINISTIC_SITES`, `fit_wedge`, `wedge_pred_draws`. Imports only jax/numpyro/numpy.
- `packages/nfp-vintages/src/nfp_vintages/assembly.py` **(create)** — `assemble_total`.
- `packages/nfp-vintages/src/nfp_vintages/wedge_diagnostics.py` **(create)** — `decomposition_residual`, `calibrate_intervention_sd`.
- `scripts/run_a5_backtest.py` **(modify)** — add `cmd_total` (wedge fit + assemble + Total scoring).
- `packages/nfp-*/tests/...` and `packages/nfp-vintages/tests/fixtures/` **(create)** — unit tests + two consensus fixtures.

**Cross-task interfaces (locked here):**
- `GovIntervention(ref_month: date, name: str, shape: str, magnitude_k: float, magnitude_sd_k: float, announcement_date: date, source_url: str, box_months: int = 1, tc_decay: float = 0.5)`
- `get_known_interventions_as_of(as_of: date) -> list[GovIntervention]`
- `intervention_column(iv: GovIntervention, ref_months: list[date]) -> np.ndarray` (length T)
- `wedge_first_print_changes(*, store_path=VINTAGE_STORE_PATH) -> pl.DataFrame` (cols `ref_date, chg00, chg05, wedge_change_k`)
- `build_wedge_model_data(*, as_of: date | None, target_month: date, store_path=VINTAGE_STORE_PATH, start=date(2017,1,1)) -> dict` with keys `y, month_of_year, T, mask, X_intervention, iv_prior_mean, iv_prior_sd, ref_months, target_idx`
- `wedge_model(data: dict)` — NumPyro model; deterministic sites `WEDGE_DETERMINISTIC_SITES = ("mu", "season")`
- `fit_wedge(data, *, settings="default", seed=0) -> FitResult` (reuse `nfp_model.sampling.FitResult`)
- `wedge_pred_draws(fit, target_idx: int, *, seed=0) -> np.ndarray` (length N = chains*draws, change-k)
- `assemble_total(private_growth_draws, wedge_change_draws, *, prev_index, idx_to_level, eta=0.0, seed=0) -> np.ndarray` (length N_wedge, change-k)

---

## Task 1: Government intervention reference data

**Files:**
- Create: `packages/nfp-lookups/src/nfp_lookups/government.py`
- Test: `packages/nfp-lookups/tests/test_government.py`

**Interfaces:**
- Produces: `GovIntervention`, `KNOWN_INTERVENTIONS`, `get_known_interventions_as_of`, `intervention_column`.

- [x] **Step 1: Write the failing test**

```python
# packages/nfp-lookups/tests/test_government.py
from datetime import date
import numpy as np
from nfp_lookups.government import (
    GovIntervention, KNOWN_INTERVENTIONS,
    get_known_interventions_as_of, intervention_column,
)

REF = [date(2025, m, 1) for m in range(1, 7)]  # Jan..Jun 2025

def test_as_of_filters_on_announcement_date():
    from datetime import timedelta
    rif = next(i for i in KNOWN_INTERVENTIONS if i.name == "federal_rif_2025")
    assert rif not in get_known_interventions_as_of(rif.announcement_date - timedelta(days=1))
    assert rif in get_known_interventions_as_of(rif.announcement_date)

def test_pulse_is_permanent_level_shift():
    iv = GovIntervention(date(2025, 3, 1), "rif", "pulse", -50.0, 25.0,
                         date(2025, 2, 11), "u")
    col = intervention_column(iv, REF)
    assert col.tolist() == [0, 0, 1, 0, 0, 0]          # one-month change
    assert np.isclose(np.cumsum(col)[-1], 1.0)          # level steps and STAYS

def test_box_is_phased_ramp():
    iv = GovIntervention(date(2025, 3, 1), "rif", "box", -60.0, 30.0,
                         date(2025, 2, 11), "u", box_months=3)
    col = intervention_column(iv, REF)
    assert np.allclose(col, [0, 0, 1/3, 1/3, 1/3, 0])
    assert np.isclose(np.cumsum(col)[-1], 1.0)          # total ramp = 1 unit

def test_tc_peaks_then_decays_back_toward_zero():
    iv = GovIntervention(date(2025, 3, 1), "census", "tc", 400.0, 50.0,
                         date(2025, 1, 1), "u", tc_decay=0.5)
    col = intervention_column(iv, REF)
    assert col[2] == 1.0 and col[3] < 0                 # +peak then giveback
    assert np.cumsum(col)[-1] < 1.0                     # decays back down

def test_missing_ref_month_is_all_zero():
    iv = GovIntervention(date(2099, 1, 1), "future", "pulse", 1.0, 1.0,
                         date(2099, 1, 1), "u")
    assert np.all(intervention_column(iv, REF) == 0)
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/nfp-lookups/tests/test_government.py -v`
Expected: FAIL — `ModuleNotFoundError: nfp_lookups.government`.

- [x] **Step 3: Write the implementation**

```python
# packages/nfp-lookups/src/nfp_lookups/government.py
"""Government wedge reference data: known interventions + change-space shapes.

Used by the government-wedge forecast (specs/completed/government_wedge.md). The table
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
```

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/nfp-lookups/tests/test_government.py -v`
Expected: PASS (5 tests).

- [x] **Step 5: Lint + commit**

```bash
uv run ruff check packages/nfp-lookups/src/nfp_lookups/government.py
git add packages/nfp-lookups/src/nfp_lookups/government.py packages/nfp-lookups/tests/test_government.py
git commit -m "feat(lookups): government intervention reference data (wedge Track B)"
```

---

## Task 2: Wedge first-print target

**Files:**
- Create: `packages/nfp-ingest/src/nfp_ingest/wedge_data.py`
- Test: `packages/nfp-ingest/tests/test_wedge_data.py`

**Interfaces:**
- Consumes: `nfp_ingest.first_print.first_print_changes` (keyword-only after `store_path`; cols `ref_date, first_print_growth, first_print_change_k, vintage_date`).
- Produces: `wedge_first_print_changes(*, store_path=VINTAGE_STORE_PATH) -> pl.DataFrame`.

- [x] **Step 1: Write the failing test** (uses a monkeypatched `first_print_changes` so it does not need the real store)

```python
# packages/nfp-ingest/tests/test_wedge_data.py
from datetime import date
import polars as pl
import pytest
from nfp_ingest import wedge_data


def _fp(codes):  # build a fake first_print_changes keyed by industry_code
    tbl = {
        "00": pl.DataFrame({"ref_date": [date(2025, 1, 1), date(2025, 2, 1)],
                            "first_print_growth": [0.001, 0.001],
                            "first_print_change_k": [150.0, 160.0],
                            "vintage_date": [date(2025, 2, 6), date(2025, 3, 6)]}),
        "05": pl.DataFrame({"ref_date": [date(2025, 1, 1), date(2025, 2, 1)],
                            "first_print_growth": [0.001, 0.001],
                            "first_print_change_k": [130.0, 145.0],
                            "vintage_date": [date(2025, 2, 6), date(2025, 3, 6)]}),
    }
    def fake(*, store_path=None, geographic_type="national", geographic_code="00",
             industry_type="total", industry_code="00"):
        return tbl[industry_code]
    return fake


def test_wedge_is_00_minus_05(monkeypatch):
    monkeypatch.setattr(wedge_data, "first_print_changes", _fp(("00", "05")))
    df = wedge_data.wedge_first_print_changes()
    assert df["wedge_change_k"].to_list() == [20.0, 15.0]   # 150-130, 160-145


def test_mismatched_vintage_raises(monkeypatch):
    bad = _fp(("00", "05"))
    def fake(*, industry_code="00", **kw):
        d = bad(industry_code=industry_code, **kw)
        if industry_code == "05":  # drift one leg's vintage out of the release window
            d = d.with_columns(vintage_date=pl.Series([date(2025, 5, 1), date(2025, 6, 1)]))
        return d
    monkeypatch.setattr(wedge_data, "first_print_changes", fake)
    with pytest.raises(ValueError, match="vintage"):
        wedge_data.wedge_first_print_changes()
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/nfp-ingest/tests/test_wedge_data.py -v`
Expected: FAIL — `ModuleNotFoundError: nfp_ingest.wedge_data`.

- [x] **Step 3: Write the implementation**

```python
# packages/nfp-ingest/src/nfp_ingest/wedge_data.py
"""Government-wedge model inputs (specs/completed/government_wedge.md).

The wedge target g = 00 - 05 first-print change comes from the store; the
intervention basis comes from nfp_lookups.government, censored by as_of.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
from nfp_lookups.paths import VINTAGE_STORE_PATH

from nfp_ingest.first_print import first_print_changes

# Largest tolerated gap between the 00 and 05 first-print release stamps. The
# rebuilt store staggers a release's revisions across ~1 week; 00 and 05 rev0 are
# co-released, so their stamps must be within one release window.
_RELEASE_WINDOW_DAYS = 15


def wedge_first_print_changes(*, store_path: Path = VINTAGE_STORE_PATH) -> pl.DataFrame:
    """Per ref_date: wedge_change_k = first_print(00) - first_print(05), same release.

    Returns columns ``ref_date, chg00, chg05, wedge_change_k`` sorted by ref_date.
    Raises ``ValueError`` if the two legs are not from the same release window.
    """
    fp00 = first_print_changes(store_path=store_path, industry_type="total",
                               industry_code="00").select(
        "ref_date", pl.col("first_print_change_k").alias("chg00"),
        pl.col("vintage_date").alias("v00"))
    fp05 = first_print_changes(store_path=store_path, industry_type="total",
                               industry_code="05").select(
        "ref_date", pl.col("first_print_change_k").alias("chg05"),
        pl.col("vintage_date").alias("v05"))
    df = fp00.join(fp05, on="ref_date", how="inner").sort("ref_date").drop_nulls(
        subset=["chg00", "chg05"])
    gap = (df["v00"] - df["v05"]).dt.total_days().abs()
    if (gap > _RELEASE_WINDOW_DAYS).any():
        raise ValueError(
            "wedge legs not from same release: 00/05 vintage_date gap exceeds "
            f"{_RELEASE_WINDOW_DAYS}d — refusing a cross-vintage difference")
    return df.with_columns(
        (pl.col("chg00") - pl.col("chg05")).alias("wedge_change_k")
    ).select("ref_date", "chg00", "chg05", "wedge_change_k")
```

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/nfp-ingest/tests/test_wedge_data.py -v`
Expected: PASS (2 tests).

- [x] **Step 5: Commit**

```bash
git add packages/nfp-ingest/src/nfp_ingest/wedge_data.py packages/nfp-ingest/tests/test_wedge_data.py
git commit -m "feat(ingest): wedge first-print target (00-05, same-vintage guard)"
```

---

## Task 3: Wedge ModelData builder (with lookahead guard)

**Files:**
- Modify: `packages/nfp-ingest/src/nfp_ingest/wedge_data.py`
- Test: `packages/nfp-ingest/tests/test_wedge_data.py`

**Interfaces:**
- Consumes: `wedge_first_print_changes`, `nfp_lookups.government.get_known_interventions_as_of` + `intervention_column`.
- Produces: `build_wedge_model_data(*, as_of, target_month, store_path=VINTAGE_STORE_PATH, start=date(2017,1,1)) -> dict` with keys `y, month_of_year, T, mask, X_intervention, iv_prior_mean, iv_prior_sd, ref_months, target_idx`.

- [x] **Step 1: Write the failing test** (monkeypatch the target so no store is needed)

```python
# append to packages/nfp-ingest/tests/test_wedge_data.py
from datetime import date as _d
import numpy as np
from nfp_lookups.government import GovIntervention


def _wedge_df():
    months = [_d(2025, m, 1) for m in range(1, 5)]            # Jan..Apr 2025
    return pl.DataFrame({"ref_date": months, "chg00": [0.0]*4, "chg05": [0.0]*4,
                         "wedge_change_k": [10.0, 12.0, -40.0, 8.0]})


def test_build_masks_target_and_builds_axis(monkeypatch):
    monkeypatch.setattr(wedge_data, "wedge_first_print_changes", lambda **k: _wedge_df())
    monkeypatch.setattr(wedge_data, "get_known_interventions_as_of", lambda a: [])
    d = wedge_data.build_wedge_model_data(
        as_of=_d(2025, 3, 20), target_month=_d(2025, 4, 1), start=_d(2025, 1, 1))
    assert d["T"] == 4 and d["target_idx"] == 3
    assert d["month_of_year"].tolist() == [1, 2, 3, 4]   # preserved for ALL rows
    assert d["mask"][3] == False                          # target month not observed
    assert d["mask"][:3].all()                            # history observed


def test_lookahead_guard_excludes_unannounced_intervention(monkeypatch):
    monkeypatch.setattr(wedge_data, "wedge_first_print_changes", lambda **k: _wedge_df())
    rif = GovIntervention(_d(2025, 3, 1), "rif", "pulse", -40.0, 20.0,
                          announcement_date=_d(2025, 3, 10), source_url="u")
    # as_of BEFORE the announcement → no intervention column
    monkeypatch.setattr(wedge_data, "get_known_interventions_as_of",
                        lambda a: [rif] if a >= _d(2025, 3, 10) else [])
    before = wedge_data.build_wedge_model_data(
        as_of=_d(2025, 3, 5), target_month=_d(2025, 4, 1), start=_d(2025, 1, 1))
    assert before["X_intervention"].shape[1] == 0
    after = wedge_data.build_wedge_model_data(
        as_of=_d(2025, 3, 15), target_month=_d(2025, 4, 1), start=_d(2025, 1, 1))
    assert after["X_intervention"].shape[1] == 1
    assert after["iv_prior_mean"].tolist() == [-40.0]
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/nfp-ingest/tests/test_wedge_data.py -k build -v`
Expected: FAIL — `AttributeError: build_wedge_model_data`.

- [x] **Step 3: Write the implementation** (append to `wedge_data.py`)

```python
# add to imports at top of wedge_data.py
import numpy as np
from nfp_lookups.government import get_known_interventions_as_of, intervention_column

# COVID and the Oct-2025 shutdown no-print hole are masked, never deleted.
_COVID = (date(2020, 1, 1), date(2021, 12, 1))
_SHUTDOWN_HOLE = {date(2025, 10, 1)}


def _month_range(start: date, end: date) -> list[date]:
    out, y, m = [], start.year, start.month
    while date(y, m, 1) <= end:
        out.append(date(y, m, 1))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def build_wedge_model_data(
    *,
    as_of: date | None,
    target_month: date,
    store_path: Path = VINTAGE_STORE_PATH,
    start: date = date(2017, 1, 1),
) -> dict:
    """Assemble the wedge model input dict for a release-eve nowcast of target_month.

    The target month's own row is present but masked (its first print is the scored
    actual, revealed at release). Interventions are censored to ``as_of`` via the
    announcement-date guard. Returns plain numpy arrays (no Polars reaches JAX).
    """
    wedge = wedge_first_print_changes(store_path=store_path)
    known = {r["ref_date"]: r["wedge_change_k"]
             for r in wedge.iter_rows(named=True)}
    # As-of censor: a wedge month is observed only if its first print is published
    # by as_of. We approximate the first-print publish date by the month's own
    # release (~5 weeks after ref month-start); the harness passes a release-eve
    # as_of, so target_month and anything not-yet-released is excluded.
    ref_months = _month_range(start, target_month)
    T = len(ref_months)
    y = np.full(T, np.nan)
    for i, rm in enumerate(ref_months):
        if rm == target_month:
            continue  # masked: scored actual, not an input
        v = known.get(rm)
        if v is not None and (as_of is None or rm < target_month):
            y[i] = v
    month_of_year = np.array([rm.month for rm in ref_months], dtype=int)
    mask = np.isfinite(y).copy()
    for i, rm in enumerate(ref_months):
        if (_COVID[0] <= rm <= _COVID[1]) or rm in _SHUTDOWN_HOLE:
            mask[i] = False
    y = np.nan_to_num(y, nan=0.0)  # masked entries contribute zero log-prob

    ivs = get_known_interventions_as_of(as_of) if as_of is not None else []
    cols = [intervention_column(iv, ref_months) for iv in ivs]
    X = np.stack(cols, axis=1) if cols else np.zeros((T, 0))
    iv_prior_mean = np.array([iv.magnitude_k for iv in ivs], dtype=float)
    iv_prior_sd = np.array([iv.magnitude_sd_k for iv in ivs], dtype=float)

    return {
        "y": y, "month_of_year": month_of_year, "T": T, "mask": mask,
        "X_intervention": X, "iv_prior_mean": iv_prior_mean, "iv_prior_sd": iv_prior_sd,
        "ref_months": ref_months, "target_idx": ref_months.index(target_month),
    }
```

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/nfp-ingest/tests/test_wedge_data.py -v`
Expected: PASS (4 tests).

- [x] **Step 5: Commit**

```bash
git add packages/nfp-ingest/src/nfp_ingest/wedge_data.py packages/nfp-ingest/tests/test_wedge_data.py
git commit -m "feat(ingest): build_wedge_model_data with announcement-date lookahead guard"
```

---

## Task 4: The NumPyro wedge model

**Files:**
- Create: `packages/nfp-model/src/nfp_model/wedge.py`
- Test: `packages/nfp-model/tests/test_wedge_model.py`

**Interfaces:**
- Produces: `wedge_model(data: dict)`, `WEDGE_DETERMINISTIC_SITES = ("mu", "season")`.

- [x] **Step 1: Write the failing test**

```python
# packages/nfp-model/tests/test_wedge_model.py
import numpy as np
import jax
import numpyro
from numpyro.infer import Predictive
from nfp_model.wedge import wedge_model, WEDGE_DETERMINISTIC_SITES

numpyro.enable_x64()


def _data(T=24, K=1):
    rng = np.random.default_rng(0)
    return {
        "y": rng.normal(10, 24, T), "month_of_year": np.array([(i % 12) + 1 for i in range(T)]),
        "T": T, "mask": np.ones(T, bool),
        "X_intervention": np.zeros((T, K)), "iv_prior_mean": np.zeros(K),
        "iv_prior_sd": np.full(K, 20.0),
    }


def test_priors_run_and_sum_to_zero():
    d = _data()
    pred = Predictive(wedge_model, num_samples=8)(jax.random.PRNGKey(0), data=d)
    season = np.asarray(pred["season"])           # (8, 12)
    assert season.shape[1] == 12
    assert np.allclose(season.sum(axis=1), 0.0, atol=1e-8)   # sum-to-zero pin
    assert np.asarray(pred["mu"]).shape == (8, d["T"])


def test_units_are_thousands():
    d = _data()
    pred = Predictive(wedge_model, num_samples=200)(jax.random.PRNGKey(1), data=d)
    # drift ~ Normal(0,50): O(tens), NOT tens-of-thousands
    assert np.abs(np.asarray(pred["mu"])).mean() < 1000
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/nfp-model/tests/test_wedge_model.py -v`
Expected: FAIL — `ModuleNotFoundError: nfp_model.wedge`.

- [x] **Step 3: Write the implementation**

```python
# packages/nfp-model/src/nfp_model/wedge.py
"""Standalone Bayesian government-wedge model (specs/completed/government_wedge.md).

Imports ONLY jax/numpyro/numpy (no nfp_* package). Models the wedge MoM CHANGE
directly in change-space (units: thousands of jobs):

    mu_t = drift + season[month_t] + X_intervention @ coef
    y_t ~ Normal(mu_t, sigma)            (masked over COVID + the Oct-2025 hole)
"""
from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

WEDGE_DETERMINISTIC_SITES = ("mu", "season")


def wedge_model(data: dict) -> None:
    T = int(data["T"])
    moy = jnp.asarray(data["month_of_year"]) - 1          # 0..11
    X = jnp.asarray(data["X_intervention"])               # (T, K)
    K = X.shape[1]

    drift = numpyro.sample("drift", dist.Normal(0.0, 50.0))

    # 11 free monthly effects, 12th pinned by sum-to-zero (deterministic).
    tau = numpyro.sample("tau_season", dist.HalfNormal(30.0))
    s_raw = numpyro.sample("season_raw", dist.Normal(0.0, 1.0).expand([11]))
    s11 = s_raw * tau                                      # non-centered
    season = numpyro.deterministic(
        "season", jnp.concatenate([s11, -s11.sum()[None]]))   # length 12, sums to 0

    if K > 0:
        pm = jnp.asarray(data["iv_prior_mean"])
        ps = jnp.asarray(data["iv_prior_sd"])
        coef = numpyro.sample("iv_coef", dist.Normal(pm, ps).to_event(1))
        iv = X @ coef
    else:
        iv = jnp.zeros(T)

    mu = numpyro.deterministic("mu", drift + season[moy] + iv)
    sigma = numpyro.sample("sigma", dist.HalfNormal(30.0))

    mask = jnp.asarray(data["mask"], dtype=bool)
    with numpyro.handlers.mask(mask=mask):                 # inline idiom (no nfp_* import)
        numpyro.sample("y_obs", dist.Normal(mu, sigma), obs=jnp.asarray(data["y"]))
```

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/nfp-model/tests/test_wedge_model.py -v`
Expected: PASS (2 tests).

- [x] **Step 5: Verify the import boundary, then commit**

Run: `uv run python -c "import ast,sys; m=ast.parse(open('packages/nfp-model/src/nfp_model/wedge.py').read()); bad=[n for n in ast.walk(m) if isinstance(n,(ast.Import,ast.ImportFrom)) and 'nfp_' in (getattr(n,'module',None) or ''.join(a.name for a in getattr(n,'names',[])))]; sys.exit(1 if bad else 0)"`
Expected: exit 0 (no `nfp_*` imports).

```bash
git add packages/nfp-model/src/nfp_model/wedge.py packages/nfp-model/tests/test_wedge_model.py
git commit -m "feat(model): standalone NumPyro government-wedge model (change-space STS)"
```

---

## Task 5: Wedge fit + predictive draws

**Files:**
- Modify: `packages/nfp-model/src/nfp_model/wedge.py`
- Modify: `packages/nfp-model/src/nfp_model/__init__.py` (export `fit_wedge`, `wedge_pred_draws`)
- Test: `packages/nfp-model/tests/test_wedge_fit.py` (marked `slow`)

**Interfaces:**
- Consumes: `nfp_model.sampling.FitResult`.
- Produces: `fit_wedge(data, *, settings="default", seed=0) -> FitResult`; `wedge_pred_draws(fit, target_idx, *, seed=0) -> np.ndarray`.

- [x] **Step 1: Write the failing test**

```python
# packages/nfp-model/tests/test_wedge_fit.py
import numpy as np
import pytest
from nfp_model.wedge import fit_wedge, wedge_pred_draws


@pytest.mark.slow
def test_fit_recovers_drift_and_predicts_target():
    rng = np.random.default_rng(0)
    T = 60
    moy = np.array([(i % 12) + 1 for i in range(T)])
    true_season = np.array([0, 5, 30, -10, -5, 20, 25, 10, -15, -20, 0, -40], float)
    true_season -= true_season.mean()
    y = 8.0 + true_season[moy - 1] + rng.normal(0, 20, T)
    mask = np.ones(T, bool); mask[-1] = False           # last = target, unobserved
    data = {"y": np.where(mask, y, 0.0), "month_of_year": moy, "T": T, "mask": mask,
            "X_intervention": np.zeros((T, 0)), "iv_prior_mean": np.zeros(0),
            "iv_prior_sd": np.zeros(0)}
    fit = fit_wedge(data, settings="light", seed=0)
    assert fit.num_divergences == 0
    drift_mean = fit.posterior["drift"].mean()
    assert 8.0 - 12 < drift_mean < 8.0 + 12             # recovered, loosely
    draws = wedge_pred_draws(fit, target_idx=T - 1, seed=1)
    assert draws.ndim == 1 and draws.shape[0] == (
        fit.posterior["drift"].shape[0] * fit.posterior["drift"].shape[1])
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/nfp-model/tests/test_wedge_fit.py -v`
Expected: FAIL — `ImportError: cannot import name 'fit_wedge'`.

- [x] **Step 3: Write the implementation** (append to `wedge.py`)

```python
# add to wedge.py imports
import time
import jax
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from nfp_model.config import PRESETS  # SamplerSettings preset table (jax-free dataclass)
from nfp_model.sampling import FitResult


def fit_wedge(data: dict, *, settings="default", seed: int = 0, progress: bool = False) -> FitResult:
    """NUTS fit of the wedge model. Mirrors fit_model's packaging (FitResult)."""
    numpyro.enable_x64()
    if isinstance(settings, str):
        settings = PRESETS[settings]
    kernel = NUTS(wedge_model, target_accept_prob=settings.target_accept,
                  max_tree_depth=settings.max_tree_depth,
                  init_strategy=init_to_median(num_samples=15))
    mcmc = MCMC(kernel, num_warmup=settings.num_warmup, num_samples=settings.num_samples,
                num_chains=settings.num_chains, chain_method=settings.chain_method,
                progress_bar=progress)
    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(seed), data=data, extra_fields=("diverging",))
    wall = time.time() - t0
    post = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
    flat = mcmc.get_samples()
    dets = Predictive(wedge_model, posterior_samples=flat,
                      return_sites=list(WEDGE_DETERMINISTIC_SITES))(
        jax.random.PRNGKey(0), data=data)
    nc, nd = settings.num_chains, settings.num_samples
    for k, v in dets.items():
        arr = np.asarray(v)
        post[k] = arr.reshape(nc, nd, *arr.shape[1:])
    div = int(np.asarray(mcmc.get_extra_fields(group_by_chain=True)["diverging"]).sum())
    return FitResult(posterior=post, num_divergences=div, settings=settings,
                     seed=seed, wall_seconds=wall)


def wedge_pred_draws(fit: FitResult, target_idx: int, *, seed: int = 0) -> np.ndarray:
    """Posterior predictive of the wedge first-print CHANGE at target_idx (length N).

    The predictive includes observation noise (mu + Normal(0, sigma)), mirroring the
    private nowcast's first-print predictive, so the two convolve like-for-like.
    """
    mu = fit.posterior["mu"][..., target_idx].reshape(-1)     # (N,)
    sigma = fit.posterior["sigma"].reshape(-1)                # (N,)
    eps = np.random.default_rng(seed).standard_normal(mu.shape[0])
    return mu + sigma * eps
```

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/nfp-model/tests/test_wedge_fit.py -v -m slow`
Expected: PASS (1 test; ~tens of seconds).

- [x] **Step 5: Export + commit**

Add to `packages/nfp-model/src/nfp_model/__init__.py`: `from .wedge import fit_wedge, wedge_pred_draws, wedge_model`.

```bash
git add packages/nfp-model/src/nfp_model/wedge.py packages/nfp-model/src/nfp_model/__init__.py packages/nfp-model/tests/test_wedge_fit.py
git commit -m "feat(model): fit_wedge + wedge_pred_draws (predictive change draws)"
```

---

## Task 6: Total assembly helper

**Files:**
- Create: `packages/nfp-vintages/src/nfp_vintages/assembly.py`
- Test: `packages/nfp-vintages/tests/test_assembly.py`

**Interfaces:**
- Consumes: `nfp_vintages.scoreboard.change_draws_k`.
- Produces: `assemble_total(private_growth_draws, wedge_change_draws, *, prev_index, idx_to_level, eta=0.0, seed=0) -> np.ndarray`.

- [x] **Step 1: Write the failing test**

```python
# packages/nfp-vintages/tests/test_assembly.py
import numpy as np
from nfp_vintages.assembly import assemble_total


def test_alignment_and_pure_sum_eta0():
    priv_growth = np.zeros((2, 50))                 # growth 0 -> change 0
    wedge = np.full((3, 100), 7.0)                  # change-k draws
    total = assemble_total(priv_growth, wedge, prev_index=1000.0, idx_to_level=1.0, eta=0.0)
    assert total.shape == (300,)                    # N = wedge chains*draws
    assert np.allclose(total, 7.0)                  # 0 (private) + 7 (wedge)
    assert not np.isnan(total).any()


def test_coupling_is_point_invariant():
    rng = np.random.default_rng(0)
    priv_growth = rng.normal(0.001, 0.0005, (4, 100))
    wedge = rng.normal(10.0, 20.0, (4, 100))
    base = assemble_total(priv_growth, wedge, prev_index=1500.0, idx_to_level=1.0, eta=0.0)
    coup = assemble_total(priv_growth, wedge, prev_index=1500.0, idx_to_level=1.0, eta=0.5, seed=0)
    assert abs(base.mean() - coup.mean()) < 1.0     # mean-zero z -> point ~invariant
    assert coup.std() > base.std()                  # intervals widen
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/nfp-vintages/tests/test_assembly.py -v`
Expected: FAIL — `ModuleNotFoundError: nfp_vintages.assembly`.

- [x] **Step 3: Write the implementation**

```python
# packages/nfp-vintages/src/nfp_vintages/assembly.py
"""Assemble a Total-NFP change posterior from the private nowcast + wedge.

Total = private + wedge is exact by construction (we forecast the wedge directly).
The private leg is growth/index space and must be converted to change-k first.
"""
from __future__ import annotations

import numpy as np

from nfp_vintages.scoreboard import change_draws_k


def assemble_total(
    private_growth_draws: np.ndarray,
    wedge_change_draws: np.ndarray,
    *,
    prev_index: float,
    idx_to_level: float,
    eta: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Element-wise Total first-print change draws (thousands).

    N = wedge draw count (wedge authoritative); the private change draws are
    resampled to N. ``eta`` enables the (default-off) residual coupling: adds
    ``eta * z`` to the wedge leg, z = standardized mean-zero private residual —
    point-invariant, widens intervals only.
    """
    priv = change_draws_k(private_growth_draws, prev_index=prev_index,
                          idx_to_level=idx_to_level)        # flattened
    wedge = np.asarray(wedge_change_draws, float).reshape(-1)
    n = wedge.shape[0]
    rng = np.random.default_rng(seed)
    priv_n = rng.choice(priv, size=n, replace=True) if priv.shape[0] != n else priv
    total = priv_n + wedge
    if eta:
        z = (priv_n - priv_n.mean()) / (priv_n.std() or 1.0)
        total = total + eta * z
    if np.isnan(total).any():
        raise ValueError("assemble_total produced NaN — check prev_index/idx_to_level anchor")
    return total
```

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/nfp-vintages/tests/test_assembly.py -v`
Expected: PASS (2 tests).

- [x] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/assembly.py packages/nfp-vintages/tests/test_assembly.py
git commit -m "feat(vintages): assemble_total (private+wedge -> Total posterior)"
```

---

## Task 7: Consensus fixtures + Total scoring path

**Files:**
- Create: `packages/nfp-vintages/tests/fixtures/consensus_populated.parquet` (built by the test setup)
- Test: `packages/nfp-vintages/tests/test_total_scoring.py`
- Modify: `packages/nfp-vintages/src/nfp_vintages/assembly.py` (add `score_total`)

**Interfaces:**
- Consumes: `assemble_total`, `nfp_vintages.competitors.consensus.Consensus`, `nfp_vintages.scoreboard.crps_sample` / `interval_coverage`.
- Produces: `score_total(total_change_draws, *, first_print_k, consensus_k) -> dict` (cols `crps, cover80, point_err, consensus_err|None`).

- [x] **Step 1: Write the failing test** (covers BOTH the null/absent and populated consensus paths)

```python
# packages/nfp-vintages/tests/test_total_scoring.py
from datetime import date
import numpy as np
import polars as pl
import pytest
from nfp_vintages.assembly import score_total
from nfp_vintages.competitors.consensus import Consensus, load_consensus


def test_absent_consensus_renders_none(tmp_path):
    assert load_consensus(tmp_path / "missing.parquet") is None
    c = Consensus(None)
    assert c.predict(date(2025, 4, 1), as_of=date(2025, 5, 1)) is None
    row = score_total(np.full(500, 100.0), first_print_k=110.0, consensus_k=None)
    assert row["consensus_err"] is None                 # column renders "—"


def test_populated_consensus_scores(tmp_path):
    p = tmp_path / "consensus_populated.parquet"
    pl.DataFrame({
        "ref_month": [date(2025, 4, 1)], "consensus_median_change_k": [150.0],
        "survey_date": [date(2025, 5, 1)], "release_date": [date(2025, 5, 2)],
        "source": ["synthetic"],
    }).write_parquet(p)
    c = Consensus(load_consensus(p))
    cons = c.predict(date(2025, 4, 1), as_of=date(2025, 5, 1))
    assert cons == 150.0
    row = score_total(np.full(500, 120.0), first_print_k=110.0, consensus_k=cons)
    assert np.isclose(row["point_err"], 10.0)           # |120 - 110|
    assert np.isclose(row["consensus_err"], 40.0)       # |150 - 110|
    assert 0.0 <= row["cover80"] <= 1.0
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/nfp-vintages/tests/test_total_scoring.py -v`
Expected: FAIL — `ImportError: cannot import name 'score_total'`.

- [x] **Step 3: Write the implementation** (append to `assembly.py`)

```python
# add to assembly.py imports
from nfp_vintages.scoreboard import crps_sample, interval_coverage


def score_total(total_change_draws, *, first_print_k, consensus_k=None) -> dict:
    """Score the assembled Total against the Total first print (+ consensus if present)."""
    draws = np.asarray(total_change_draws, float).reshape(-1)
    point = float(draws.mean())
    return {
        "crps": crps_sample(draws, first_print_k),
        "cover80": float(interval_coverage(draws, first_print_k, level=0.80)),
        "point_err": abs(point - first_print_k),
        "consensus_err": (None if consensus_k is None else abs(consensus_k - first_print_k)),
    }
```

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/nfp-vintages/tests/test_total_scoring.py -v`
Expected: PASS (2 tests).

- [x] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/assembly.py packages/nfp-vintages/tests/test_total_scoring.py
git commit -m "feat(vintages): score_total + consensus null/populated paths"
```

---

## Task 8: Government signal diagnostics (decomposition + RIF calibration)

**Files:**
- Create: `packages/nfp-vintages/src/nfp_vintages/wedge_diagnostics.py`
- Test: `packages/nfp-vintages/tests/test_wedge_diagnostics.py`

**Interfaces:**
- Produces: `decomposition_residual(wedge_change, gov90_change) -> dict` (stats on `r = wedge − 90`); `calibrate_intervention_sd(observed_federal_change, baseline_sd) -> float`.

- [x] **Step 1: Write the failing test**

```python
# packages/nfp-vintages/tests/test_wedge_diagnostics.py
import numpy as np
from nfp_vintages.wedge_diagnostics import decomposition_residual, calibrate_intervention_sd


def test_decomposition_residual_small_when_wedge_tracks_90():
    rng = np.random.default_rng(0)
    g90 = rng.normal(15, 20, 80)
    wedge = g90 + rng.normal(0, 3, 80)          # small SA-additivity residual r
    out = decomposition_residual(wedge, g90)
    assert out["r_std"] < out["wedge_std"]      # r is a small fraction of wedge variance
    assert abs(out["r_mean"]) < 5


def test_calibrate_intervention_sd_is_robust():
    fed = np.array([-45.0, -52.0, -48.0])       # observed federal moves around a RIF
    sd = calibrate_intervention_sd(fed, baseline_sd=25.0)
    assert sd > 0
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/nfp-vintages/tests/test_wedge_diagnostics.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [x] **Step 3: Write the implementation**

```python
# packages/nfp-vintages/src/nfp_vintages/wedge_diagnostics.py
"""Diagnostics for the government wedge (specs/completed/government_wedge.md §3.2/§7).

These NEVER enter the model likelihood; they validate the wedge decomposition
and calibrate intervention priors from public government data.
"""
from __future__ import annotations

import numpy as np


def decomposition_residual(wedge_change, gov90_change) -> dict:
    """r = wedge - published_90: the SA-additivity residual we target the wedge to escape."""
    w = np.asarray(wedge_change, float)
    g = np.asarray(gov90_change, float)
    r = w - g
    return {"r_mean": float(r.mean()), "r_std": float(r.std()),
            "wedge_std": float(w.std()), "r_share": float(r.std() / (w.std() or 1.0))}


def calibrate_intervention_sd(observed_federal_change, baseline_sd: float) -> float:
    """An honest prior sd for a federal-shock magnitude: max(empirical spread, baseline)."""
    obs = np.asarray(observed_federal_change, float)
    return float(max(obs.std(ddof=1) if obs.size > 1 else 0.0, baseline_sd))
```

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/nfp-vintages/tests/test_wedge_diagnostics.py -v`
Expected: PASS (2 tests).

- [x] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/wedge_diagnostics.py packages/nfp-vintages/tests/test_wedge_diagnostics.py
git commit -m "feat(vintages): wedge decomposition + intervention-sd calibration diagnostics"
```

---

## Task 9: Wire the Total backtest into the A5 harness (build gate)

**Files:**
- Modify: `scripts/run_a5_backtest.py`
- Test: `packages/nfp-vintages/tests/test_total_backtest_smoke.py` (marked `slow` + `real_store`)

**Interfaces:**
- Consumes: everything above + the existing private grid (`nowcast_pred_draws`, `provenance.base_index/idx_to_level`).

- [x] **Step 1: Write the failing test** (an end-to-end build-gate check; self-skips without store)

```python
# packages/nfp-vintages/tests/test_total_backtest_smoke.py
import os
import numpy as np
import pytest
from dotenv import load_dotenv

pytestmark = [pytest.mark.slow, pytest.mark.real_store]


@pytest.mark.skipif(not os.environ.get("NFP_STORE_URI"), reason="needs real store")
def test_wedge_fit_converges_on_clean_window():
    load_dotenv(".env")
    from datetime import date
    from nfp_ingest.wedge_data import build_wedge_model_data
    from nfp_model.wedge import fit_wedge
    data = build_wedge_model_data(as_of=None, target_month=date(2026, 1, 1))
    fit = fit_wedge(data, settings="default", seed=0)
    # Build gate: convergence, not accuracy.
    assert fit.num_divergences == 0
    assert np.isfinite(fit.posterior["mu"]).all()
```

- [x] **Step 2: Run test to verify it fails (or skips without store)**

Run: `uv run pytest packages/nfp-vintages/tests/test_total_backtest_smoke.py -v`
Expected: SKIP locally without `NFP_STORE_URI`; on the store box, FAIL until the harness wiring lands, then PASS.

- [x] **Step 3: Add `cmd_total` to `scripts/run_a5_backtest.py`**

```python
# scripts/run_a5_backtest.py — add a new subcommand that reuses the private grid.
def cmd_total(root):
    """Fit the wedge per target, assemble Total, score vs first print + consensus."""
    from datetime import date
    import numpy as np
    from nfp_ingest.wedge_data import build_wedge_model_data
    from nfp_model.wedge import fit_wedge, wedge_pred_draws
    from nfp_vintages.assembly import assemble_total, score_total
    from nfp_vintages.competitors.consensus import Consensus, load_consensus

    manifest = _read_json(root / "grid_manifest.json")
    prov = manifest["provenance"]
    consensus = Consensus(load_consensus())
    rows = {}
    for rname in REGIMES:
        reg = manifest["regimes"][rname]
        for key, t in sorted(reg["targets"].items()):
            if "error" in t:
                continue
            target = date.fromisoformat(key)
            as_of = date.fromisoformat(t["as_of"])
            # private leg: persisted nowcast_pred_draws from the batched private fit
            batched = np.load(root / f"{rname}_batched_{key}.npz")
            priv_growth = batched["nowcast_pred_draws"]                # growth/index
            # wedge leg: as-of-censored fit for this release-eve
            wdata = build_wedge_model_data(as_of=as_of, target_month=target)
            wfit = fit_wedge(wdata, settings=PRESET, seed=BATCH_SEED)
            wedge = wedge_pred_draws(wfit, wdata["target_idx"], seed=BATCH_SEED)
            total = assemble_total(priv_growth, wedge,
                                   prev_index=float(t["prev_index"]),
                                   idx_to_level=float(prov["idx_to_level"]))
            # Scored actual = the Total (00) first print, stored at grid-build time
            # (see the Step-3 note). KeyError if missing — never silently fall back to
            # the 05 first print, which would score Total against the wrong actual.
            cons = consensus.predict(target, as_of=as_of)
            rows[f"{rname}:{key}"] = score_total(
                total, first_print_k=t["total_first_print_k"], consensus_k=cons)
    _write_json(root / "total_scores.json", rows)
    print(f"Scored {len(rows)} Total targets → {root/'total_scores.json'}")
```

Note: extend the grid-build step (cmd_grid, ~line 185) to also store `total_first_print_k = first_print_changes(industry_code="00")` per target (the scored actual), mirroring the existing `first_print_change_k` line; register `cmd_total` in the CLI dispatch (`if cmd == "total": cmd_total(root)`).

- [x] **Step 4: Run the build gate on the store box**

Run (on the store box): `NFP_STORE_URI=… uv run pytest packages/nfp-vintages/tests/test_total_backtest_smoke.py -v -m "slow and real_store"`
Expected: PASS — wedge fit converges (0 divergences, finite `mu`). Then `uv run python scripts/run_a5_backtest.py total <grid_root>` produces `total_scores.json`.

- [x] **Step 5: Commit**

```bash
git add scripts/run_a5_backtest.py packages/nfp-vintages/tests/test_total_backtest_smoke.py
git commit -m "feat(eval): Total backtest — wedge fit + assemble + score vs consensus"
```

---

## Task 10: Spec housekeeping + suite green

**Files:**
- Modify: `specs/completed/government_wedge.md` (none expected; verify), `scripts/_wedge_diag.py` / `scripts/_store_layout.py` (remove throwaways)

- [x] **Step 1:** Remove throwaway diagnostics: `git rm -f scripts/_wedge_diag.py scripts/_store_layout.py scripts/_validate_alfred.py` (confirm none are imported: `grep -rn "_wedge_diag\|_store_layout\|_validate_alfred" packages scripts`). — **Done with sanctioned deviation** (commit `c6edb42`): `_store_layout.py` dropped, but `_wedge_diag.py` was deliberately **kept** as the spec's working-read reference (`specs/completed/government_wedge.md` §2/§10). `_validate_alfred.py` is an untracked scratch from a *different* session (never committed). None are imported.
- [x] **Step 2:** Run the fast suite: `uv run pytest -m "not network and not slow" --no-cov` — Expected: PASS (new wedge unit tests included; store/slow tests skip).
- [x] **Step 3:** Lint: `uv run ruff check .` — Expected: clean.
- [x] **Step 4:** Commit: `git commit -am "chore: remove throwaway wedge diagnostics; suite green"`.

---

## Self-Review

**Spec coverage** (each spec section → task):
- §2 target → Task 2; §3.1 → Task 2; §3.2 signals artifact/Census → Task 1 (`GOVERNMENT_INDICATORS`) + Task 8 diagnostics (Census `read_census_table` is dormant/2030 — noted as deferred, not built); §4 model → Task 4; §4.3 interventions → Tasks 1+3; §5 assembly → Task 6; §6 scoring + 2 fixtures → Task 7; §7 signals-not-likelihood → enforced (signals never imported into `wedge.py`/`wedge_data` likelihood); §8 RIF open input → Task 1 placeholder + flagged; §9 phasing/build gate → Task 9; §10 tests → distributed; §11 boundary → Task 4 Step 5 guard; §12 risks → mitigations in Tasks 3 (lookahead), 6 (anchor/NaN), 4 (sum-to-zero/funnel).
- **Gap noted:** the Census `read_census_table()` is *specified but deferred to 2030* (spec §3.2/§9) — intentionally NOT built in v1; the dormant slot is exercised only structurally. The drift-anchor (rejected) and component likelihood (port) are correctly absent.

**Placeholder scan:** the only "PLACEHOLDER" strings are the 2025 RIF priors in `KNOWN_INTERVENTIONS` — this is the deliberate, spec-§8 human-input gate (real code around it), not a plan gap.

**Type consistency:** `wedge_pred_draws` returns length-N change-k → consumed by `assemble_total(wedge_change_draws=...)`; `build_wedge_model_data` keys (`X_intervention`, `iv_prior_mean/sd`, `target_idx`, `mask`, `month_of_year`) match `wedge_model`/`fit_wedge`/`wedge_pred_draws` consumption; `change_draws_k(prev_index=, idx_to_level=)` signature matches Task 6/9 calls; `Consensus.predict(ref_month, *, as_of)` matches Task 7/9.
