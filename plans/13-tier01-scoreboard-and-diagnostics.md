# Tier 0 + Tier 1 — Scoreboard Correctness & Diagnostics Gate (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the A5 evaluation harness with a regime-decomposed, calibration-aware scoreboard (Tier 0) and a diagnostics suite — Aruoba revision regression, Mincer–Zarnowitz efficiency, provider-ablation — whose outputs gate the model-side Tiers 2/3 (Tier 1).

**Architecture:** Pure, unit-testable logic lives in two new `nfp-vintages` modules (`scoreboard.py`, `diagnostics.py`); the `scripts/` files stay thin CLIs that plumb store/manifest data into those functions. No `nfp-model` code is touched (firewall, `specs/model_improvements.md` §2/§8) — calibration is computed from the posterior draws the batch step *already* persists (`nowcast_pred_draws`). OLS is hand-rolled on `numpy`/`scipy` (no statsmodels in the workspace).

**Tech Stack:** Python 3.12, polars, numpy, scipy (`scipy.stats.chi2` for the MZ Wald test), the existing `nfp_ingest` / `nfp_vintages` data APIs. Tests with pytest (markers `slow`, `real_store`).

**Source of truth:** `specs/model_improvements.md` §3 (Tier 0), §4 (Tier 1), §8 (parity governance); evidence base `specs/model_research.md`; the harness being extended is `scripts/run_a5_backtest.py` + `specs/a5_real_competitors.md`.

---

## Firewall & scope (read before starting)

- **NO changes to `packages/nfp-model`.** Not `model.py`, not `config.py`, not `batch.py`, not `nowcast.py`. Tier 0/1 are evaluation-side only (`specs/model_improvements.md` §2). Tiers 2/3 (model changes behind new parity baselines) are **out of scope** for this plan.
- **NO changes to `transform_to_panel`, `build_model_data`, or any A1/A2/A3 golden-mastered path** (`specs/a5_real_competitors.md` firewall).
- **Read-only on the vintage store.** Never call a store-writing function in a test (see `store-write-test-safety`). Store-touching tests are marked `@pytest.mark.real_store` and self-skip when the store is unavailable.
- **Skeleton vs full venue is first-class.** Locally (public store) there is no ADP/NFCI/consensus data; diagnostics run on the public regressor set (claims, JOLTS, lagged revisions, cyclical state) and every output is venue-tagged. A providerless local result is *expected*, never a failure (`specs/model_improvements.md` §10).

## File structure

| File | New / Modify | Responsibility |
|---|---|---|
| `packages/nfp-vintages/src/nfp_vintages/scoreboard.py` | **Create** | Tier 0 pure helpers: month-type classification, calibration (coverage, CRPS), predictive change-draws extraction, venue tag. |
| `packages/nfp-vintages/src/nfp_vintages/diagnostics.py` | **Create** | Tier 1 pure helpers: OLS, revision table, Aruoba design + regression, Mincer–Zarnowitz, gate decision. |
| `packages/nfp-vintages/tests/test_scoreboard.py` | **Create** | Unit tests for `scoreboard.py`. |
| `packages/nfp-vintages/tests/test_diagnostics.py` | **Create** | Unit tests for `diagnostics.py` (pure helpers + `real_store` revision-table test). |
| `scripts/run_a5_backtest.py` | **Modify** (`cmd_score`, ~159–239) | Wire Tier 0: regime × month-type decomposition, calibration + venue columns, second QCEW scoreboard. |
| `scripts/run_tier1_diagnostics.py` | **Create** | Tier 1 CLI: run Aruoba + MZ + provider-ablation, write `tier1_diagnostics.md`/`.parquet`, print the gate decision. |
| `plans/0-port_and_staged_plan.md` | **Modify** (gate log) | Record Tier 0/1 completion. |

Module placement rationale: `nfp-vintages` is top of the data chain and may import `nfp_ingest` (`first_print_changes`, store readers); `nfp-model`'s import boundary does **not** apply to it. Scripts importing `nfp_vintages` + `nfp_ingest` + `scipy` is allowed (only `nfp-model`'s own `src/` is boundary-checked).

## Execution order & dependencies

Tasks are **grouped by concern** (Tier 0 / Tier 1), but a few have dependencies that cross the numeric order — the revision table is shared infrastructure that Tier 0's month-classification and Tier 1's Aruoba LHS both need. Execute in this dependency-correct order:

**1 → 2 → 3 → 6 → 7 → 4 → 5 → 8 → 9 → 10 → 11 → 12 → 13.**

That is: build the pure `scoreboard.py` helpers (1–3), then the `diagnostics.py` OLS + revision table (6–7), *then* wire Tier 0's `cmd_score` (4) and the second scoreboard (5), then the remaining Tier 1 diagnostics (8–11), the CLI (12), and verification (13). Each task below carries an explicit **Depends on** line where its prerequisites are not simply "all lower-numbered tasks."

---

# Phase A — Tier 0: scoreboard correctness

## Task 1: Month-type classifier

Classify each reference month into `normal` / `large_revision` / `turning_point` / `benchmark_window` per `specs/model_improvements.md` §3 (lines 40–41). Pure function over a revision series + a claims-momentum series + config.

**Files:**
- Create: `packages/nfp-vintages/src/nfp_vintages/scoreboard.py`
- Test: `packages/nfp-vintages/tests/test_scoreboard.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/nfp-vintages/tests/test_scoreboard.py
from datetime import date

import numpy as np

from nfp_vintages.scoreboard import MonthTypeConfig, classify_month_types


def test_classify_month_types_buckets():
    # ref months: 6 normal-ish, one big revision, one Feb (benchmark window),
    # one with a sharp claims jump (turning point).
    ref = [date(2023, m, 1) for m in range(1, 13)]
    # |first->third revision| in thousands; index 6 (July) is the large one.
    revision_abs = np.array([5, 8, 6, 7, 9, 4, 120, 10, 6, 5, 8, 7], dtype=float)
    # claims 3-month momentum (level change); index 9 (Oct) spikes.
    claims_mom = np.array([1, -2, 0, 1, -1, 2, 1, 0, 1, 60, 2, -1], dtype=float)
    cfg = MonthTypeConfig(
        large_revision_pctl=90.0,
        claims_momentum_k=40.0,
        benchmark_months=(2,),
    )

    out = classify_month_types(ref, revision_abs, claims_mom, cfg)

    assert out[date(2023, 7, 1)] == "large_revision"   # 120 > p90
    assert out[date(2023, 2, 1)] == "benchmark_window"  # February
    assert out[date(2023, 10, 1)] == "turning_point"    # claims spike >= 40
    assert out[date(2023, 1, 1)] == "normal"
    # Precedence: a month that is both large-revision and benchmark is labeled
    # by the first matching rule (large_revision wins — it is the rarer signal).
    assert set(out.values()) <= {
        "normal", "large_revision", "turning_point", "benchmark_window"
    }


def test_classify_month_types_precedence_large_over_benchmark():
    ref = [date(2023, 2, 1), date(2023, 3, 1)]
    revision_abs = np.array([200.0, 5.0])
    claims_mom = np.array([0.0, 0.0])
    cfg = MonthTypeConfig(large_revision_pctl=90.0, claims_momentum_k=40.0,
                          benchmark_months=(2,))
    out = classify_month_types(ref, revision_abs, claims_mom, cfg)
    assert out[date(2023, 2, 1)] == "large_revision"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/nfp-vintages/tests/test_scoreboard.py -q --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'nfp_vintages.scoreboard'`.

- [ ] **Step 3: Write minimal implementation**

```python
# packages/nfp-vintages/src/nfp_vintages/scoreboard.py
"""Tier 0 scoreboard helpers — regime decomposition, calibration, venue tag.

Evaluation-side only; imports no nfp-model code. See specs/model_improvements.md
section 3 and plans/13-tier01-scoreboard-and-diagnostics.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

MonthType = str  # "normal" | "large_revision" | "turning_point" | "benchmark_window"


@dataclass(frozen=True)
class MonthTypeConfig:
    """Operational definitions for the four month buckets (spec section 3)."""

    large_revision_pctl: float = 90.0          # |first->third| above this pctl
    claims_momentum_k: float = 40.0            # |3mo claims change| (thousands) for a turn
    benchmark_months: tuple[int, ...] = (2,)   # Feb-release annual-benchmark window


def classify_month_types(
    ref_months: list[date],
    revision_abs_k: np.ndarray,
    claims_momentum_k: np.ndarray,
    cfg: MonthTypeConfig,
) -> dict[date, MonthType]:
    """Map each ref month to one bucket.

    Precedence (rarest signal wins): large_revision > turning_point >
    benchmark_window > normal. ``revision_abs_k`` is |first-print - later-print|
    in thousands; ``claims_momentum_k`` is the absolute 3-month change in initial
    claims (thousands), aligned index-for-index with ``ref_months``.
    """
    revision_abs_k = np.asarray(revision_abs_k, dtype=float)
    claims_momentum_k = np.asarray(claims_momentum_k, dtype=float)
    finite = revision_abs_k[np.isfinite(revision_abs_k)]
    thresh = np.percentile(finite, cfg.large_revision_pctl) if finite.size else np.inf

    out: dict[date, MonthType] = {}
    for i, m in enumerate(ref_months):
        rev = revision_abs_k[i]
        mom = abs(claims_momentum_k[i]) if np.isfinite(claims_momentum_k[i]) else 0.0
        if np.isfinite(rev) and rev >= thresh:
            out[m] = "large_revision"
        elif mom >= cfg.claims_momentum_k:
            out[m] = "turning_point"
        elif m.month in cfg.benchmark_months:
            out[m] = "benchmark_window"
        else:
            out[m] = "normal"
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/nfp-vintages/tests/test_scoreboard.py -q --no-cov`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/scoreboard.py packages/nfp-vintages/tests/test_scoreboard.py
git commit -m "feat(eval): month-type classifier for Tier 0 regime decomposition"
```

## Task 2: Calibration metrics — coverage and CRPS

Pure metrics on a predictive sample. `interval_coverage` returns whether the actual falls in the central interval; `crps_sample` is the standard sample-based CRPS estimator (Gneiting–Raftery energy form).

**Files:**
- Modify: `packages/nfp-vintages/src/nfp_vintages/scoreboard.py`
- Test: `packages/nfp-vintages/tests/test_scoreboard.py`

- [ ] **Step 1: Write the failing test**

```python
# append to packages/nfp-vintages/tests/test_scoreboard.py
from nfp_vintages.scoreboard import crps_sample, interval_coverage


def test_interval_coverage_central_interval():
    draws = np.linspace(-100.0, 100.0, 2001)  # symmetric around 0
    assert interval_coverage(draws, actual=0.0, level=0.80) is True
    assert interval_coverage(draws, actual=95.0, level=0.80) is False   # outside p10..p90
    assert interval_coverage(draws, actual=95.0, level=0.95) is True    # inside p2.5..p97.5


def test_crps_point_mass_is_absolute_error():
    # CRPS of a degenerate (point-mass) forecast == |forecast - actual|.
    draws = np.full(500, 10.0)
    assert crps_sample(draws, actual=13.0) == pytest_approx(3.0)


def test_crps_smaller_when_sharper_and_centered():
    actual = 0.0
    sharp = np.random.default_rng(0).normal(0.0, 5.0, 4000)
    wide = np.random.default_rng(0).normal(0.0, 50.0, 4000)
    assert crps_sample(sharp, actual) < crps_sample(wide, actual)
```

Add at the top of the test file:
```python
from pytest import approx as pytest_approx
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/nfp-vintages/tests/test_scoreboard.py -q --no-cov`
Expected: FAIL — `ImportError: cannot import name 'crps_sample'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to packages/nfp-vintages/src/nfp_vintages/scoreboard.py
def interval_coverage(draws: np.ndarray, actual: float, level: float) -> bool:
    """True iff ``actual`` lies in the central ``level`` predictive interval."""
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0 or not np.isfinite(actual):
        return False
    lo = np.percentile(d, 100.0 * (1.0 - level) / 2.0)
    hi = np.percentile(d, 100.0 * (1.0 + level) / 2.0)
    return bool(lo <= actual <= hi)


def crps_sample(draws: np.ndarray, actual: float) -> float:
    """Sample-based CRPS (energy form): E|X-y| - 0.5 E|X-X'|.

    Lower is better. For a point-mass forecast this reduces to |forecast - y|.
    """
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n == 0 or not np.isfinite(actual):
        return float("nan")
    term1 = np.abs(d - actual).mean()
    # 0.5 * mean_{i,j} |x_i - x_j| via the sorted-array O(n log n) identity.
    s = np.sort(d)
    i = np.arange(1, n + 1)
    # mean pairwise abs diff = (2 / n^2) * sum_i (2i - n - 1) * s_i
    mean_pairwise = (2.0 / (n * n)) * np.sum((2 * i - n - 1) * s)
    return float(term1 - 0.5 * mean_pairwise)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/nfp-vintages/tests/test_scoreboard.py -q --no-cov`
Expected: PASS (5 tests total).

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/scoreboard.py packages/nfp-vintages/tests/test_scoreboard.py
git commit -m "feat(eval): interval-coverage and sample-CRPS calibration metrics"
```

## Task 3: Predictive change-draws extraction + venue tag

Convert the persisted `nowcast_pred_draws` (per-draw predicted log-growth at the target month) into a predictive distribution of the first-print change in thousands, using the linearization `change_k ≈ prev_index · (exp(g) − 1) · idx_to_level` (consistent with the batch reduction in `batch.py:316–325`). `venue_for` tags a row `full` vs `public-only`.

**Files:**
- Modify: `packages/nfp-vintages/src/nfp_vintages/scoreboard.py`
- Test: `packages/nfp-vintages/tests/test_scoreboard.py`

- [ ] **Step 1: Write the failing test**

```python
# append to packages/nfp-vintages/tests/test_scoreboard.py
from nfp_vintages.scoreboard import change_draws_k, venue_for


def test_change_draws_linearization():
    # tiny growth draws around g; with prev_index and idx_to_level the change
    # draws must match prev_index*(exp(g)-1)*idx_to_level elementwise.
    g = np.array([0.001, 0.002, -0.0005])
    prev_index = 150_000.0
    idx_to_level = 1.0  # 1 index point == 1k jobs in this fixture
    out = change_draws_k(g, prev_index=prev_index, idx_to_level=idx_to_level)
    expected = prev_index * (np.exp(g) - 1.0) * idx_to_level
    assert np.allclose(out, expected)


def test_change_draws_flattens_chains_draws():
    g2d = np.zeros((2, 50))  # (chains, draws)
    out = change_draws_k(g2d, prev_index=150_000.0, idx_to_level=1.0)
    assert out.shape == (100,)


def test_venue_for():
    assert venue_for(providers_present=True) == "full"
    assert venue_for(providers_present=False) == "public-only"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/nfp-vintages/tests/test_scoreboard.py -q --no-cov`
Expected: FAIL — `ImportError: cannot import name 'change_draws_k'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to packages/nfp-vintages/src/nfp_vintages/scoreboard.py
def change_draws_k(
    pred_growth_draws: np.ndarray, *, prev_index: float, idx_to_level: float
) -> np.ndarray:
    """Predictive sample of the first-print change (thousands) from growth draws.

    Mirrors the batch nowcast arithmetic (batch.py reduction): a month-over-month
    change of growth g multiplies the prior index, change_k = prev*(exp(g)-1)*scale.
    Uses the mean prior index as a fixed scale (a first-order linearization that is
    exact to O(g^2); adequate for interval coverage / CRPS).
    """
    g = np.asarray(pred_growth_draws, dtype=float).reshape(-1)
    return prev_index * (np.exp(g) - 1.0) * idx_to_level


def venue_for(*, providers_present: bool) -> str:
    """Tag a scored row by information regime (spec section 3)."""
    return "full" if providers_present else "public-only"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/nfp-vintages/tests/test_scoreboard.py -q --no-cov`
Expected: PASS (8 tests total).

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/scoreboard.py packages/nfp-vintages/tests/test_scoreboard.py
git commit -m "feat(eval): predictive change-draws extraction and venue tag"
```

## Task 4: Wire Tier 0 into `cmd_score`

Extend `scripts/run_a5_backtest.py::cmd_score` to: classify each ref month, decompose every metric by `(regime × month_type)`, add calibration (coverage_80, coverage_90, crps) for the model from the persisted draws, stamp a `venue` per row, flag shutdown-frontier months, and keep the COVID exclusion. This is integration; the pure pieces are already unit-tested.

**Depends on:** Tasks 1–3 (`scoreboard.py`) and Tasks 6–7 (`diagnostics.ols`, `build_revision_table`). Do those first.

**Files:**
- Modify: `scripts/run_a5_backtest.py` (`cmd_score`, lines 159–239; imports near 159)

- [ ] **Step 1: Add a claims-momentum helper + shutdown flag set near the top of the file**

Insert after the constants block (after `REGIMES = {...}`, ~line 26):

```python
# Months delayed/distorted by the 2025 government shutdown — flagged, not pooled
# (see memory ces-oct2025-shutdown; specs/model_improvements.md section 3).
SHUTDOWN_FLAGGED = frozenset({date(2025, 10, 1), date(2025, 9, 1)})


def _claims_momentum_k() -> dict[date, float]:
    """3-month change in monthly initial claims (thousands), keyed by month-start.

    Returns {} if the claims indicator is absent locally (skeleton venue)."""
    from nfp_ingest.indicators import read_indicator

    df = read_indicator("claims")
    if df is None or df.is_empty():
        return {}
    import polars as pl

    monthly = (
        df.with_columns(pl.col("ref_date").dt.truncate("1mo").alias("m"))
        .group_by("m").agg(pl.col("value").mean().alias("v"))
        .sort("m")
        .with_columns((pl.col("v") - pl.col("v").shift(3)).alias("mom3"))
    )
    return {r["m"]: (r["mom3"] / 1000.0 if r["mom3"] is not None else float("nan"))
            for r in monthly.iter_rows(named=True)}
```

- [ ] **Step 2: Build the revision-abs map and month-type classification at the top of `cmd_score`**

Replace the import block + setup at the start of `cmd_score` (lines 160–170) with:

```python
    import numpy as np
    import polars as pl

    from nfp_ingest.first_print import first_print_changes
    from nfp_vintages.a5 import score
    from nfp_vintages.competitors.consensus import Consensus, load_consensus
    from nfp_vintages.competitors.naive import RandomWalk, TrailingMean
    from nfp_vintages.diagnostics import build_revision_table
    from nfp_vintages.scoreboard import (
        MonthTypeConfig,
        change_draws_k,
        classify_month_types,
        crps_sample,
        interval_coverage,
        venue_for,
    )

    manifest = _read_json(root / "grid_manifest.json")
    prov = manifest["provenance"]
    idx_to_level = float(prov["idx_to_level"])
    fp = first_print_changes()
    fp_hist = fp.select(["ref_date", "first_print_change_k", "vintage_date"])
    consensus = Consensus(load_consensus())  # None until Bloomberg file lands → "—"
    naive_rw, naive_mean = RandomWalk(fp_hist), TrailingMean(fp_hist, window=12)

    # Month-type inputs (skeleton-safe: empty maps degrade to "normal"/"benchmark").
    rev_tbl = build_revision_table()  # [ref_date, first_print_change_k, later_change_k, revision_k]
    rev_months = [r["ref_date"] for r in rev_tbl.iter_rows(named=True)]
    rev_abs = np.array([abs(r["revision_k"]) if r["revision_k"] is not None else np.nan
                        for r in rev_tbl.iter_rows(named=True)], dtype=float)
    mom = _claims_momentum_k()
    claims_arr = np.array([mom.get(m, np.nan) for m in rev_months], dtype=float)
    month_type = classify_month_types(rev_months, rev_abs, claims_arr, MonthTypeConfig())
```

- [ ] **Step 3: Compute calibration + venue + month_type per scored row**

Replace the scoring loop body (lines 184–199) with:

```python
            model = batched[key]["nowcast_change_k"]
            # Predictive draws for calibration (model only) from the persisted npz.
            cov80 = cov90 = crps = None
            npz_path = root / f"{rname}_batched_{key}.npz"
            if npz_path.exists() and actual is not None:
                with np.load(npz_path) as z:
                    if "nowcast_pred_draws" in z:
                        prev_index = float(t["prev_index"])
                        cd = change_draws_k(
                            z["nowcast_pred_draws"],
                            prev_index=prev_index, idx_to_level=idx_to_level,
                        )
                        cov80 = interval_coverage(cd, actual, 0.80)
                        cov90 = interval_coverage(cd, actual, 0.90)
                        crps = crps_sample(cd, actual)
            providers_present = bool(t.get("n_providers", 0))
            mtype = month_type.get(ref, "normal")
            preds = {
                "model": model,
                "consensus": consensus.predict(ref, as_of=as_of),
                "naive_rw": naive_rw.predict(ref, as_of=as_of),
                "naive_mean": naive_mean.predict(ref, as_of=as_of),
            }
            for comp, pred in preds.items():
                rows.append({
                    "regime": rname,
                    "ref_month": ref,
                    "month_type": mtype,
                    "venue": venue_for(providers_present=providers_present),
                    "shutdown_flag": ref in SHUTDOWN_FLAGGED,
                    "competitor": comp,
                    "pred_change_k": pred,
                    "actual_first_print_k": actual,
                    "error_k": None if pred is None else actual - pred,
                    # calibration only meaningful for the model row
                    "coverage_80": cov80 if comp == "model" else None,
                    "coverage_90": cov90 if comp == "model" else None,
                    "crps_k": crps if comp == "model" else None,
                })
```

> NOTE on `prev_index` / `n_providers`: `cmd_snapshot` already records `prev_index` per target (run_a5_backtest.py ~line 100). If `n_providers` is not yet in the manifest, add it in `cmd_snapshot` where the snapshot meta is read: `target_entry["n_providers"] = len(meta.get("provider_names", []))`. Locally this is 0 or 1 (no ADP) → `public-only`.

- [ ] **Step 4: Decompose the report by month type and add a calibration block**

Replace the report-rendering loop (lines 211–237) with:

```python
    df = pl.DataFrame(rows)
    scored = df.filter(
        pl.col("error_k").is_not_null()
        & ~pl.col("ref_month").dt.year().is_in([2020, 2021])
        & ~pl.col("shutdown_flag")
    )
    df.write_parquet(root / "a5_results.parquet")

    venues = sorted({v for v in df["venue"].unique() if v is not None})
    lines = ["# A5 backtest report", "",
             "Model vs competitors on the CES **first print**, at T−7 and T−1, "
             "decomposed by month type.",
             "Consensus is T−1-only and renders `—` until the Bloomberg file lands.",
             f"Venue(s) in this run: **{', '.join(venues) or 'public-only'}** — a "
             "`public-only` run scores a providerless skeleton (spec section 10).",
             "COVID (2020–2021) and shutdown-flagged months excluded from metrics.", ""]
    order = ["normal", "large_revision", "turning_point", "benchmark_window"]
    for rname in REGIMES:
        lines += [f"## Regime {rname}", ""]
        for mtype in order:
            sub = scored.filter((pl.col("regime") == rname) & (pl.col("month_type") == mtype))
            n_months = sub.select(pl.col("ref_month").n_unique()).item()
            lines += [f"### {mtype} ({n_months} months)", "",
                      "| competitor | n | ME | MAE | RMSE |", "|---|---|---|---|---|"]
            for comp in ["model", "consensus", "naive_rw", "naive_mean"]:
                e = sub.filter(pl.col("competitor") == comp)["error_k"].to_numpy()
                m = score(e)
                if m["n"] == 0:
                    lines.append(f"| {comp} | 0 | — | — | — |")
                else:
                    lines.append(
                        f"| {comp} | {m['n']} | {m['me']:+,.0f}k | {m['mae']:,.0f}k "
                        f"| {m['rmse']:,.0f}k |")
            # Model calibration row for this bucket.
            mc = sub.filter(pl.col("competitor") == "model")
            cov80 = mc["coverage_80"].drop_nulls().mean()
            cov90 = mc["coverage_90"].drop_nulls().mean()
            crps = mc["crps_k"].drop_nulls().mean()
            if cov80 is not None:
                lines += ["",
                          f"model calibration — 80% coverage: {cov80:.0%}, "
                          f"90% coverage: {cov90:.0%}, mean CRPS: {crps:,.0f}k", ""]
            else:
                lines.append("")
    (root / "a5_report.md").write_text("\n".join(lines) + "\n")
    print((root / "a5_report.md").read_text())
    return 0
```

- [ ] **Step 5: Smoke-run against an existing backtest dir (no new MCMC), verify the report renders**

If a populated `data/backtests/a5` exists locally:
Run: `uv run python scripts/run_a5_backtest.py score data/backtests/a5`
Expected: a Markdown report with four month-type subsections per regime and a `model calibration` line; `consensus` rows render `—` at T−7 and where the Bloomberg file is absent. If no backtest dir exists, instead run the lint + targeted import check:
Run: `uv run python -c "import ast; ast.parse(open('scripts/run_a5_backtest.py').read())"` and `uv run ruff check scripts/run_a5_backtest.py`
Expected: no syntax/lint errors.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_a5_backtest.py
git commit -m "feat(eval): Tier 0 — regime-decomposed, calibration-aware scoreboard with venue tag"
```

## Task 5: Second QCEW-scored scoreboard

Add a second scoreboard scoring the model (and ADP, when present) against the **QCEW-settled** change — the fair target for QCEW-anchored competitors (spec §1/§3; `specs/model_research.md` §3 target map). Locally the model-vs-QCEW comparison runs for settled months; the ADP column self-renders `—` (ADP is Bloomberg-only). The truth extractor lives in `diagnostics.py` (store access) and is reused by Tier 1.

**Files:**
- Modify: `packages/nfp-vintages/src/nfp_vintages/diagnostics.py` (created in Task 6; this task adds `qcew_settled_changes` — implement after Task 6, or stub the module first)
- Modify: `scripts/run_a5_backtest.py` (`cmd_score`: append a second report section)
- Test: `packages/nfp-vintages/tests/test_diagnostics.py`

**Depends on:** Tasks 6–7 (`diagnostics.py` must exist) and Task 4 (the `cmd_score` body this appends to). It is listed under Phase A because it is a Tier 0 deliverable, but executes after the Tier 1 OLS/revision-table groundwork — see the execution order above.

- [ ] **Step 1: Write the failing test for the QCEW-settled extractor**

```python
# append to packages/nfp-vintages/tests/test_diagnostics.py
import pytest

from nfp_vintages.diagnostics import qcew_settled_changes


@pytest.mark.real_store
def test_qcew_settled_changes_shape():
    df = qcew_settled_changes()
    assert {"ref_date", "qcew_settled_change_k"} <= set(df.columns)
    assert df.height > 0
    # change values are in a sane band (thousands per month)
    vals = df["qcew_settled_change_k"].drop_nulls().to_numpy()
    assert (abs(vals) < 5000).all()
```

- [ ] **Step 2: Run to verify it fails (or skips without store)**

Run: `NFP_STORE_URI=$NFP_STORE_URI uv run pytest packages/nfp-vintages/tests/test_diagnostics.py::test_qcew_settled_changes_shape -q --no-cov -m real_store`
Expected: FAIL (`ImportError`/`AttributeError`) — or SKIP if no store env. (Self-skip is acceptable evidence the marker wiring works.)

- [ ] **Step 3: Implement `qcew_settled_changes` in `diagnostics.py`**

```python
# add to packages/nfp-vintages/src/nfp_vintages/diagnostics.py
def qcew_settled_changes(store_path=None) -> "pl.DataFrame":
    """Latest-vintage QCEW national total over-the-month change (thousands).

    The 'truth' target for QCEW-anchored competitors. Selects max(vintage_date)
    per ref month, level-differences, and converts to thousands.
    """
    import polars as pl
    from nfp_ingest.vintage_store import read_vintage_store
    from nfp_lookups.paths import VINTAGE_STORE_PATH

    store_path = store_path or VINTAGE_STORE_PATH
    lf = read_vintage_store(
        store_path, source="qcew", seasonally_adjusted=True,
        geographic_type="national", geographic_code="00",
        industry_type="total", industry_code="00",
    )
    df = (
        lf.collect()
        .sort(["ref_date", "vintage_date"])
        .group_by("ref_date").agg(pl.col("employment").last().alias("level"))
        .sort("ref_date")
        .with_columns(((pl.col("level") - pl.col("level").shift(1))).alias("qcew_settled_change_k"))
        .select(["ref_date", "qcew_settled_change_k"])
        .drop_nulls("qcew_settled_change_k")
    )
    return df
```

> If the store's QCEW employment column is already in thousands, the diff is already in `k`; if in persons, multiply by `1/1000`. Verify against one known month in Step 4 and adjust the unit factor with a comment.

- [ ] **Step 4: Verify units against a known month, then append the second report section in `cmd_score`**

After writing `a5_report.md` in Task 4 Step 4 (before `return 0`), append:

```python
    # ---- Second scoreboard: model & ADP vs QCEW-settled truth ----
    from nfp_vintages.diagnostics import qcew_settled_changes
    try:
        qcew = {r["ref_date"]: r["qcew_settled_change_k"]
                for r in qcew_settled_changes().iter_rows(named=True)}
    except Exception as exc:  # store unavailable locally
        qcew = {}
        print(f"[qcew scoreboard] skipped: {exc}")
    if qcew:
        qlines = ["", "## Truth scoreboard (vs QCEW-settled change)", "",
                  "Fair target for QCEW-anchored competitors (model, ADP). "
                  "ADP renders `—` until Bloomberg data lands.",
                  "| regime | competitor | n | ME | MAE | RMSE |",
                  "|---|---|---|---|---|---|"]
        model_rows = df.filter(pl.col("competitor") == "model")
        for rname in REGIMES:
            sub = model_rows.filter(pl.col("regime") == rname)
            errs = []
            for r in sub.iter_rows(named=True):
                truth = qcew.get(r["ref_month"])
                if truth is not None and r["pred_change_k"] is not None:
                    errs.append(truth - r["pred_change_k"])
            mm = score(np.array(errs, dtype=float))
            cell = (f"| {rname} | model | {mm['n']} | {mm['me']:+,.0f}k "
                    f"| {mm['mae']:,.0f}k | {mm['rmse']:,.0f}k |") if mm["n"] else \
                   f"| {rname} | model | 0 | — | — | — |"
            qlines.append(cell)
            qlines.append(f"| {rname} | adp | 0 | — | — | — |")  # Bloomberg-only
        with (root / "a5_report.md").open("a") as fh:
            fh.write("\n".join(qlines) + "\n")
```

- [ ] **Step 5: Run the diagnostics test (with store) and the lint**

Run: `uv run pytest packages/nfp-vintages/tests/test_diagnostics.py -q --no-cov` (real_store test self-skips without env) and `uv run ruff check scripts/run_a5_backtest.py packages/nfp-vintages/src/nfp_vintages/diagnostics.py`
Expected: PASS/SKIP; no lint errors.

- [ ] **Step 6: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/diagnostics.py packages/nfp-vintages/tests/test_diagnostics.py scripts/run_a5_backtest.py
git commit -m "feat(eval): Tier 0 — second QCEW-settled truth scoreboard"
```

---

# Phase B — Tier 1: diagnostics gate

## Task 6: OLS helper

Hand-rolled OLS (no statsmodels): returns coefficients, R², residuals, and the coefficient covariance (for the MZ Wald test). This is the foundation for Tasks 9–10.

**Files:**
- Create: `packages/nfp-vintages/src/nfp_vintages/diagnostics.py`
- Test: `packages/nfp-vintages/tests/test_diagnostics.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/nfp-vintages/tests/test_diagnostics.py
import numpy as np
from pytest import approx

from nfp_vintages.diagnostics import OLSResult, ols


def test_ols_recovers_known_line():
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(size=n)
    y = 2.0 + 3.0 * x + rng.normal(scale=1e-6, size=n)
    X = np.column_stack([np.ones(n), x])
    res = ols(X, y)
    assert isinstance(res, OLSResult)
    assert res.coeffs[0] == approx(2.0, abs=1e-3)
    assert res.coeffs[1] == approx(3.0, abs=1e-3)
    assert res.r2 == approx(1.0, abs=1e-6)
    assert res.cov.shape == (2, 2)


def test_ols_r2_zero_for_constant_target_no_slope():
    X = np.column_stack([np.ones(10), np.arange(10.0)])
    y = np.full(10, 5.0)
    res = ols(X, y)
    assert res.coeffs[1] == approx(0.0, abs=1e-9)
    assert res.r2 == approx(0.0, abs=1e-9)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/nfp-vintages/tests/test_diagnostics.py -q --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'nfp_vintages.diagnostics'`.

- [ ] **Step 3: Write minimal implementation**

```python
# packages/nfp-vintages/src/nfp_vintages/diagnostics.py
"""Tier 1 diagnostics — Aruoba revision regression, Mincer-Zarnowitz, gate.

Evaluation-side only; imports no nfp-model code. OLS is hand-rolled on numpy
(no statsmodels in the workspace). See specs/model_improvements.md section 4 and
plans/13-tier01-scoreboard-and-diagnostics.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OLSResult:
    coeffs: np.ndarray      # (k,)
    cov: np.ndarray         # (k, k) coefficient covariance
    r2: float
    n: int
    residuals: np.ndarray   # (n,)


def ols(X: np.ndarray, y: np.ndarray) -> OLSResult:
    """Ordinary least squares with classical (homoskedastic) covariance."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 0.0 if tss == 0.0 else 1.0 - rss / tss
    dof = max(n - k, 1)
    sigma2 = rss / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    return OLSResult(coeffs=beta, cov=sigma2 * xtx_inv, r2=r2, n=n, residuals=resid)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest packages/nfp-vintages/tests/test_diagnostics.py -q --no-cov`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/diagnostics.py packages/nfp-vintages/tests/test_diagnostics.py
git commit -m "feat(eval): OLS helper for Tier 1 diagnostics"
```

## Task 7: Revision table (first-print vs later-vintage)

Build the Aruoba LHS: per ref month, `revision_k = later_change − first_print_change`. `first_print_changes()` supplies the first print; the later/settled change comes from the latest CES vintage. Used by Task 1's classifier and Task 9.

**Files:**
- Modify: `packages/nfp-vintages/src/nfp_vintages/diagnostics.py`
- Test: `packages/nfp-vintages/tests/test_diagnostics.py`

- [ ] **Step 1: Write the failing test (real_store + a pure-merge unit test)**

```python
# append to packages/nfp-vintages/tests/test_diagnostics.py
from datetime import date as _d

import polars as pl
import pytest

from nfp_vintages.diagnostics import _join_revision, build_revision_table


def test_join_revision_pure():
    fp = pl.DataFrame({"ref_date": [_d(2023, 1, 1), _d(2023, 2, 1)],
                       "first_print_change_k": [100.0, 150.0]})
    later = pl.DataFrame({"ref_date": [_d(2023, 1, 1), _d(2023, 2, 1)],
                          "later_change_k": [120.0, 140.0]})
    out = _join_revision(fp, later)
    by = {r["ref_date"]: r for r in out.iter_rows(named=True)}
    assert by[_d(2023, 1, 1)]["revision_k"] == pytest.approx(20.0)   # 120 - 100
    assert by[_d(2023, 2, 1)]["revision_k"] == pytest.approx(-10.0)  # 140 - 150


@pytest.mark.real_store
def test_build_revision_table_real():
    tbl = build_revision_table()
    assert {"ref_date", "first_print_change_k", "later_change_k", "revision_k"} <= set(tbl.columns)
    assert tbl.height > 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/nfp-vintages/tests/test_diagnostics.py -k revision -q --no-cov`
Expected: FAIL — `ImportError: cannot import name '_join_revision'`.

- [ ] **Step 3: Implement the later-change extractor + join**

```python
# add to packages/nfp-vintages/src/nfp_vintages/diagnostics.py
def _latest_ces_changes(store_path=None) -> "pl.DataFrame":
    """Latest-vintage CES SA national total over-the-month change (thousands)."""
    import polars as pl
    from nfp_ingest.vintage_store import read_vintage_store
    from nfp_lookups.paths import VINTAGE_STORE_PATH

    store_path = store_path or VINTAGE_STORE_PATH
    lf = read_vintage_store(
        store_path, source="ces", seasonally_adjusted=True,
        geographic_type="national", geographic_code="00",
        industry_type="total", industry_code="00",
    )
    return (
        lf.collect()
        .sort(["ref_date", "vintage_date"])
        .group_by("ref_date").agg(pl.col("employment").last().alias("level"))
        .sort("ref_date")
        .with_columns((pl.col("level") - pl.col("level").shift(1)).alias("later_change_k"))
        .select(["ref_date", "later_change_k"])
        .drop_nulls("later_change_k")
    )


def _join_revision(fp: "pl.DataFrame", later: "pl.DataFrame") -> "pl.DataFrame":
    return (
        fp.join(later, on="ref_date", how="inner")
        .with_columns((pl.col("later_change_k") - pl.col("first_print_change_k")).alias("revision_k"))
        .sort("ref_date")
    )


def build_revision_table(store_path=None) -> "pl.DataFrame":
    """[ref_date, first_print_change_k, later_change_k, revision_k] over the store."""
    import polars as pl  # noqa: F401  (re-exported types for callers)
    from nfp_ingest.first_print import first_print_changes

    fp = first_print_changes(**({"store_path": store_path} if store_path else {}))
    fp = fp.select(["ref_date", "first_print_change_k"])
    later = _latest_ces_changes(store_path)
    return _join_revision(fp, later)
```

Add `import polars as pl` at module top so the `_join_revision` body resolves `pl`.

- [ ] **Step 4: Run to verify the pure test passes (real_store self-skips without env)**

Run: `uv run pytest packages/nfp-vintages/tests/test_diagnostics.py -k revision -q --no-cov`
Expected: PASS for `test_join_revision_pure`; SKIP for `test_build_revision_table_real` without store env. With store env: `NFP_STORE_URI=… uv run pytest … -m real_store` → PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/diagnostics.py packages/nfp-vintages/tests/test_diagnostics.py
git commit -m "feat(eval): revision table (first-print vs latest-vintage) for Tier 1"
```

## Task 8: Aruoba design matrix

Assemble the as-of-censored regressor matrix `X_t` at first-print time from the **public** regressor set (claims momentum, JOLTS level, lagged revision, cyclical state). ADP/NFCI/biz_apps are full-regime additions, included only when their series are present — `available_regressors` records which were used so the report can tag skeleton vs full.

**Files:**
- Modify: `packages/nfp-vintages/src/nfp_vintages/diagnostics.py`
- Test: `packages/nfp-vintages/tests/test_diagnostics.py`

- [ ] **Step 1: Write the failing test**

```python
# append to packages/nfp-vintages/tests/test_diagnostics.py
from datetime import date

from nfp_vintages.diagnostics import build_aruoba_design


def test_build_aruoba_design_skeleton():
    ref = [date(2023, m, 1) for m in range(1, 7)]
    regressors = {
        "claims_mom": {m: float(i) for i, m in enumerate(ref)},
        "jolts": {m: 9_000.0 + i for i, m in enumerate(ref)},
        "lagged_revision": {m: float(-i) for i, m in enumerate(ref)},
    }
    X, names, used = build_aruoba_design(ref, regressors)
    assert X.shape == (6, len(names))
    assert names[0] == "const"
    assert set(used) == {"claims_mom", "jolts", "lagged_revision"}
    assert np.allclose(X[:, 0], 1.0)  # intercept column


def test_build_aruoba_design_drops_all_nan_regressor():
    ref = [date(2023, m, 1) for m in range(1, 4)]
    regressors = {
        "claims_mom": {m: 1.0 for m in ref},
        "nfci": {m: float("nan") for m in ref},  # absent locally → dropped
    }
    X, names, used = build_aruoba_design(ref, regressors)
    assert "nfci" not in names
    assert "nfci" not in used
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/nfp-vintages/tests/test_diagnostics.py -k aruoba_design -q --no-cov`
Expected: FAIL — `ImportError: cannot import name 'build_aruoba_design'`.

- [ ] **Step 3: Implement**

```python
# add to packages/nfp-vintages/src/nfp_vintages/diagnostics.py
def build_aruoba_design(
    ref_months: list, regressors: dict[str, dict]
) -> tuple[np.ndarray, list[str], list[str]]:
    """Design matrix X_t with an intercept, plus a name list and the used set.

    ``regressors`` maps name -> {ref_month: value}. A regressor that is entirely
    NaN/missing across ``ref_months`` is dropped (records the skeleton vs full
    regime). Rows with any NaN in a *kept* column are the caller's responsibility
    to drop (see aruoba_regression).
    """
    cols, names, used = [np.ones(len(ref_months))], ["const"], []
    for name, series in regressors.items():
        vals = np.array([series.get(m, np.nan) for m in ref_months], dtype=float)
        if np.all(~np.isfinite(vals)):
            continue
        cols.append(vals)
        names.append(name)
        used.append(name)
    return np.column_stack(cols), names, used
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest packages/nfp-vintages/tests/test_diagnostics.py -k aruoba_design -q --no-cov`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/diagnostics.py packages/nfp-vintages/tests/test_diagnostics.py
git commit -m "feat(eval): Aruoba design-matrix builder (skeleton-aware)"
```

## Task 9: Aruoba revision regression (intercept = bias, R² = forecastable share)

Regress `revision_k = α + γ'·X + u`, pooled and by month type. Returns the intercept α (feeds §5A) and R² (gates §6/§7). Operates on aligned LHS + design (built in Tasks 7–8), so it is pure and fully unit-testable.

**Files:**
- Modify: `packages/nfp-vintages/src/nfp_vintages/diagnostics.py`
- Test: `packages/nfp-vintages/tests/test_diagnostics.py`

- [ ] **Step 1: Write the failing test**

```python
# append to packages/nfp-vintages/tests/test_diagnostics.py
from nfp_vintages.diagnostics import AruobaResult, aruoba_regression


def test_aruoba_recovers_intercept_and_r2():
    rng = np.random.default_rng(1)
    n = 400
    x = rng.normal(size=n)
    # revision = 12 (bias) + 4*x + small noise → R^2 high, intercept ~12.
    rev = 12.0 + 4.0 * x + rng.normal(scale=0.01, size=n)
    X = np.column_stack([np.ones(n), x])
    res = aruoba_regression(rev, X, ["const", "x"])
    assert isinstance(res, AruobaResult)
    assert res.intercept_k == approx(12.0, abs=0.1)
    assert res.r2 > 0.99


def test_aruoba_low_r2_for_pure_noise():
    rng = np.random.default_rng(2)
    n = 300
    rev = rng.normal(scale=20.0, size=n)         # unforecastable
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    res = aruoba_regression(rev, X, ["const", "x"])
    assert res.r2 < 0.1                           # below the gate threshold
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/nfp-vintages/tests/test_diagnostics.py -k aruoba_reg -q --no-cov`
Expected: FAIL — `ImportError: cannot import name 'aruoba_regression'`.

- [ ] **Step 3: Implement**

```python
# add to packages/nfp-vintages/src/nfp_vintages/diagnostics.py
@dataclass(frozen=True)
class AruobaResult:
    intercept_k: float          # alpha — the first-print bias (feeds section 5A)
    r2: float                   # forecastable share of revision variance
    coef_names: list[str]
    coeffs: np.ndarray
    n: int


def aruoba_regression(revision_k: np.ndarray, X: np.ndarray, names: list[str]) -> AruobaResult:
    """Fit revision = alpha + gamma'.X. Drops rows with any non-finite entry."""
    revision_k = np.asarray(revision_k, dtype=float)
    mask = np.isfinite(revision_k) & np.isfinite(X).all(axis=1)
    Xm, ym = X[mask], revision_k[mask]
    res = ols(Xm, ym)
    return AruobaResult(intercept_k=float(res.coeffs[0]), r2=res.r2,
                        coef_names=list(names), coeffs=res.coeffs, n=res.n)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest packages/nfp-vintages/tests/test_diagnostics.py -k aruoba_reg -q --no-cov`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/diagnostics.py packages/nfp-vintages/tests/test_diagnostics.py
git commit -m "feat(eval): Aruoba revision regression (intercept=bias, R2=forecastable share)"
```

## Task 10: Mincer–Zarnowitz efficiency regression

Regress `actual = α + β·forecast`; Wald-test the joint null `α=0, β=1` (χ² with 2 dof, `scipy.stats.chi2`). Applied to both the model nowcast and consensus (Task 12 supplies the paired series).

**Files:**
- Modify: `packages/nfp-vintages/src/nfp_vintages/diagnostics.py`
- Test: `packages/nfp-vintages/tests/test_diagnostics.py`

- [ ] **Step 1: Write the failing test**

```python
# append to packages/nfp-vintages/tests/test_diagnostics.py
from nfp_vintages.diagnostics import MZResult, mincer_zarnowitz


def test_mz_efficient_forecast_not_rejected():
    rng = np.random.default_rng(3)
    forecast = rng.normal(100.0, 50.0, 400)
    actual = forecast + rng.normal(0.0, 1.0, 400)   # efficient: alpha~0, beta~1
    res = mincer_zarnowitz(actual, forecast)
    assert isinstance(res, MZResult)
    assert res.alpha == approx(0.0, abs=2.0)
    assert res.beta == approx(1.0, abs=0.05)
    assert res.joint_p > 0.05                         # null not rejected


def test_mz_biased_forecast_rejected():
    rng = np.random.default_rng(4)
    forecast = rng.normal(100.0, 50.0, 400)
    actual = 30.0 + 0.5 * forecast + rng.normal(0.0, 1.0, 400)  # inefficient
    res = mincer_zarnowitz(actual, forecast)
    assert res.joint_p < 0.01                         # null rejected
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/nfp-vintages/tests/test_diagnostics.py -k mz -q --no-cov`
Expected: FAIL — `ImportError: cannot import name 'mincer_zarnowitz'`.

- [ ] **Step 3: Implement**

```python
# add to packages/nfp-vintages/src/nfp_vintages/diagnostics.py
@dataclass(frozen=True)
class MZResult:
    alpha: float
    beta: float
    joint_stat: float    # Wald chi^2 for (alpha=0, beta=1)
    joint_p: float
    r2: float
    n: int


def mincer_zarnowitz(actual: np.ndarray, forecast: np.ndarray) -> MZResult:
    """Efficiency regression actual = alpha + beta*forecast; test alpha=0, beta=1."""
    from scipy.stats import chi2

    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(forecast)
    a, f = actual[mask], forecast[mask]
    X = np.column_stack([np.ones(a.size), f])
    res = ols(X, a)
    theta = res.coeffs                       # [alpha, beta]
    theta0 = np.array([0.0, 1.0])
    diff = theta - theta0
    wald = float(diff @ np.linalg.inv(res.cov) @ diff)
    p = float(chi2.sf(wald, df=2))
    return MZResult(alpha=float(theta[0]), beta=float(theta[1]),
                    joint_stat=wald, joint_p=p, r2=res.r2, n=res.n)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest packages/nfp-vintages/tests/test_diagnostics.py -k mz -q --no-cov`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/diagnostics.py packages/nfp-vintages/tests/test_diagnostics.py
git commit -m "feat(eval): Mincer-Zarnowitz efficiency regression with joint Wald test"
```

## Task 11: Gate decision

Encode the thresholds that gate Tiers 2/3 (spec §4 lines 50–51, §6 line 76): normal-month R² < ~0.1 → diagonal adequate, do **not** fund the first-release-vintage rebuild (§7); turning-point R² materially above normal-month R² → fund §6. Pure function over Aruoba results by month type.

**Files:**
- Modify: `packages/nfp-vintages/src/nfp_vintages/diagnostics.py`
- Test: `packages/nfp-vintages/tests/test_diagnostics.py`

- [ ] **Step 1: Write the failing test**

```python
# append to packages/nfp-vintages/tests/test_diagnostics.py
from nfp_vintages.diagnostics import GateConfig, gate_decision


def test_gate_diagonal_adequate_when_normal_r2_low():
    r2_by_type = {"normal": 0.04, "turning_point": 0.05, "benchmark_window": 0.06}
    g = gate_decision(r2_by_type, GateConfig())
    assert g["fund_first_release_rebuild"] is False
    assert g["fund_tier3_bd"] is False


def test_gate_funds_bd_when_turning_point_r2_concentrated():
    r2_by_type = {"normal": 0.05, "turning_point": 0.40, "benchmark_window": 0.08}
    g = gate_decision(r2_by_type, GateConfig())
    assert g["fund_tier3_bd"] is True
    assert "turning_point" in g["rationale"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/nfp-vintages/tests/test_diagnostics.py -k gate -q --no-cov`
Expected: FAIL — `ImportError: cannot import name 'gate_decision'`.

- [ ] **Step 3: Implement**

```python
# add to packages/nfp-vintages/src/nfp_vintages/diagnostics.py
@dataclass(frozen=True)
class GateConfig:
    normal_r2_floor: float = 0.10           # below this, monthly revisions ~ noise
    turning_point_excess: float = 0.15      # tp R2 must exceed normal by this to fund BD


def gate_decision(r2_by_month_type: dict[str, float], cfg: GateConfig) -> dict:
    """Translate Aruoba R^2-by-month-type into Tier 2/3 funding decisions."""
    normal = r2_by_month_type.get("normal", 0.0)
    tp = r2_by_month_type.get("turning_point", 0.0)
    bench = r2_by_month_type.get("benchmark_window", 0.0)
    fund_rebuild = normal >= cfg.normal_r2_floor
    fund_bd = (tp - normal) >= cfg.turning_point_excess
    rationale = (
        f"normal R²={normal:.3f} ({'>=' if fund_rebuild else '<'} {cfg.normal_r2_floor}); "
        f"turning_point R²={tp:.3f}, benchmark R²={bench:.3f}; "
        f"BD funded={fund_bd} (turning_point excess {tp - normal:+.3f} "
        f"vs {cfg.turning_point_excess})."
    )
    return {
        "fund_first_release_rebuild": bool(fund_rebuild),
        "fund_tier3_bd": bool(fund_bd),
        "rationale": rationale,
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest packages/nfp-vintages/tests/test_diagnostics.py -k gate -q --no-cov`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/diagnostics.py packages/nfp-vintages/tests/test_diagnostics.py
git commit -m "feat(eval): Tier 1 gate-decision encoding the section 4/6 thresholds"
```

## Task 12: Tier 1 CLI + provider-ablation harness

A thin script that assembles the diagnostic inputs and writes the report. Aruoba + MZ run locally on the skeleton regressor set; the provider-ablation block runs only where providers exist (it diffs an existing providered backtest dir against a no-provider rerun) and **self-skips** with a logged note when providerless — it is forward-looking to the Bloomberg regime (spec §4 line 52).

**Depends on:** Task 1 (`classify_month_types`) and Tasks 6–11 (all of `diagnostics.py`).

**Files:**
- Create: `scripts/run_tier1_diagnostics.py`

- [ ] **Step 1: Write the script**

```python
# scripts/run_tier1_diagnostics.py
"""Tier 1 diagnostics: Aruoba revision regression + Mincer-Zarnowitz + gate.

Usage:
    uv run python scripts/run_tier1_diagnostics.py data/backtests/a5

Reads the A5 results parquet (for the model's MZ) and the vintage store (for the
Aruoba LHS + design), writes tier1_diagnostics.md / .parquet, prints the gate.
Provider-ablation is forward-looking (Bloomberg-only) and self-skips locally.
"""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    from nfp_ingest.indicators import read_indicator
    from nfp_vintages.diagnostics import (
        GateConfig,
        aruoba_regression,
        build_aruoba_design,
        build_revision_table,
        gate_decision,
        mincer_zarnowitz,
    )

    rev_tbl = build_revision_table()
    ref = [r["ref_date"] for r in rev_tbl.iter_rows(named=True)]
    rev = np.array([r["revision_k"] for r in rev_tbl.iter_rows(named=True)], dtype=float)

    # ---- regressors (skeleton: claims momentum, JOLTS, lagged revision) ----
    def _monthly(name: str) -> dict:
        df = read_indicator(name)
        if df is None or df.is_empty():
            return {}
        m = (df.with_columns(pl.col("ref_date").dt.truncate("1mo").alias("m"))
             .group_by("m").agg(pl.col("value").mean().alias("v")).sort("m"))
        return {r["m"]: r["v"] for r in m.iter_rows(named=True)}

    claims = read_indicator("claims")
    claims_mom = {}
    if claims is not None and not claims.is_empty():
        mm = (claims.with_columns(pl.col("ref_date").dt.truncate("1mo").alias("m"))
              .group_by("m").agg(pl.col("value").mean().alias("v")).sort("m")
              .with_columns((pl.col("v") - pl.col("v").shift(3)).alias("mom3")))
        claims_mom = {r["m"]: r["mom3"] for r in mm.iter_rows(named=True)}
    lagged_rev = {ref[i]: rev[i - 1] for i in range(1, len(ref))}
    regressors = {"claims_mom": claims_mom, "jolts": _monthly("jolts"),
                  "lagged_revision": lagged_rev}

    X, names, used = build_aruoba_design(ref, regressors)
    pooled = aruoba_regression(rev, X, names)
    skeleton = sorted(set(regressors) - set(used))

    # ---- per-month-type Aruoba (reuse the Tier 0 classifier) ----
    from nfp_vintages.scoreboard import MonthTypeConfig, classify_month_types

    claims_arr = np.array([claims_mom.get(m, np.nan) / 1000.0 for m in ref], dtype=float)
    mtypes = classify_month_types(ref, np.abs(rev), claims_arr, MonthTypeConfig())
    r2_by_type: dict[str, float] = {}
    for mt in ["normal", "large_revision", "turning_point", "benchmark_window"]:
        idx = [i for i, m in enumerate(ref) if mtypes[m] == mt]
        if len(idx) > X.shape[1] + 2:  # enough dof
            r2_by_type[mt] = aruoba_regression(rev[idx], X[idx], names).r2

    # ---- Mincer-Zarnowitz on the model nowcast ----
    mz_lines = []
    results_path = root / "a5_results.parquet"
    if results_path.exists():
        df = pl.read_parquet(results_path)
        mrows = df.filter((pl.col("competitor") == "model")
                          & pl.col("error_k").is_not_null())
        if mrows.height > 5:
            actual = mrows["actual_first_print_k"].to_numpy()
            pred = mrows["pred_change_k"].to_numpy()
            mz = mincer_zarnowitz(actual, pred)
            mz_lines = [f"- model: alpha={mz.alpha:+.1f}k, beta={mz.beta:.3f}, "
                        f"joint p(alpha=0,beta=1)={mz.joint_p:.3f}, n={mz.n}"]
        # consensus MZ only if consensus predictions are present
        crows = df.filter((pl.col("competitor") == "consensus")
                          & pl.col("error_k").is_not_null())
        if crows.height > 5:
            mzc = mincer_zarnowitz(crows["actual_first_print_k"].to_numpy(),
                                   crows["pred_change_k"].to_numpy())
            mz_lines.append(f"- consensus: alpha={mzc.alpha:+.1f}k, beta={mzc.beta:.3f}, "
                            f"joint p={mzc.joint_p:.3f}, n={mzc.n}")
        else:
            mz_lines.append("- consensus: — (no consensus predictions present; "
                            "Bloomberg file not landed)")

    gate = gate_decision(r2_by_type, GateConfig())

    # ---- provider-ablation (forward-looking; self-skip locally) ----
    ablation_note = ("- provider-ablation: skipped (public-only venue; no provider "
                     "data). Forward-looking to the Bloomberg regime — spec section 4.")

    lines = ["# Tier 1 diagnostics", "",
             f"Venue: **{'full' if not skeleton else 'public-only (skeleton)'}**. "
             f"Regressors used: {used}. Missing (full-regime only): {skeleton}.", "",
             "## Aruoba revision regression (revision = alpha + gamma'.X)",
             f"- pooled: intercept (bias) = {pooled.intercept_k:+.1f}k, "
             f"R² (forecastable share) = {pooled.r2:.3f}, n={pooled.n}",
             "- R² by month type: " + ", ".join(f"{k}={v:.3f}" for k, v in r2_by_type.items()),
             "", "## Mincer-Zarnowitz efficiency", *mz_lines,
             "", "## Provider ablation", ablation_note,
             "", "## Gate decision", f"- {gate['rationale']}",
             f"- fund first-release-vintage rebuild (section 7): {gate['fund_first_release_rebuild']}",
             f"- fund turning-point birth/death (section 6): {gate['fund_tier3_bd']}", ""]
    (root / "tier1_diagnostics.md").write_text("\n".join(lines) + "\n")

    pl.DataFrame([{"month_type": k, "aruoba_r2": v} for k, v in r2_by_type.items()] +
                 [{"month_type": "pooled", "aruoba_r2": pooled.r2}]).write_parquet(
        root / "tier1_diagnostics.parquet")
    print((root / "tier1_diagnostics.md").read_text())
    print(f"[gate] {gate['rationale']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Lint + import-parse check**

Run: `uv run ruff check scripts/run_tier1_diagnostics.py` and `uv run python -c "import ast; ast.parse(open('scripts/run_tier1_diagnostics.py').read())"`
Expected: no errors.

- [ ] **Step 3: Real-store smoke (requires store env; otherwise document as pending)**

Run: `NFP_STORE_URI=$NFP_STORE_URI uv run python scripts/run_tier1_diagnostics.py data/backtests/a5`
Expected: prints a Tier 1 report with a pooled Aruoba intercept + R², an R²-by-month-type line, the model MZ row (consensus `—`), provider-ablation skipped, and a gate decision. If no store/backtest dir is available locally, record this smoke as a follow-up to run in the full regime (note it in the commit body).

- [ ] **Step 4: Commit**

```bash
git add scripts/run_tier1_diagnostics.py
git commit -m "feat(eval): Tier 1 diagnostics CLI (Aruoba + MZ + gate; ablation forward-looking)"
```

---

# Phase C — verification & handoff

## Task 13: Full suite, lint, spec-coverage self-review, gate-log update

**Files:**
- Modify: `plans/0-port_and_staged_plan.md` (gate log)

- [ ] **Step 1: Run the fast suite and lint**

Run: `uv run pytest packages/nfp-vintages -m "not network and not slow" --no-cov`
Expected: all new `test_scoreboard.py` + `test_diagnostics.py` tests PASS; `real_store` tests SKIP without store env.
Run: `uv run ruff check .`
Expected: clean.

- [ ] **Step 2: Run the store-backed tests (if store env available)**

Run: `NFP_STORE_URI=$NFP_STORE_URI uv run pytest packages/nfp-vintages -m real_store --no-cov -v`
Expected: `test_build_revision_table_real` and `test_qcew_settled_changes_shape` PASS.

- [ ] **Step 3: Spec-coverage self-review (checklist, fix inline)**

Confirm each §3/§4 deliverable maps to a task:
- §3 month-type decomposition → Tasks 1, 4. §3 calibration (coverage, CRPS) → Tasks 2, 4. §3 venue tag → Tasks 3, 4. §3 second QCEW scoreboard → Task 5. §3 COVID/shutdown handling → Task 4 Step 4.
- §4 Aruoba (intercept + R²) → Tasks 7–9, 12. §4 Mincer–Zarnowitz (model + consensus) → Tasks 10, 12. §4 provider-ablation (forward-looking) → Task 12. §4 gate thresholds → Tasks 11, 12.
- Firewall: grep the diff for forbidden paths — `git diff --name-only main... | grep -E 'nfp-model|transform_to_panel|build_model_data'` must be **empty**. Fix any leak before proceeding.

- [ ] **Step 4: Update the gate log in `plans/0-port_and_staged_plan.md`**

Add a line under the staged-plan gate log recording: "Tier 0 + Tier 1 (plan 13) implemented — regime-decomposed/calibration scoreboard + Aruoba/MZ diagnostics + gate; no nfp-model change; Aruoba R² and gate decision pending a full store run." Keep wording consistent with the existing gate-log entries.

- [ ] **Step 5: Commit**

```bash
git add plans/0-port_and_staged_plan.md
git commit -m "docs(plans): record Tier 0 + Tier 1 completion in the gate log"
```

---

## Open items carried forward (not blockers)

- **Aruoba R² readout is data-gated.** The intercept that feeds §5A and the R² that gates §6/§7 require a full store run; locally the suite proves the *machinery* (against synthetic priors), not the NFP-specific numbers.
- **Consensus MZ + provider-ablation are forward-looking.** Both render `—`/skip until the Bloomberg consensus file and provider microdata land (`specs/model_improvements.md` §11; `specs/bloomberg_consensus.md`).
- **QCEW change unit factor** (Task 5 Step 4) must be eyeballed against one known month before trusting the truth scoreboard.
- **Next plan (14+)** picks up §5A (the locally-testable first-print offset, which *does* touch `nowcast.py` and therefore starts the model-side sequence behind the §8 parity governance) — only after Task 13's gate readout justifies it.
