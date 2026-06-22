# Consensus-on-both-tracks — Evaluation Upgrade Implementation Plan (plans/18)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the private and Total consensus first-class evaluation benchmarks on *both* tracks,
add the Total-error decomposition (private-side vs government-side), and compute the falsifiable
**combination gate** — all off the model firewall, locally correctness-testable.

**Architecture:** Pure evaluation-layer additions. New stats helpers in `nfp_vintages/diagnostics.py`
(implied-government series; forecast-encompassing + Bates–Granger combination weight; per-cell gate),
a new `ImpliedGovernment` competitor in `competitors/consensus.py`, and wiring into the two harness
entrypoints (`scripts/run_a5_backtest.py`: `cmd_score` for Track A, `cmd_total` for Track B). The
model is untouched — the gate is *computed*, never *fires* locally (providerless skeleton; the firing,
the weights, and any `model_combo` layer are Bloomberg-port decisions).

**Tech Stack:** Python 3.12, Polars, NumPy, SciPy (`scipy.stats.chi2`, already a dep — `diagnostics.py:208`),
pytest, ruff (line 100; E,W,F,I,B,C4,UP). uv workspace.

**Spec:** [specs/model_improvements.md](../model_improvements.md) §12 (the design of record).

---

## Global Constraints

- **Firewall (HARD).** Do **not** modify the *logic* of any model-mastered path: `nfp-model/**`,
  `nfp_ingest/transform_to_panel`, `build_model_data`, `model_data.py`, `first_print.py`,
  `wedge_data.py`, `nfp_vintages/a5.py`, or any A1/A2/A3 golden. This plan **calls** `a5.score()` and
  `first_print_changes()` but never edits them (docstring-only edits would be OK; none are needed). All
  new code lands in `diagnostics.py`, `competitors/consensus.py`, `assembly.py` (additions only), and
  `scripts/run_a5_backtest.py`.
- **Build-here / validate-on-port.** Locally the model is a providerless **skeleton**, so the gate
  **cannot truly fire** — Phase 1 builds and unit-tests the gate *machinery* and confirms it runs; the
  actual firing, the combination weights, the wedge-prior change, and any `model_combo` layer are
  deferred to Bloomberg (spec §12.5/§12.6). **Do not build `model_combo` or any in-model conditioning.**
- **T−1-only consensus.** `Consensus.predict` withholds the value until `as_of ≥ release_date − 1 day`
  (`consensus.py:_LOCK_LAG`), so consensus is naturally `None` at **t7** and present at **t1**. The gate
  runs on **t1 cells only**; t7 is the model's standalone (vs naive floors) regime. Do not relax the lock.
- **No store writes in tests.** Never run a store-writing function against MinIO in a test. New tests use
  **synthetic** frames + `tmp_path`; real-store tests (if any) carry `@pytest.mark.real_store` and self-skip.
- **Commits:** scoped — `git add <exact paths>`, never `git add -A`/`-u`/`.` (the tree may carry unrelated
  WIP). End every commit message with:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- **Branch:** all tasks land on `consensus-both-tracks` (already created; the §12 spec amendment is its
  first commit).
- **Gate after every task:** `uv run pytest -m "not network" --no-cov` green, `uv run ruff check .` clean.

---

## File structure

| File | Change | Responsibility |
|---|---|---|
| `packages/nfp-vintages/src/nfp_vintages/diagnostics.py` | **modify** (add) | `implied_government_consensus`, `EncompassingResult`, `encompassing`, `combination_gate` |
| `packages/nfp-vintages/src/nfp_vintages/competitors/consensus.py` | **modify** (add) | `ImpliedGovernment` competitor (Total − Private, T−1 lock) |
| `scripts/run_a5_backtest.py` | **modify** | `cmd_score`: private-consensus competitor + collect cells + emit gate; `cmd_total`: Total-error decomposition + implied-govt benchmark; fix the stale "consensus absent" comment |
| `packages/nfp-vintages/src/nfp_vintages/tests/test_diagnostics.py` | **modify** (add) | unit tests for the three new diagnostics functions |
| `packages/nfp-vintages/src/nfp_vintages/tests/test_competitors.py` | **modify** (add) | unit tests for `ImpliedGovernment` |
| `.env.example` | **modify** | document `NFP_CONSENSUS_PATH` |

**Task order** (dependencies): T1 → T2 → T3 are independent pure helpers (T3 consumes T2). T4 consumes the
`ImpliedGovernment` of T5's sibling and the gate of T3; T5 consumes T1. Build T1, T2, T3, then T4, T5.

---

### Task 1: Implied-government consensus series

**Files:**
- Modify: `packages/nfp-vintages/src/nfp_vintages/diagnostics.py`
- Test: `packages/nfp-vintages/src/nfp_vintages/tests/test_diagnostics.py`

**Interfaces:**
- Consumes: the loaded consensus frame (cols `ownership, industry_type, industry_code, ref_date, release_date, consensus_mean, consensus_median`; `'00'` Total + `'05'` private rows).
- Produces: `implied_government_consensus(table, *, statistic="median") -> pl.DataFrame` with columns `ref_date, release_date, implied_govt_k` (one row per ref_date present in **both** series).

- [ ] **Step 1: Write the failing test**

In `test_diagnostics.py` (with the other pure tests, near the top so it needs no store):

```python
def test_implied_government_consensus_is_total_minus_private():
    import polars as pl
    from datetime import date
    from nfp_vintages.diagnostics import implied_government_consensus

    tbl = pl.DataFrame({
        "ownership": ["total", "private", "total", "private"],
        "industry_type": ["total"] * 4,
        "industry_code": ["00", "05", "00", "05"],
        "ref_date": [date(2024, 1, 1), date(2024, 1, 1), date(2024, 2, 1), date(2024, 2, 1)],
        "release_date": [date(2024, 2, 2), date(2024, 2, 2), date(2024, 3, 8), date(2024, 3, 8)],
        "consensus_mean": [180.0, 160.0, 200.0, 175.0],
        "consensus_median": [185.0, 165.0, 210.0, 180.0],
    })
    out = implied_government_consensus(tbl)  # median by default
    assert out.columns == ["ref_date", "release_date", "implied_govt_k"]
    assert out.height == 2
    got = dict(zip(out["ref_date"].to_list(), out["implied_govt_k"].to_list()))
    assert got[date(2024, 1, 1)] == 185.0 - 165.0   # 20.0
    assert got[date(2024, 2, 1)] == 210.0 - 180.0   # 30.0
```

- [ ] **Step 2: Run it — expect failure**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_diagnostics.py::test_implied_government_consensus_is_total_minus_private -q`
Expected: FAIL — `ImportError: cannot import name 'implied_government_consensus'`.

- [ ] **Step 3: Implement**

Add to `diagnostics.py` (after the existing revision-table helpers; `import polars as pl` is already present):

```python
def implied_government_consensus(
    table: pl.DataFrame, *, statistic: str = "median"
) -> pl.DataFrame:
    """Implied government consensus = Total (``'00'``) − Private (``'05'``) per ref month.

    The street's monthly expectation for the government contribution to headline NFP,
    derived from the two published consensus series. It is the government wedge's first
    external benchmark (the wedge previously had none). One row per ``ref_date`` present
    in **both** series.

    Parameters
    ----------
    table : pl.DataFrame
        The loaded consensus file (``load_consensus``), carrying ``'00'`` and ``'05'``.
    statistic : {"median", "mean"}, default "median"
        Which survey statistic to difference.

    Returns
    -------
    pl.DataFrame
        Columns ``ref_date``, ``release_date``, ``implied_govt_k`` (thousands), sorted by ref_date.
    """
    if statistic not in ("median", "mean"):
        raise ValueError(f"statistic must be 'median' or 'mean', got {statistic!r}")
    col = f"consensus_{statistic}"
    total = table.filter(pl.col("industry_code") == "00").select(
        "ref_date", "release_date", pl.col(col).alias("_total")
    )
    private = table.filter(pl.col("industry_code") == "05").select(
        "ref_date", pl.col(col).alias("_private")
    )
    return (
        total.join(private, on="ref_date", how="inner")
        .with_columns((pl.col("_total") - pl.col("_private")).alias("implied_govt_k"))
        .select("ref_date", "release_date", "implied_govt_k")
        .sort("ref_date")
    )
```

- [ ] **Step 4: Run it — expect pass**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_diagnostics.py::test_implied_government_consensus_is_total_minus_private -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/diagnostics.py \
        packages/nfp-vintages/src/nfp_vintages/tests/test_diagnostics.py
git commit -m "feat(diagnostics): implied-government consensus (Total − Private) [plans/18 T1]

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Forecast-encompassing regression + Bates–Granger combination weight

**Files:**
- Modify: `packages/nfp-vintages/src/nfp_vintages/diagnostics.py`
- Test: `packages/nfp-vintages/src/nfp_vintages/tests/test_diagnostics.py`

**Interfaces:**
- Consumes: `ols` (`diagnostics.py:26`, returns `OLSResult(coeffs, cov, r2, n, residuals)`); `scipy.stats.chi2`.
- Produces:
  - `EncompassingResult` dataclass: `n:int, w_model:float, p_model_adds_info:float, model_mae:float, consensus_mae:float, combo_mae:float, b:float, c:float`.
  - `encompassing(actual, model, consensus) -> EncompassingResult | None` (None if < 5 finite paired obs).

`w_model` is the Bates–Granger optimal convex weight on the model; `p_model_adds_info` is the Wald
p-value that the model's coefficient in `actual = a + b·model + c·consensus` is zero (small ⇒ the model
adds information beyond consensus, i.e. consensus does **not** encompass it).

- [ ] **Step 1: Write the failing tests**

```python
def test_encompassing_returns_none_below_min_obs():
    import numpy as np
    from nfp_vintages.diagnostics import encompassing
    assert encompassing(np.arange(4.0), np.arange(4.0), np.arange(4.0)) is None


def test_encompassing_model_adds_info_when_consensus_is_noise():
    import numpy as np
    from nfp_vintages.diagnostics import encompassing
    rng = np.random.default_rng(0)
    actual = rng.normal(150, 50, 60)
    model = actual + rng.normal(0, 5, 60)      # model tracks actual tightly
    consensus = rng.normal(150, 50, 60)        # consensus ~ pure noise
    r = encompassing(actual, model, consensus)
    assert r is not None and r.n == 60
    assert r.p_model_adds_info < 0.05          # model clearly adds info
    assert r.w_model > 0.8                     # weight piles onto the model
    assert r.combo_mae <= min(r.model_mae, r.consensus_mae) + 1e-6


def test_encompassing_consensus_encompasses_model_when_model_is_noise():
    import numpy as np
    from nfp_vintages.diagnostics import encompassing
    rng = np.random.default_rng(1)
    actual = rng.normal(150, 50, 60)
    consensus = actual + rng.normal(0, 5, 60)  # consensus tracks actual
    model = rng.normal(150, 50, 60)            # model ~ pure noise
    r = encompassing(actual, model, consensus)
    assert r is not None
    assert r.p_model_adds_info > 0.10          # cannot reject b == 0
    assert r.w_model < 0.2                      # weight piles onto consensus
```

- [ ] **Step 2: Run them — expect failure**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_diagnostics.py -k encompassing -q`
Expected: FAIL — `ImportError: cannot import name 'encompassing'`.

- [ ] **Step 3: Implement**

Add to `diagnostics.py` (`import numpy as np` and `from dataclasses import dataclass` are already present;
keep the `from scipy.stats import chi2` import local to the function, mirroring `mincer_zarnowitz`):

```python
@dataclass(frozen=True)
class EncompassingResult:
    """Forecast-encompassing + optimal-combination readout for one cell.

    ``actual = a + b·model + c·consensus``; ``p_model_adds_info`` is the Wald p that
    ``b == 0`` (small ⇒ consensus does NOT encompass the model). ``w_model`` is the
    Bates–Granger optimal convex weight on the model from the error covariance.
    """
    n: int
    w_model: float
    p_model_adds_info: float
    model_mae: float
    consensus_mae: float
    combo_mae: float
    b: float
    c: float


def encompassing(
    actual: np.ndarray, model: np.ndarray, consensus: np.ndarray
) -> EncompassingResult | None:
    """Encompassing test + Bates–Granger combination weight for paired forecasts.

    Returns ``None`` when fewer than 5 finite ``(actual, model, consensus)`` triples
    exist (e.g. a t7 cell, where consensus has not locked).
    """
    from scipy.stats import chi2

    a_, m_, c_ = (np.asarray(x, dtype=float) for x in (actual, model, consensus))
    ok = np.isfinite(a_) & np.isfinite(m_) & np.isfinite(c_)
    n = int(ok.sum())
    if n < 5:
        return None
    av, mv, cv = a_[ok], m_[ok], c_[ok]

    # Fair–Shiller encompassing regression actual = a + b·model + c·consensus.
    X = np.column_stack([np.ones(n), mv, cv])
    res = ols(X, av)
    b, c = float(res.coeffs[1]), float(res.coeffs[2])
    var_b = float(res.cov[1, 1])
    wald = (b * b) / var_b if var_b > 0 else 0.0
    p_model_adds_info = float(chi2.sf(wald, df=1))

    # Bates–Granger optimal convex weight on the model from error (co)variance.
    em, ec = av - mv, av - cv
    vm, vc = float(np.var(em)), float(np.var(ec))
    cov = float(np.cov(em, ec)[0, 1]) if n > 1 else 0.0
    denom = vm + vc - 2.0 * cov
    w_model = float(np.clip((vc - cov) / denom, 0.0, 1.0)) if denom > 0 else 0.5

    combo = w_model * mv + (1.0 - w_model) * cv
    return EncompassingResult(
        n=n,
        w_model=w_model,
        p_model_adds_info=p_model_adds_info,
        model_mae=float(np.mean(np.abs(em))),
        consensus_mae=float(np.mean(np.abs(ec))),
        combo_mae=float(np.mean(np.abs(av - combo))),
        b=b,
        c=c,
    )
```

- [ ] **Step 4: Run them — expect pass**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_diagnostics.py -k encompassing -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/diagnostics.py \
        packages/nfp-vintages/src/nfp_vintages/tests/test_diagnostics.py
git commit -m "feat(diagnostics): forecast-encompassing + Bates-Granger weight [plans/18 T2]

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Per-cell combination gate (month-type × horizon)

**Files:**
- Modify: `packages/nfp-vintages/src/nfp_vintages/diagnostics.py`
- Test: `packages/nfp-vintages/src/nfp_vintages/tests/test_diagnostics.py`

**Interfaces:**
- Consumes: `encompassing` (T2).
- Produces: `combination_gate(cells, *, alpha=0.10) -> dict[tuple[str, str], dict]`.
  - `cells`: `{(month_type, horizon): {"actual": [...], "model": [...], "consensus": [...]}}`.
  - Each value: `{"result": EncompassingResult, "fires": bool}` or, when `encompassing` returns `None`,
    `{"result": None, "fires": False, "reason": "insufficient_paired_obs", "n": <int>}`.
  - **Fires** iff `p_model_adds_info < alpha` **and** `combo_mae < min(model_mae, consensus_mae)`.
    Named distinctly from the existing `gate_decision` (the §4 model-layer Aruoba gate).

- [ ] **Step 1: Write the failing test**

```python
def test_combination_gate_fires_only_where_model_adds_info_and_combo_wins():
    import numpy as np
    from nfp_vintages.diagnostics import combination_gate
    rng = np.random.default_rng(2)
    # turning_point/t1: model adds info → should fire
    a_tp = rng.normal(0, 80, 60)
    cells = {
        ("turning_point", "t1"): {
            "actual": a_tp.tolist(),
            "model": (a_tp + rng.normal(0, 8, 60)).tolist(),
            "consensus": rng.normal(0, 80, 60).tolist(),
        },
        # normal/t1: consensus encompasses; model is noise → must not fire
        ("normal", "t1"): {
            "actual": (a_n := rng.normal(150, 40, 60)).tolist(),
            "model": rng.normal(150, 40, 60).tolist(),
            "consensus": (a_n + rng.normal(0, 6, 60)).tolist(),
        },
        # turning_point/t7: no consensus → skipped (insufficient obs)
        ("turning_point", "t7"): {"actual": [1.0, 2.0], "model": [1.0, 2.0], "consensus": [float("nan"), float("nan")]},
    }
    out = combination_gate(cells)
    assert out[("turning_point", "t1")]["fires"] is True
    assert out[("normal", "t1")]["fires"] is False
    assert out[("turning_point", "t7")]["fires"] is False
    assert out[("turning_point", "t7")]["reason"] == "insufficient_paired_obs"
```

- [ ] **Step 2: Run it — expect failure**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_diagnostics.py::test_combination_gate_fires_only_where_model_adds_info_and_combo_wins -q`
Expected: FAIL — `ImportError: cannot import name 'combination_gate'`.

- [ ] **Step 3: Implement**

```python
def combination_gate(
    cells: dict[tuple[str, str], dict], *, alpha: float = 0.10
) -> dict[tuple[str, str], dict]:
    """Run the encompassing/combination analysis per (month_type, horizon) cell.

    Distinct from ``gate_decision`` (the §4 Aruoba model-layer gate). This is the §12
    *combination* gate: for each cell with consensus present (t1), decide whether a
    model–consensus blend earns its keep. t7 cells carry no consensus and are skipped.

    A cell **fires** iff the model adds information beyond consensus
    (``p_model_adds_info < alpha``) AND the blend beats both standalone
    (``combo_mae < min(model_mae, consensus_mae)``).
    """
    out: dict[tuple[str, str], dict] = {}
    for key, d in cells.items():
        r = encompassing(
            np.asarray(d["actual"], dtype=float),
            np.asarray(d["model"], dtype=float),
            np.asarray(d["consensus"], dtype=float),
        )
        if r is None:
            out[key] = {"result": None, "fires": False,
                        "reason": "insufficient_paired_obs", "n": len(d["actual"])}
            continue
        fires = (r.p_model_adds_info < alpha) and (r.combo_mae < min(r.model_mae, r.consensus_mae))
        out[key] = {"result": r, "fires": fires}
    return out
```

- [ ] **Step 4: Run it — expect pass**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_diagnostics.py::test_combination_gate_fires_only_where_model_adds_info_and_combo_wins -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/diagnostics.py \
        packages/nfp-vintages/src/nfp_vintages/tests/test_diagnostics.py
git commit -m "feat(diagnostics): per-cell combination gate (month-type × horizon) [plans/18 T3]

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `ImpliedGovernment` competitor

**Files:**
- Modify: `packages/nfp-vintages/src/nfp_vintages/competitors/consensus.py`
- Test: `packages/nfp-vintages/src/nfp_vintages/tests/test_competitors.py`

**Interfaces:**
- Consumes: the loaded consensus frame; the `Competitor` protocol (`predict(ref_month, *, as_of) -> float | None`);
  `implied_government_consensus` (T1) for the per-month value; the same `_LOCK_LAG` censoring as `Consensus`.
- Produces: `ImpliedGovernment(table)` — `name="implied_govt"`, predicts `Total − Private` for `ref_month`,
  withheld until release-eve (so `None` at t7), `None` when unconfigured/absent.

- [ ] **Step 1: Write the failing test**

Mirror the existing consensus fixture in `test_competitors.py` (`_consensus_file(tmp_path)` already writes a
`'00'`+`'05'` parquet). Add:

```python
def test_implied_government_predicts_total_minus_private_at_t1(tmp_path):
    from datetime import date, timedelta
    from nfp_vintages.competitors.consensus import ImpliedGovernment, load_consensus

    tbl = load_consensus(_consensus_file(tmp_path))
    ig = ImpliedGovernment(tbl)
    assert ig.name == "implied_govt"

    # pick a row from the fixture; consensus locks at release_date − 1 day
    row = tbl.filter(tbl["industry_code"] == "00").row(0, named=True)
    ref, rel = row["ref_date"], row["release_date"]
    priv = tbl.filter((tbl["industry_code"] == "05") & (tbl["ref_date"] == ref)).row(0, named=True)
    expected = row["consensus_median"] - priv["consensus_median"]

    assert ig.predict(ref, as_of=rel - timedelta(days=1)) == expected   # locked at release-eve
    assert ig.predict(ref, as_of=rel - timedelta(days=7)) is None       # t7: not yet locked


def test_implied_government_none_when_unconfigured():
    from datetime import date
    from nfp_vintages.competitors.consensus import ImpliedGovernment
    ig = ImpliedGovernment(None)
    assert ig.predict(date(2024, 1, 1), as_of=date(2024, 3, 1)) is None
```

- [ ] **Step 2: Run it — expect failure**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_competitors.py -k implied_government -q`
Expected: FAIL — `ImportError: cannot import name 'ImpliedGovernment'`.

- [ ] **Step 3: Implement**

Add to `consensus.py` (after the `Consensus` class). Reuse the same month-bucketing + lock discipline:

```python
class ImpliedGovernment:
    """T−1-only competitor for the government wedge: Total − Private consensus.

    The street's implied monthly government contribution. Same release-eve lock as
    :class:`Consensus`, so it is ``None`` at t7 and present at t1. ``None`` when no
    table is loaded or the month/series is absent.
    """

    name = "implied_govt"

    def __init__(self, table: "pl.DataFrame | None", *, statistic: str = "median") -> None:
        if statistic not in ("median", "mean"):
            raise ValueError(f"statistic must be 'median' or 'mean', got {statistic!r}")
        if table is None:
            self._table = None
        else:
            from ..diagnostics import implied_government_consensus

            self._table = implied_government_consensus(table, statistic=statistic)

    def predict(self, ref_month: date, *, as_of: date) -> float | None:
        if self._table is None:
            return None
        month = date(ref_month.year, ref_month.month, 1)
        hit = self._table.filter(pl.col("ref_date") == month)
        if hit.height == 0:
            return None
        row = hit.row(0, named=True)
        if as_of < row["release_date"] - _LOCK_LAG:
            return None
        return float(row["implied_govt_k"])
```

*Note:* the `from ..diagnostics import implied_government_consensus` is a local import to avoid a module-level
cycle (`diagnostics` imports nothing from `competitors`, but keep it local for safety).

- [ ] **Step 4: Run it — expect pass**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_competitors.py -k implied_government -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/competitors/consensus.py \
        packages/nfp-vintages/src/nfp_vintages/tests/test_competitors.py
git commit -m "feat(competitors): ImpliedGovernment (Total − Private) competitor [plans/18 T4]

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Wire private consensus + the combination gate into `cmd_score` (Track A)

**Files:**
- Modify: `scripts/run_a5_backtest.py` (the `cmd_score` function)

**Interfaces:**
- Consumes: `Consensus` (`industry_code="05"`), `combination_gate` (T3), the existing scoring loop
  (`ref`, `as_of`, `actual`, `model`, the `month_type` dict, the `REGIMES` keys `t7`/`t1`).
- Produces: a `consensus` competitor row in the scoreboard (present only at t1), a per-cell gate section in
  `a5_report.md`, and the corrected "consensus absent" explanatory text.

> This task edits a CLI script that reads run-manifests, so it is **integration-wired** rather than unit-TDD'd;
> the gate *math* is already unit-tested (T3). Verify by a dry construction test (below) + the existing
> harness smoke. **Do not change `a5.score()` or `first_print_changes` (firewall).**

- [ ] **Step 1: Add the private-consensus competitor**

Near the naive-floor instantiation (`run_a5_backtest.py:~290`), add:

```python
from nfp_vintages.competitors.consensus import Consensus, load_consensus
consensus_priv = Consensus(load_consensus(), industry_code="05")  # private '05'; None at t7 (lock)
```

In the `preds` dict (`~351`), add a `consensus` entry (it returns `None` at t7 automatically):

```python
        "consensus": consensus_priv.predict(ref, as_of=as_of),
```

Add `"consensus"` to the report-table competitor loop (`~412`): `for comp in ["model", "model_5a", "consensus", "naive_rw", "naive_mean"]:` — the report already tolerates `None`/empty error arrays via `a5.score()` (returns `n=0`).

- [ ] **Step 2: Collect gate cells + run the gate**

In the scoring loop, accumulate per-cell triples (only `t1` will have non-`None` consensus):

```python
    gate_cells: dict[tuple[str, str], dict] = {}
    # ... inside the per-target loop, after `mtype` and `preds` are known:
        cons = preds["consensus"]
        if cons is not None:
            cell = gate_cells.setdefault((mtype, rname), {"actual": [], "model": [], "consensus": []})
            cell["actual"].append(actual)
            cell["model"].append(preds["model"])
            cell["consensus"].append(cons)
```

After the loop, compute and render the gate:

```python
    from nfp_vintages.diagnostics import combination_gate
    gate = combination_gate(gate_cells)
    # append a "## Combination gate (t1; build-here/validate-on-port)" section to a5_report.md:
    #   one row per (month_type, horizon): n, w_model, p_model_adds_info, model/consensus/combo MAE, fires
    #   with a banner: "Skeleton run — informational only; the gate fires on the Bloomberg full regime (§12.6)."
```

- [ ] **Step 3: Fix the stale comment**

Replace the `run_a5_backtest.py:~397` explanatory text — currently *"Consensus and ADP are absent by design:
consensus forecasts **Total** …"* — with: consensus (private `'05'`) is now a competitor present at **t1**
(it locks release-eve, so it is absent at t7); **ADP** stays out by design; the **Total** consensus contest
lives in `cmd_total`.

- [ ] **Step 4: Dry construction test + harness smoke**

Add a lightweight test that the gate-cell shape feeds `combination_gate` (no manifests needed):

```python
# packages/nfp-vintages/src/nfp_vintages/tests/test_diagnostics.py
def test_combination_gate_accepts_harness_cell_shape():
    from nfp_vintages.diagnostics import combination_gate
    cells = {("normal", "t1"): {"actual": [1.0]*6, "model": [1.1]*6, "consensus": [0.9]*6}}
    out = combination_gate(cells)
    assert ("normal", "t1") in out and "fires" in out[("normal", "t1")]
```

Run the full non-network suite (catches any import/wiring regression in the script's importable helpers):
`uv run pytest -m "not network" --no-cov`
Expected: green. Then `uv run ruff check .` clean.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_a5_backtest.py packages/nfp-vintages/src/nfp_vintages/tests/test_diagnostics.py
git commit -m "feat(a5): private-consensus competitor + combination gate in cmd_score [plans/18 T5]

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Total-error decomposition + implied-govt benchmark in `cmd_total` (Track B); env wiring

**Files:**
- Modify: `scripts/run_a5_backtest.py` (the `cmd_total` function)
- Modify: `.env.example`

**Interfaces:**
- Consumes: the existing `cmd_total` assembly (`assemble_total`, `score_total`, the per-target `total_actual`,
  `cons` Total-consensus value at `~538`), `ImpliedGovernment` (T4), `first_print_changes('05')` (read-only).
- Produces: a Total-error **decomposition** (private-side vs government-side) and the **implied-govt** column
  in the Total report; the documented `NFP_CONSENSUS_PATH` env var.

- [ ] **Step 1: Add the implied-govt benchmark + decomposition to `cmd_total`**

Alongside the existing `consensus = Consensus(load_consensus())` (`~499`), add:

```python
from nfp_vintages.competitors.consensus import ImpliedGovernment
implied_govt = ImpliedGovernment(load_consensus())
```

Per target (where `total_actual` and the private first print are both available), record:
- **private-side error** = private nowcast − private first print (`first_print_change_k` for `'05'`);
- **government-side error** = wedge mean − (Total first print − private first print)  *(the realized government change)*;
- **implied-govt benchmark** = `implied_govt.predict(target, as_of=as_of)` (its error vs the realized government change).

Render a `## Total-error decomposition` section: per target and pooled, `total_err = private_err + govt_err`
(identity check), plus the implied-govt benchmark error beside the wedge error.

- [ ] **Step 2: Document the env var**

In `.env.example`, under the data/competitor URIs, add:

```
# Consensus survey file (Total '00' + private '05'); unset ⇒ local data/competitors/consensus.parquet
NFP_CONSENSUS_PATH=s3://alt-nfp/competitors/consensus.parquet
```

- [ ] **Step 3: Verify**

`uv run pytest -m "not network" --no-cov` green; `uv run ruff check .` clean. (The decomposition is an
arithmetic identity over values already computed in `cmd_total`; the unit-tested pieces are T1–T4.)

- [ ] **Step 4: Commit**

```bash
git add scripts/run_a5_backtest.py .env.example
git commit -m "feat(a5): Total-error decomposition + implied-govt benchmark in cmd_total [plans/18 T6]

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Out of scope (deferred to the Bloomberg port — spec §12.5/§12.6)

- The `model_combo` combination layer (post-hoc blend of `model_5a` + `consensus`) — gated, and the gate
  cannot fire on the local providerless skeleton.
- Any in-model consensus conditioning (the `5A→5B`-style escalation).
- Folding implied-govt into the **wedge's priors** (the possible RIF-input-gate retirement, §12.2) — validate
  the wedge against implied-govt first, on the port.
- The actual gate **firing**, combination **weights**, tuning, and keep/drop — full-regime decisions.

## Self-review

- **Spec coverage (§12.6 build-now list):** wire 3 consensus competitors → private `'05'` (T5), implied-govt
  `Total−Private` (T4, used T6), Total `'00'` (already in `cmd_total`) ✓; Total-error decomposition (T6) ✓;
  gate computation (T2+T3, wired T5) ✓; `NFP_CONSENSUS_PATH` wiring (T6) ✓. Deferred items match §12.6 ✓.
- **Firewall:** no edits to `a5.py`/`first_print.py`/`wedge_data.py`/`model_data.py`/`nfp-model`/goldens; new
  code only in `diagnostics.py`, `competitors/consensus.py`, `run_a5_backtest.py`, `.env.example` ✓.
- **Naming:** `combination_gate` is distinct from the existing `gate_decision` (the §4 Aruoba gate) ✓.
- **Type consistency:** `EncompassingResult` fields used identically in T2/T3; `predict(ref_month, *, as_of)
  -> float | None` matches the `Competitor` protocol for `ImpliedGovernment` ✓.
- **No placeholders:** every code step carries runnable code; T5/T6 are script-integration tasks (precise
  edit sites + the gate math unit-tested in T2/T3) — the one acceptable prose-over-code spot, by design.
