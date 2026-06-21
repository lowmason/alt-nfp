# A5 — Real competitors in the harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score the banked model against real competitors (consensus, a smart
bridge baseline, naive floors) on the CES **first print**, at two
near-release information regimes (T−7, T−1), via a new `a5_report.md`
scoreboard — without touching any `nfp-model` or golden-mastered path.

**Architecture:** Pure evaluation layer. A new first-print *target* extractor
reads store **levels** (additive, Option A from `specs/ces_growth_convention.md`).
Competitors implement a small protocol producing a `change_k` per
(ref_month, regime). Two new snapshot grids (model run at
`BLS-release(M) − {7,1}` days) reuse the A4 batched harness unchanged. A
scoreboard scores everyone against the first print.

**Tech Stack:** Python 3.12, Polars (lazy store I/O), NumPy, pytest, the
existing `nfp_ingest` store/snapshot/model-data APIs and `nfp_model` batched
fit. Design of record: `specs/a5_real_competitors.md`; consensus contract:
`specs/bloomberg_consensus.md`.

**Before starting:** we are on `main`. Create a working branch:
`git switch -c a5-real-competitors`. All commits below land there.

---

## File structure

| File | Responsibility |
|---|---|
| `packages/nfp-ingest/src/nfp_ingest/first_print.py` | **CREATE** — Option A first-print target extractor over store levels |
| `packages/nfp-ingest/tests/test_first_print.py` | **CREATE** — extractor tests vs published headlines (store-skip) |
| `packages/nfp-vintages/src/nfp_vintages/competitors/__init__.py` | **CREATE** — `Competitor` protocol, `Prediction` type, registry |
| `packages/nfp-vintages/src/nfp_vintages/competitors/naive.py` | **CREATE** — random-walk + trailing-12-month-mean floors |
| `packages/nfp-vintages/src/nfp_vintages/competitors/consensus.py` | **CREATE** — `load_consensus()` pluggable adapter (contract schema) |
| `packages/nfp-vintages/src/nfp_vintages/competitors/bridge.py` | **CREATE (cut-line)** — vintage-censored claims/JOLTS bridge regression |
| `packages/nfp-vintages/src/nfp_vintages/a5.py` | **CREATE** — `release_date_for`, `near_release_asof`, `score`, scoreboard assembly |
| `packages/nfp-vintages/tests/test_competitors.py` | **CREATE** — naive + consensus + protocol (synthetic, no store) |
| `packages/nfp-vintages/tests/test_a5.py` | **CREATE** — as-of helper + scoring math (synthetic; store-skip for release-date lookup) |
| `scripts/run_a5_backtest.py` | **CREATE** — grids (T−7/T−1) + batched fits + scoreboard, reusing A4 machinery |
| `plans/0-port_and_staged_plan.md` | **MODIFY** — A5 gate annotation on completion |

---

## Task 1: First-print target extractor

**Files:**
- Create: `packages/nfp-ingest/src/nfp_ingest/first_print.py`
- Test: `packages/nfp-ingest/tests/test_first_print.py`

The first print BLS announces for month *p* is `L(p, rev0) − L(p−1, partner)`
in thousands, where the partner is the prior month's second print
`(rev1, bmr0)`, falling back at benchmark months to the prior month's latest
published level `(max benchmark_revision, max revision, max vintage_date)`.
Per `specs/ces_growth_convention.md` §5. Units: store CES employment is in
thousands, so the level difference is `change_k` directly.

- [ ] **Step 1: Write the failing test**

```python
# packages/nfp-ingest/tests/test_first_print.py
"""A5 first-print extractor: the within-release headline change BLS announces.

Validated against published headlines. Skips when the store is unavailable.
"""
from __future__ import annotations

from datetime import date

import pytest
from nfp_ingest.first_print import first_print_changes
from nfp_lookups.paths import VINTAGE_STORE_PATH


def _store_available() -> bool:
    try:
        return VINTAGE_STORE_PATH.exists() and (
            next(VINTAGE_STORE_PATH.glob("**/*.parquet"), None) is not None
        )
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _store_available(),
    reason="Vintage store not available",
)


def _change_for(df, y: int, m: int) -> float:
    row = df.filter(
        (df["ref_date"].dt.year() == y) & (df["ref_date"].dt.month() == m)
    )
    assert row.height == 1, f"expected one row for {y}-{m:02d}, got {row.height}"
    return float(row["first_print_change_k"][0])


def test_ordinary_month_headline_2025_07():
    # L(Jul-25 rev0) 159,539 − L(Jun-25 rev1) 159,466 = +73k (published headline)
    df = first_print_changes()
    assert _change_for(df, 2025, 7) == pytest.approx(73.0, abs=1.0)


def test_benchmark_fallback_headline_2026_01():
    # Dec-25 rev1 was shadowed; fall back to Dec-25 (rev2,bmr1) 158,497.
    # L(Jan-26 rev0) 158,627 − 158,497 = +130k (published headline)
    df = first_print_changes()
    assert _change_for(df, 2026, 1) == pytest.approx(130.0, abs=1.0)


def test_growth_and_change_consistent():
    import numpy as np

    df = first_print_changes().drop_nulls("first_print_change_k")
    # change_k and growth agree: change ≈ expm1(growth) * L_prev, and both signs match
    assert (df["first_print_growth"].is_not_null()).all()
    g = df["first_print_growth"].to_numpy()
    c = df["first_print_change_k"].to_numpy()
    assert np.all(np.sign(g) == np.sign(c))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/nfp-ingest && uv run pytest tests/test_first_print.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nfp_ingest.first_print'`.

- [ ] **Step 3: Write minimal implementation**

```python
# packages/nfp-ingest/src/nfp_ingest/first_print.py
"""A5 first-print target: the within-release headline change BLS announces.

Option A from ``specs/ces_growth_convention.md`` §5 — an *additive* read over
store **levels**. Touches no golden-mastered path: it computes a new derived
series and never alters the panel ``growth`` column or any selection logic.

The first print for reference month ``p`` is

    change_k(p) = L(p, rev0, bmr0) − L(p−1, partner)

with ``partner`` = the prior month's second print ``(rev1, bmr0)`` as
published alongside ``p``'s first print; at benchmark months where that row
is absent/shadowed, fall back to the prior month's latest published level
(highest ``(benchmark_revision, revision, vintage_date)``). CES employment is
in thousands, so the level difference is ``change_k`` directly.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
from nfp_lookups.paths import VINTAGE_STORE_PATH

from nfp_ingest.vintage_store import read_vintage_store


def first_print_changes(
    *,
    store_path: Path = VINTAGE_STORE_PATH,
    geographic_type: str = "national",
    geographic_code: str = "00",
    industry_type: str = "national",
    industry_code: str = "00",
) -> pl.DataFrame:
    """Per reference month: the first-print headline change and growth.

    Returns a DataFrame sorted by ``ref_date`` with columns
    ``ref_date``, ``first_print_growth``, ``first_print_change_k``,
    ``vintage_date`` (the first-print release date). Months with no partner
    (history edge) get null growth/change.
    """
    levels = (
        read_vintage_store(
            store_path,
            source="ces",
            seasonally_adjusted=True,
            geographic_type=geographic_type,
            geographic_code=geographic_code,
            industry_type=industry_type,
            industry_code=industry_code,
        )
        .select("ref_date", "vintage_date", "revision", "benchmark_revision", "employment")
        .with_columns(pl.col("ref_date").dt.truncate("1mo").alias("period"))
        .collect()
    )

    first = (
        levels.filter((pl.col("revision") == 0) & (pl.col("benchmark_revision") == 0))
        .sort("vintage_date")
        .group_by("period")
        .last()  # one first print per month (earliest is the rev-0 release)
        .select(
            "period",
            pl.col("employment").alias("L_p"),
            pl.col("vintage_date"),
        )
        .with_columns(prev_period=pl.col("period").dt.offset_by("-1mo"))
    )

    rev1 = (
        levels.filter((pl.col("revision") == 1) & (pl.col("benchmark_revision") == 0))
        .sort("vintage_date")
        .group_by("period")
        .first()
        .select("period", pl.col("employment").alias("L_prev_primary"))
    )

    latest = (
        levels.sort("benchmark_revision", "revision", "vintage_date")
        .group_by("period")
        .last()
        .select("period", pl.col("employment").alias("L_prev_fallback"))
    )

    out = (
        first.join(rev1, left_on="prev_period", right_on="period", how="left")
        .join(latest, left_on="prev_period", right_on="period", how="left")
        .with_columns(L_prev=pl.coalesce("L_prev_primary", "L_prev_fallback"))
        .with_columns(
            first_print_change_k=(pl.col("L_p") - pl.col("L_prev")),
            first_print_growth=(pl.col("L_p").log() - pl.col("L_prev").log()),
        )
        .select(
            pl.col("period").alias("ref_date"),
            "first_print_growth",
            "first_print_change_k",
            "vintage_date",
        )
        .sort("ref_date")
    )
    return out


__all__ = ["first_print_changes"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/nfp-ingest && uv run pytest tests/test_first_print.py -v`
Expected: PASS (3 tests). If `test_benchmark_fallback_headline_2026_01` fails,
confirm Dec-2025 truly lacks a `(rev1, bmr0)` row in the store (the §4c-i
shadow) so the fallback path is exercised.

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-ingest/src/nfp_ingest/first_print.py \
        packages/nfp-ingest/tests/test_first_print.py
git commit -m "feat(a5): first-print target extractor (Option A over store levels)"
```

---

## Task 2: Competitor protocol + naive floors

**Files:**
- Create: `packages/nfp-vintages/src/nfp_vintages/competitors/__init__.py`
- Create: `packages/nfp-vintages/src/nfp_vintages/competitors/naive.py`
- Test: `packages/nfp-vintages/tests/test_competitors.py`

A competitor maps a target reference month + an as-of date to a predicted
`change_k` (or `None` if it has no value there). Naive floors operate on the
first-print history, censored to the as-of (only months whose first print was
released `<= as_of`).

- [ ] **Step 1: Write the failing test**

```python
# packages/nfp-vintages/tests/test_competitors.py
"""A5 competitor adapters — naive floors + consensus (synthetic, no store)."""
from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from nfp_vintages.competitors.naive import RandomWalk, TrailingMean


def _history() -> pl.DataFrame:
    # months 2024-01..2024-06, change_k = 100..150, released the next month's 5th
    rows = []
    for i, m in enumerate(range(1, 7)):
        rows.append(
            {
                "ref_date": date(2024, m, 1),
                "first_print_change_k": 100.0 + 10 * i,
                "vintage_date": date(2024, m + 1, 5),
            }
        )
    return pl.DataFrame(rows)


def test_random_walk_repeats_last_published_print():
    hist = _history()
    rw = RandomWalk(hist)
    # nowcasting 2024-07 as of 2024-07-31: last published is 2024-06 (=150)
    assert rw.predict(date(2024, 7, 1), as_of=date(2024, 7, 31)) == pytest.approx(150.0)


def test_random_walk_respects_as_of_censoring():
    hist = _history()
    rw = RandomWalk(hist)
    # as of 2024-06-10: 2024-05's print (released 2024-06-05) is the latest known
    assert rw.predict(date(2024, 6, 1), as_of=date(2024, 6, 10)) == pytest.approx(140.0)


def test_trailing_mean_of_known_prints():
    hist = _history()
    tm = TrailingMean(hist, window=3)
    # as of 2024-07-31, last 3 known prints = 130,140,150 -> mean 140
    assert tm.predict(date(2024, 7, 1), as_of=date(2024, 7, 31)) == pytest.approx(140.0)


def test_predict_none_when_no_history_available():
    hist = _history()
    rw = RandomWalk(hist)
    assert rw.predict(date(2024, 1, 1), as_of=date(2024, 1, 1)) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/nfp-vintages && uv run pytest tests/test_competitors.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nfp_vintages.competitors'`.

- [ ] **Step 3: Write minimal implementation**

```python
# packages/nfp-vintages/src/nfp_vintages/competitors/__init__.py
"""A5 competitors: each maps (ref_month, as_of) -> predicted change_k.

Competitors are scored against the first-print target across the T−7 and T−1
regimes (``specs/a5_real_competitors.md``). The protocol is keyed on
ref_month + as_of so the same adapters extend to supersector series later
(B1) without a rebuild.
"""
from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable


@runtime_checkable
class Competitor(Protocol):
    name: str

    def predict(self, ref_month: date, *, as_of: date) -> float | None:
        """Predicted change_k for ``ref_month`` using only data known by
        ``as_of``; ``None`` if the competitor has no value there."""
        ...


__all__ = ["Competitor"]
```

```python
# packages/nfp-vintages/src/nfp_vintages/competitors/naive.py
"""Naive baseline competitors: sanity floors, never gates."""
from __future__ import annotations

from datetime import date

import polars as pl


def _known(history: pl.DataFrame, as_of: date) -> pl.DataFrame:
    """First-print rows released on or before ``as_of``, oldest→newest."""
    return (
        history.filter(pl.col("vintage_date") <= as_of)
        .drop_nulls("first_print_change_k")
        .sort("ref_date")
    )


class RandomWalk:
    """Predict the last published first-print change."""

    name = "naive_rw"

    def __init__(self, history: pl.DataFrame) -> None:
        self.history = history

    def predict(self, ref_month: date, *, as_of: date) -> float | None:
        known = _known(self.history, as_of)
        if known.height == 0:
            return None
        return float(known["first_print_change_k"][-1])


class TrailingMean:
    """Predict the mean of the last ``window`` published first-print changes."""

    name = "naive_mean"

    def __init__(self, history: pl.DataFrame, window: int = 12) -> None:
        self.history = history
        self.window = window

    def predict(self, ref_month: date, *, as_of: date) -> float | None:
        known = _known(self.history, as_of)
        if known.height == 0:
            return None
        tail = known["first_print_change_k"][-self.window :]
        return float(tail.mean())


__all__ = ["RandomWalk", "TrailingMean"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/nfp-vintages && uv run pytest tests/test_competitors.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/competitors/__init__.py \
        packages/nfp-vintages/src/nfp_vintages/competitors/naive.py \
        packages/nfp-vintages/tests/test_competitors.py
git commit -m "feat(a5): competitor protocol + naive floors (random-walk, trailing-mean)"
```

---

## Task 3: Consensus adapter (pluggable, staged)

**Files:**
- Create: `packages/nfp-vintages/src/nfp_vintages/competitors/consensus.py`
- Modify: `packages/nfp-vintages/tests/test_competitors.py` (append)

`load_consensus(path)` reads the contract file (`specs/bloomberg_consensus.md`
§1), validates it, and returns a DataFrame — or `None` when no file is
configured (the staged state). The `Consensus` competitor is **T−1-only**:
it returns `None` unless the as-of is on/after the survey lock and on/before
the release.

- [ ] **Step 1: Write the failing test (append to test_competitors.py)**

```python
def _consensus_file(tmp_path):
    df = pl.DataFrame(
        {
            "ref_month": [date(2024, 5, 1), date(2024, 6, 1)],
            "consensus_median_change_k": [180.0, 190.0],
            "survey_date": [date(2024, 6, 5), date(2024, 7, 3)],
            "release_date": [date(2024, 6, 7), date(2024, 7, 5)],
            "source": ["bloomberg", "bloomberg"],
        }
    )
    p = tmp_path / "consensus.parquet"
    df.write_parquet(p)
    return p


def test_load_consensus_absent_returns_none(tmp_path):
    from nfp_vintages.competitors.consensus import load_consensus

    assert load_consensus(tmp_path / "missing.parquet") is None


def test_load_consensus_validates_and_reads(tmp_path):
    from nfp_vintages.competitors.consensus import load_consensus

    df = load_consensus(_consensus_file(tmp_path))
    assert df is not None
    assert df.height == 2
    assert set(["ref_month", "consensus_median_change_k", "survey_date",
                "release_date", "source"]).issubset(df.columns)


def test_consensus_competitor_t1_lookup(tmp_path):
    from nfp_vintages.competitors.consensus import Consensus, load_consensus

    c = Consensus(load_consensus(_consensus_file(tmp_path)))
    # at T−1 (release_date − 1 = 2024-07-04) consensus for 2024-06 is known
    assert c.predict(date(2024, 6, 1), as_of=date(2024, 7, 4)) == pytest.approx(190.0)


def test_consensus_none_when_unconfigured():
    from nfp_vintages.competitors.consensus import Consensus

    c = Consensus(None)
    assert c.predict(date(2024, 6, 1), as_of=date(2024, 7, 4)) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/nfp-vintages && uv run pytest tests/test_competitors.py -k consensus -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nfp_vintages.competitors.consensus'`.

- [ ] **Step 3: Write minimal implementation**

```python
# packages/nfp-vintages/src/nfp_vintages/competitors/consensus.py
"""Consensus survey-median competitor (pluggable, Bloomberg-sourced).

Reads the contract file defined in ``specs/bloomberg_consensus.md`` §1, or
returns ``None`` when unconfigured (the staged state — the scoreboard then
renders the consensus column as ``—``). A T−1-only competitor: the street
median locks ~release-eve.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import polars as pl

_REQUIRED = ("ref_month", "consensus_median_change_k", "survey_date",
             "release_date", "source")


def consensus_path(path: str | Path | None = None) -> Path:
    """Resolve path → arg → ``NFP_CONSENSUS_PATH`` → default."""
    if path is not None:
        return Path(path)
    env = os.environ.get("NFP_CONSENSUS_PATH")
    if env:
        return Path(env)
    from nfp_lookups.paths import DATA_DIR

    return DATA_DIR / "competitors" / "consensus.parquet"


def load_consensus(path: str | Path | None = None) -> pl.DataFrame | None:
    """Load + validate the consensus file, or ``None`` if it does not exist."""
    p = consensus_path(path)
    if not p.exists():
        return None
    df = pl.read_parquet(p)
    missing = set(_REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"consensus file missing required columns: {sorted(missing)}")
    if df["ref_month"].n_unique() != df.height:
        raise ValueError("consensus ref_month must be unique")
    bad = df.filter(pl.col("survey_date") >= pl.col("release_date"))
    if bad.height:
        raise ValueError("consensus survey_date must precede release_date")
    return df.sort("ref_month")


class Consensus:
    """T−1-only competitor: returns the median once the survey has locked."""

    name = "consensus"

    def __init__(self, table: pl.DataFrame | None) -> None:
        self.table = table

    def predict(self, ref_month: date, *, as_of: date) -> float | None:
        if self.table is None:
            return None
        row = self.table.filter(pl.col("ref_month") == ref_month)
        if row.height == 0:
            return None
        survey = row["survey_date"][0]
        if as_of < survey:  # not locked yet (e.g. T−7)
            return None
        return float(row["consensus_median_change_k"][0])


__all__ = ["consensus_path", "load_consensus", "Consensus"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/nfp-vintages && uv run pytest tests/test_competitors.py -v`
Expected: PASS (all, including the 5 consensus tests).

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/competitors/consensus.py \
        packages/nfp-vintages/tests/test_competitors.py
git commit -m "feat(a5): pluggable consensus adapter (contract schema, T-1, staged)"
```

---

## Task 4: Near-release as-of helper + scoring math

**Files:**
- Create: `packages/nfp-vintages/src/nfp_vintages/a5.py`
- Test: `packages/nfp-vintages/tests/test_a5.py`

`release_date_for(ref_month)` returns the actual first-print release date from
the store (the rev-0 `vintage_date`), falling back to the first-Friday-of-next-
month rule. `near_release_asof(ref_month, days_before)` subtracts. `score`
returns ME/MAE/RMSE over an error array.

- [ ] **Step 1: Write the failing test**

```python
# packages/nfp-vintages/tests/test_a5.py
"""A5 helpers: near-release as-of dates + scoring math."""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from nfp_vintages.a5 import first_friday_release, near_release_asof, score


def test_first_friday_release_basic():
    # June-2025 reference -> first Friday of July 2025 = 2025-07-04 -> shifts +7
    # (Independence Day) -> 2025-07-11
    assert first_friday_release(date(2025, 6, 1)) == date(2025, 7, 11)


def test_first_friday_release_ordinary():
    # May-2025 reference -> first Friday of June 2025 = 2025-06-06
    assert first_friday_release(date(2025, 5, 1)) == date(2025, 6, 6)


def test_near_release_asof_offsets():
    rel = date(2025, 6, 6)
    assert near_release_asof(date(2025, 5, 1), days_before=1, release=rel) == date(2025, 6, 5)
    assert near_release_asof(date(2025, 5, 1), days_before=7, release=rel) == date(2025, 5, 30)


def test_score_metrics():
    errors = np.array([10.0, -20.0, 30.0])
    m = score(errors)
    assert m["me"] == pytest.approx(20.0 / 3)
    assert m["mae"] == pytest.approx(20.0)
    assert m["rmse"] == pytest.approx(np.sqrt((100 + 400 + 900) / 3))
    assert m["n"] == 3


def test_score_empty():
    m = score(np.array([]))
    assert m["n"] == 0
    assert m["mae"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/nfp-vintages && uv run pytest tests/test_a5.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nfp_vintages.a5'`.

- [ ] **Step 3: Write minimal implementation**

```python
# packages/nfp-vintages/src/nfp_vintages/a5.py
"""A5 scoreboard helpers: near-release as-of dates and error metrics.

``release_date_for`` prefers the store's actual rev-0 vintage_date; callers
without a store use ``first_friday_release`` (the BLS first-Friday-of-next-
month rule, holiday-shifted). Kept local to avoid importing a private
``release_dates`` symbol across packages.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl


def _first_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def first_friday_release(ref_month: date) -> date:
    """BLS first-print release date for a reference month (first Friday of the
    next month; shifted +7 days if it lands on Jan 1 or Jul 4)."""
    y, m = ref_month.year, ref_month.month
    pm, py = (1, y + 1) if m == 12 else (m + 1, y)
    friday = _first_friday(py, pm)
    if (friday.month == 1 and friday.day == 1) or (friday.month == 7 and friday.day == 4):
        friday += timedelta(days=7)
    return friday


def release_date_for(
    ref_month: date,
    *,
    store_path: Path | None = None,
) -> date:
    """Actual first-print release date from the store (rev-0 vintage_date),
    falling back to ``first_friday_release`` when the store has no such row."""
    if store_path is not None:
        from nfp_ingest.vintage_store import read_vintage_store

        lf = read_vintage_store(
            store_path,
            source="ces",
            seasonally_adjusted=True,
            geographic_type="national",
            industry_code="00",
        ).filter(
            (pl.col("revision") == 0)
            & (pl.col("benchmark_revision") == 0)
            & (pl.col("ref_date").dt.truncate("1mo") == ref_month.replace(day=1))
        )
        got = lf.select(pl.col("vintage_date").min()).collect()
        if got.height and got["vintage_date"][0] is not None:
            return got["vintage_date"][0]
    return first_friday_release(ref_month)


def near_release_asof(
    ref_month: date,
    *,
    days_before: int,
    release: date | None = None,
    store_path: Path | None = None,
) -> date:
    """As-of date = release(ref_month) − days_before."""
    rel = release if release is not None else release_date_for(ref_month, store_path=store_path)
    return rel - timedelta(days=days_before)


def score(errors: np.ndarray) -> dict:
    """ME / MAE / RMSE over an error array (actual − predicted)."""
    e = np.asarray(errors, dtype=float)
    e = e[~np.isnan(e)]
    if e.size == 0:
        return {"n": 0, "me": None, "mae": None, "rmse": None}
    return {
        "n": int(e.size),
        "me": float(e.mean()),
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt((e**2).mean())),
    }


__all__ = [
    "first_friday_release",
    "release_date_for",
    "near_release_asof",
    "score",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/nfp-vintages && uv run pytest tests/test_a5.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/a5.py \
        packages/nfp-vintages/tests/test_a5.py
git commit -m "feat(a5): near-release as-of helper + scoring math"
```

---

## Task 5: Snapshot grids at T−7 and T−1

**Files:**
- Create: `scripts/run_a5_backtest.py` (the `snapshot` subcommand)

Mirror `scripts/run_a4_backtest.py`'s `cmd_snapshot`, but build **two**
grids — one per regime — at `near_release_asof(M, days_before=k)` for
`k ∈ {7, 1}`, and attach the first-print target from Task 1. The uncensored
panel and `idx_to_level` provenance are computed exactly as in A4.

- [ ] **Step 1: Scaffold the script with shared helpers and `snapshot`**

```python
# scripts/run_a5_backtest.py
"""A5 backtest — model vs competitors on the first print, at T−7 and T−1.

    uv run python scripts/run_a5_backtest.py snapshot data/backtests/a5
    uv run python scripts/run_a5_backtest.py batched  data/backtests/a5
    uv run python scripts/run_a5_backtest.py score    data/backtests/a5

Reuses the A4 batched harness verbatim (``fit_model_batch``); only the as-of
dates differ (release(M) − {7,1}). Snapshots live under ``<root>/<regime>/``.
"""
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

N_BACKTEST = 24
END_YEAR = 2026
PRESET = "light"
BATCH_SEED = 9100
REGIMES = {"t7": 7, "t1": 1}  # name -> days_before release


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text())


def _write_json(p: Path, obj: dict) -> None:
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def cmd_snapshot(root: Path) -> None:
    import polars as pl
    from nfp_ingest.first_print import first_print_changes
    from nfp_ingest.model_data import PROVIDERS_DEFAULT, panel_to_model_data
    from nfp_ingest.panel import build_panel
    from nfp_ingest.snapshots import load_snapshot, snapshot_model_data
    from nfp_lookups.paths import VINTAGE_STORE_PATH
    from nfp_vintages.a5 import near_release_asof

    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "grid_manifest.json"
    manifest: dict = _read_json(manifest_path) if manifest_path.exists() else {"regimes": {}}

    print("Building uncensored panel (truth side)...", flush=True)
    panel_full = build_panel(end_year=END_YEAR)
    data_full = panel_to_model_data(panel_full, list(PROVIDERS_DEFAULT))
    dates = data_full["dates"]
    levels = data_full["levels"]
    ces_sa_index = levels["ces_sa_index"].to_numpy().astype(float)
    base_row_idx = int(np.argmin(np.abs(ces_sa_index - 100.0)))
    ces_sa_base_level = float(levels["ces_sa_level"].to_numpy().astype(float)[base_row_idx])
    idx_to_level = ces_sa_base_level / 100.0

    fp = first_print_changes()  # ref_date -> first_print_change_k
    fp_map = dict(
        fp.select(["ref_date", "first_print_change_k"]).iter_rows()
    )

    T = len(dates)
    target_indices = list(range(T - N_BACKTEST, T))
    manifest["provenance"] = {
        "base_index": float(ces_sa_index[0]),
        "idx_to_level": idx_to_level,
        "end_year": END_YEAR,
        "preset": PRESET,
        "n_backtest": N_BACKTEST,
    }

    for rname, days_before in REGIMES.items():
        snap_dir = root / rname
        reg = manifest["regimes"].setdefault(rname, {"days_before": days_before, "targets": {}})
        for n, t_idx in enumerate(target_indices):
            target = dates[t_idx]
            key = target.isoformat()
            as_of = near_release_asof(
                target, days_before=days_before, store_path=VINTAGE_STORE_PATH
            )
            hits = sorted((snap_dir / f"asof={as_of.isoformat()}").glob("model_data_*.npz"))
            path = hits[0] if hits else None
            if path is None:
                print(f"[{rname} {n + 1}/{N_BACKTEST}] target {key} as_of {as_of}: building", flush=True)
                try:
                    path, _ = snapshot_model_data(as_of, out_root=snap_dir, end_year=END_YEAR)
                except Exception as e:  # noqa: BLE001 — A1 negative-master pattern
                    print(f"  UNBUILDABLE: {e}", flush=True)
                    reg["targets"][key] = {"error": str(e), "as_of": as_of.isoformat()}
                    _write_json(manifest_path, manifest)
                    continue
            _, meta = load_snapshot(path)
            cdates = [date.fromisoformat(d) for d in meta["dates"]]
            c_idx = cdates.index(target) if target in cdates else len(cdates) - 1
            actual_index = float(ces_sa_index[t_idx])
            prev_index = float(ces_sa_index[t_idx - 1])
            reg["targets"][key] = {
                "t_idx": t_idx,
                "as_of": as_of.isoformat(),
                "T": len(cdates),
                "c_idx": int(c_idx),
                "content_hash": meta["content_hash"],
                "snapshot": str(path.relative_to(root)),
                "first_print_change_k": fp_map.get(target),
                "best_avail_change_k": (actual_index - prev_index) * idx_to_level,
            }
            _write_json(manifest_path, manifest)
    print(f"Grid built under {root}")
```

- [ ] **Step 2: Run the snapshot build (store required)**

Run: `uv run python scripts/run_a5_backtest.py snapshot data/backtests/a5`
Expected: prints per-regime per-target lines; `grid_manifest.json` lists both
`t7` and `t1` regimes with `as_of` ≈ release−{7,1}, a `first_print_change_k`
for most targets, and any unbuildable frontier months recorded (not fatal).

- [ ] **Step 3: Smoke-check the manifest**

Run: `uv run python -c "import json;m=json.load(open('data/backtests/a5/grid_manifest.json'));print({r:len(v['targets']) for r,v in m['regimes'].items()})"`
Expected: both regimes present with ~24 targets each.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_a5_backtest.py
git commit -m "feat(a5): T-7/T-1 snapshot grids with first-print target"
```

---

## Task 6: Batched fits for both grids

**Files:**
- Modify: `scripts/run_a5_backtest.py` (add `cmd_batched`)

Reuse `pad_model_inputs` + `fit_model_batch` exactly as A4 does, once per
regime, writing per-date reductions.

- [ ] **Step 1: Add `cmd_batched`**

```python
def cmd_batched(root: Path) -> None:
    from nfp_model import fit_model_batch, model_inputs, pad_model_inputs
    from nfp_model.batch import active_cyclicals  # noqa: F401 (parity of imports)
    from nfp_ingest.snapshots import load_snapshot
    from nfp_model import from_snapshot

    manifest = _read_json(root / "grid_manifest.json")
    prov = manifest["provenance"]

    def _data(snap_rel: str):
        path = root / snap_rel
        arrays, meta = load_snapshot(path)
        return from_snapshot(arrays, meta)

    for rname in REGIMES:
        reg = manifest["regimes"][rname]
        targets = [(k, t) for k, t in sorted(reg["targets"].items()) if "error" not in t]
        if not targets:
            continue
        print(f"[{rname}] loading {len(targets)} snapshots...", flush=True)
        inputs = [model_inputs(_data(t["snapshot"])) for _, t in targets]
        c_idx = [int(t["c_idx"]) for _, t in targets]
        bi = pad_model_inputs(inputs, c_idx=c_idx)
        t0 = time.time()
        batch = fit_model_batch(
            bi,
            settings=PRESET,
            seed=BATCH_SEED,
            base_index=float(prov["base_index"]),
            idx_to_level=float(prov["idx_to_level"]),
        )
        print(f"[{rname}] batched fit {batch.wall_seconds / 60:.1f} min "
              f"({time.time() - t0:.0f}s wall)", flush=True)
        entries: dict = {}
        for i, (key, _t) in enumerate(targets):
            arrays, meta = batch.date_arrays(i)
            np.savez(root / f"{rname}_batched_{key}.npz", **arrays)
            entries[key] = meta
        _write_json(root / f"{rname}_batched_manifest.json",
                    {"entries": entries, "batch_wall_seconds": round(batch.wall_seconds, 1)})
    print("Batched fits complete.")
```

- [ ] **Step 2: Run the batched fits (store + compute)**

Run: `uv run python scripts/run_a5_backtest.py batched data/backtests/a5`
Expected: per-regime "batched fit … min" lines; `t7_batched_manifest.json`
and `t1_batched_manifest.json` written, each with one `nowcast_change_k` per
target and `num_divergences` near zero.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_a5_backtest.py
git commit -m "feat(a5): batched model fits for T-7/T-1 grids"
```

---

## Task 7: Scoreboard + report

**Files:**
- Modify: `scripts/run_a5_backtest.py` (add `cmd_score`, `main`)

Assemble model + competitors per regime, score against the first print, write
`a5_report.md` + `a5_results.parquet`.

- [ ] **Step 1: Add `cmd_score` and `main`**

```python
def cmd_score(root: Path) -> int:
    import polars as pl
    from nfp_ingest.first_print import first_print_changes
    from nfp_vintages.a5 import near_release_asof, score
    from nfp_vintages.competitors.consensus import Consensus, load_consensus
    from nfp_vintages.competitors.naive import RandomWalk, TrailingMean
    from nfp_lookups.paths import VINTAGE_STORE_PATH

    manifest = _read_json(root / "grid_manifest.json")
    fp = first_print_changes()
    fp_hist = fp.select(["ref_date", "first_print_change_k", "vintage_date"])
    consensus = Consensus(load_consensus())  # None until Bloomberg file lands → "—"
    naive_rw, naive_mean = RandomWalk(fp_hist), TrailingMean(fp_hist, window=12)

    rows = []
    for rname, days_before in REGIMES.items():
        reg = manifest["regimes"][rname]
        batched = _read_json(root / f"{rname}_batched_manifest.json")["entries"]
        for key, t in sorted(reg["targets"].items()):
            if "error" in t or key not in batched:
                continue
            ref = date.fromisoformat(key)
            as_of = date.fromisoformat(t["as_of"])
            actual = t["first_print_change_k"]
            if actual is None:
                continue
            model = batched[key]["nowcast_change_k"]
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
                    "competitor": comp,
                    "pred_change_k": pred,
                    "actual_first_print_k": actual,
                    "error_k": None if pred is None else actual - pred,
                })

    df = pl.DataFrame(rows)
    # Exclude COVID (2020–2021) from headline metrics (decided-questions rule)
    scored = df.filter(
        pl.col("error_k").is_not_null()
        & ~pl.col("ref_month").dt.year().is_in([2020, 2021])
    )
    df.write_parquet(root / "a5_results.parquet")

    lines = ["# A5 backtest report", "",
             "Model vs competitors on the CES **first print**, at T−7 and T−1.",
             "Consensus is T−1-only and renders `—` until the Bloomberg file lands.",
             "COVID (2020–2021) excluded from metrics.", ""]
    for rname in REGIMES:
        lines += [f"## Regime {rname}", "", "| competitor | n | ME | MAE | RMSE |",
                  "|---|---|---|---|---|"]
        for comp in ["model", "consensus", "naive_rw", "naive_mean"]:
            e = scored.filter(
                (pl.col("regime") == rname) & (pl.col("competitor") == comp)
            )["error_k"].to_numpy()
            m = score(e)
            if m["n"] == 0:
                lines.append(f"| {comp} | 0 | — | — | — |")
            else:
                lines.append(
                    f"| {comp} | {m['n']} | {m['me']:+,.0f}k | {m['mae']:,.0f}k "
                    f"| {m['rmse']:,.0f}k |"
                )
        lines.append("")
    (root / "a5_report.md").write_text("\n".join(lines) + "\n")
    print((root / "a5_report.md").read_text())
    return 0


def main() -> None:
    mode, root_arg = sys.argv[1], sys.argv[2]
    root = Path(root_arg).resolve()
    {"snapshot": cmd_snapshot, "batched": cmd_batched}.get(mode, lambda r: None)(root)
    if mode == "score":
        raise SystemExit(cmd_score(root))
    elif mode not in ("snapshot", "batched"):
        raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the scoreboard**

Run: `uv run python scripts/run_a5_backtest.py score data/backtests/a5`
Expected: prints `a5_report.md` — a per-regime table with `model`,
`consensus` (`—`, staged), `naive_rw`, `naive_mean` rows showing ME/MAE/RMSE;
`a5_results.parquet` written.

- [ ] **Step 3: Sanity-check the result**

Verify the model's MAE is finite and the naive floors are present at both
regimes; confirm consensus renders `—` (no Bloomberg file yet). The gate is
**structurally satisfied** (model vs consensus-slot vs naive at each regime).

- [ ] **Step 4: Commit**

```bash
git add scripts/run_a5_backtest.py
git commit -m "feat(a5): scoreboard report (model vs consensus vs naive, T-7/T-1)"
```

---

## Task 8 (cut-line, optional): Smart bridge baseline

**Files:**
- Create: `packages/nfp-vintages/src/nfp_vintages/competitors/bridge.py`
- Modify: `packages/nfp-vintages/tests/test_competitors.py`
- Modify: `scripts/run_a5_backtest.py` (wire into `cmd_score`)

The honest *ceiling*: predict month M's first-print change from
**vintage-censored** claims/JOLTS — reuse each snapshot's `claims_c`/`jolts_c`
arrays (already censored to the as-of and aligned to `dates`), so the
regressors carry the same censoring as the model. Fit an expanding-window OLS
of first-print `change_k` on the cyclical covariates at the nowcast index;
predict M. **Defer this task if the regressor alignment proves fiddly — the
gate is already met by Tasks 1–7.**

- [ ] **Step 1: Write the failing test (synthetic OLS)**

```python
def test_bridge_fits_and_predicts_linear_signal():
    import numpy as np
    from nfp_vintages.competitors.bridge import fit_bridge, predict_bridge

    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 2))
    beta = np.array([1.5, -2.0])
    y = 50 + x @ beta  # exact linear, no noise
    coef = fit_bridge(x, y)
    pred = predict_bridge(coef, np.array([0.5, 0.5]))
    assert pred == pytest.approx(50 + 0.5 * 1.5 + 0.5 * -2.0, abs=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/nfp-vintages && uv run pytest tests/test_competitors.py -k bridge -v`
Expected: FAIL — no module `nfp_vintages.competitors.bridge`.

- [ ] **Step 3: Write minimal implementation**

```python
# packages/nfp-vintages/src/nfp_vintages/competitors/bridge.py
"""Smart bridge baseline: OLS of first-print change_k on vintage-censored
claims/JOLTS. The ceiling competitor — asks whether the full state-space
model beats a cheap regression on the same censored inputs.
"""
from __future__ import annotations

import numpy as np


def fit_bridge(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS with intercept. Returns coef vector [intercept, *slopes]."""
    xd = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(xd, y, rcond=None)
    return coef


def predict_bridge(coef: np.ndarray, x_row: np.ndarray) -> float:
    return float(coef[0] + np.dot(coef[1:], x_row))


__all__ = ["fit_bridge", "predict_bridge"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/nfp-vintages && uv run pytest tests/test_competitors.py -k bridge -v`
Expected: PASS.

- [ ] **Step 5: Wire into `cmd_score`** (extract `claims_c`/`jolts_c` at
`c_idx` from each snapshot for the expanding-window features; target =
first-print `change_k` of months with `vintage_date <= as_of`; add a
`"bridge"` competitor row). Add `bridge` to the report's competitor list.
Re-run `score` and confirm a finite `bridge` row appears at both regimes.

- [ ] **Step 6: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/competitors/bridge.py \
        packages/nfp-vintages/tests/test_competitors.py \
        scripts/run_a5_backtest.py
git commit -m "feat(a5): smart bridge baseline on vintage-censored claims/JOLTS"
```

---

## Task 9: Lint, full suite, gate annotation

**Files:**
- Modify: `plans/0-port_and_staged_plan.md` (A5 gate status block)

- [ ] **Step 1: Lint**

Run: `uv run ruff check .`
Expected: clean (line 100; fix any E/W/F/I/B/C4/UP findings).

- [ ] **Step 2: Run the fast suite**

Run: `uv run pytest -m "not network and not slow" --no-cov`
Expected: green, including the new `test_first_print.py` (skips without store),
`test_competitors.py`, `test_a5.py`.

- [ ] **Step 3: Annotate the A5 gate in plans/0**

Add a `> **Gate status: …**` block under the A5 section quoting: the regimes
(T−7/T−1), the competitor set (consensus staged, naive floors, optional
bridge), the first-print target, and the scoreboard path
(`data/backtests/a5/a5_report.md`). State the gate is structurally met with
consensus pending the Bloomberg file.

- [ ] **Step 4: Move the spec to archive (workflow convention)**

```bash
git mv specs/a5_real_competitors.md archive/a5_real_competitors.md
# Keep specs/bloomberg_consensus.md in specs/ (consensus data not yet sourced).
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "docs(a5): gate annotation in plans/0; archive A5 spec"
```

---

## Self-review

**Spec coverage** (`specs/a5_real_competitors.md`):
- §2a first-print target (Option A) → Task 1. ✓
- §2b revised-truth reference (unscored, optional) → intentionally **not**
  built (the spec marks it "defer if not needed"; the scoreboard scores only
  the first print). Acceptable gap by design.
- §3 regimes T−7/T−1 + grid construction → Tasks 4–6. ✓
- §4 competitors: consensus → Task 3; naive → Task 2; smart baseline → Task 8
  (cut-line); ADP → correctly absent. ✓
- §5 scoreboard + COVID exclusion → Task 7. ✓
- §5 forward-compat (series-keyed protocol) → `Competitor.predict(ref_month,
  as_of)` is series-agnostic; supersector extension needs no protocol change. ✓
- §6 module layout → matches Tasks 1–8 file map. ✓
- §7 sequencing → Tasks ordered cheapest-first (extractor → naive → consensus
  → as-of/scoring → grids → fits → scoreboard → bridge). ✓

**Placeholder scan:** no TBD/TODO; Task 8 Step 5 describes wiring in prose
(it's the cut-line integration, scoped to reuse already-defined
`fit_bridge`/`predict_bridge` + snapshot arrays) — acceptable as the optional
tail, not a core-task placeholder.

**Type consistency:** `first_print_changes()` returns `ref_date,
first_print_growth, first_print_change_k, vintage_date` (used identically in
Tasks 1, 5, 7). Competitor `.predict(ref_month, *, as_of) -> float | None`
consistent across naive/consensus/bridge and the scoreboard loop. `score()`
keys `{n, me, mae, rmse}` consistent between Task 4 and Task 7.
