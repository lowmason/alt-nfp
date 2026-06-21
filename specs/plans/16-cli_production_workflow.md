# CLI Production Workflow Implementation Plan (v2 — rewritten 2026-06-20)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reshape the `alt-nfp` CLI into a production month-T workflow — capture each month's BLS current print via the API and append it to the vintage store, with feed-driven cron automation — and move the one-time historical rebuild to a script.

**Architecture:** Three new CLI commands (`update` / `status` / `watch`) over the existing `append_to_vintage_store` / `compact_partition` store primitives (plus the already-built `capture_*` adapters); the bulk bootstrap moves to `scripts/bootstrap_store.py`; the legacy `download`/`process`/`current`/`build` lineage is retired. **Firewall:** no changes to `nfp-model`, `transform_to_panel`, `build_model_data`, `first_print.py`/`wedge_data.py` *logic*, or the A1/A2/A3 golden paths.

**Tech Stack:** Python 3.12, Typer, Polars 1.41.2, httpx / curl_cffi (BLS/FRED), uv workspace. Tests: pytest (TDD, `-m "not network"`); ruff (line length 100; rules E,W,F,I,B,C4,UP).

**Spec:** [specs/cli_production_workflow.md](../cli_production_workflow.md)

---

## Status — Phases 1–4 are COMPLETE (this rewrite covers Phases 5–9 only)

This is a **rewrite** of the original plan 16. Phases 1–4 are implemented, committed, and verified
on branch `a5-rebuilt-integration` (range `1a47deb..5ec59fd`). They are recorded here so the
remaining phases **consume** their deliverables rather than re-spec them. **Do not re-implement
Phases 1–4.**

| Phase | Deliverable (DONE) | Key commits |
|---|---|---|
| **1** Relocate QCEW acquire | `nfp_ingest/qcew_acquire.py` — **public** `acquire_qcew_levels(start_year=2017, end_year=None)` / `acquire_qcew_size_native(start_year=2017, end_year=None)` (+ `_fetch_qcew_csv`, `_prep_area_raw`, `_size_raw_to_native`). `rebuild_store.py` re-exports the old private names. | `1a47deb`, `865c8e4`, `80dab6a` |
| **2** ukey under-keying fix (§6.1) | `append_to_vintage_store` **and** `compact_partition` now key on the **10-column** ukey (7 base cols + `ownership`, `size_class_type`, `size_class_code`); append's anti-join uses `nulls_equal=True`. | `48d8a0f`, `a2affcf` |
| **3** Calendar-advance callable (§5.0) | `nfp_vintages/calendar.py` — **public** `advance_release_calendar() -> None` (container-safe writes; graceful-403 fallback). `process` rewired to call it; the in-CLI `_build_release_calendar` is **deleted**. | `3d84092`, `3d4ee89` |
| **4** CES capture adapter (§5.1) | `nfp_ingest/capture.py` — `capture_ces_print(as_of, *, store_path=VINTAGE_STORE_PATH) -> CaptureResult`; `_remap_ces_to_store_schema`; `_detect_corrected_levels`; `CaptureResult`/`CorrectedLevel` dataclasses; **`capture_qcew_quarter` STUB** (`raise NotImplementedError` — Phase 6 replaces it). | `f4b3845`, `d38815a`, `5ec59fd` |

---

## Global Constraints

Every task's requirements implicitly include this section. Values copied verbatim from the spec /
the codebase's hard rules.

- **Firewall — do NOT touch:** `nfp-model/*`, `transform_to_panel`, `build_model_data`,
  `model_data.py`, the *logic* of `first_print.py` / `wedge_data.py` / `a5.py`, and the A1/A2/A3
  golden paths. New code may *read* `first_print_changes`/`wedge_first_print_changes` (guardrails)
  but must not modify them.
- **Container storage contract (plans/15) — no code writes under `./data`.** Every persistent
  artifact goes to S3 via an env URI; each unset ⇒ local `data/` fallback. **Every `write_parquet`
  / `mkdir` site MUST thread `storage_options_for(path)` + an `is_remote(path)` mkdir guard + pass
  `str(path)` to polars** (a `UPath` object passed to `pl.read_parquet`/`write_parquet` fails on
  `s3://`). Rebuild scratch (raw downloads, HTTP cache, SAE checkpoint) → `tempfile`; dev scripts
  take their output root as an arg/env, never `data/`.
- **Store-write test safety (HARD):** `.env` sets `NFP_STORE_URI=s3://alt-nfp/store` (canonical
  MinIO) and the root `conftest.py` `load_dotenv()`s **before any `nfp_*` import**, so under pytest
  `VINTAGE_STORE_PATH` resolves to the **canonical MinIO store**. **Every store-writing test MUST
  pin `store_path=tmp_path`** (or monkeypatch the path) — a red-phase test against unpinned
  `VINTAGE_STORE_PATH` would write to/wipe canonical MinIO. Never run a store-writer against real
  MinIO in a test.
- **Deferred imports in command bodies:** every import inside a Typer command body is deferred so
  the `main` callback's `load_dotenv()` runs before `VINTAGE_STORE_PATH` binds (`paths.py` binds at
  import).
- **Polars:** use `nulls_equal=True` (not deprecated `join_nulls=`) in null-aware joins on the
  pinned polars 1.41.2.
- **Lint (ruff E,W,F,I,B,C4,UP, line 100):** no `dict(a=1)` kwargs calls (use `{...}` literals —
  C408); `zip(..., strict=True)` (B905); imports sorted (I001) and at module top, even in extended
  test files (E402); add an import only at the task that first uses it (avoid F401).
- **Commits are scoped per-file** (`git add <exact paths>`), never `-A`/`.` — the working tree
  carries unrelated user WIP. End commit messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **TDD every task:** write the failing test → run it, confirm it fails for the stated reason →
  implement → run, confirm pass → run the full touched-package suite (`-m "not network and not
  slow"`) → ruff clean → commit. The full-suite run is mandatory (it catches stale fixtures the
  narrow test misses).

---

## Interface Contract (canonical — all Phase 5–9 code uses these exact names/signatures)

**Already built (Phases 1–4) — consume, do not redefine:**

```python
# nfp_ingest/capture.py
@dataclass
class CorrectedLevel:
    ref_date: date; industry_code: str; revision: int; benchmark_revision: int
    stored_employment: float; incoming_employment: float
@dataclass
class CaptureResult:
    appended: int; corrected: list[CorrectedLevel]; skipped: int
def capture_ces_print(as_of: date, *, store_path: Path = VINTAGE_STORE_PATH) -> CaptureResult: ...
def capture_qcew_quarter(as_of: date, *, store_path: Path = VINTAGE_STORE_PATH) -> CaptureResult: ...  # Phase 4 STUB → Phase 6 real
def _detect_corrected_levels(new_rows, store_path, source, seasonally_adjusted) -> list[CorrectedLevel]: ...

# nfp_ingest/qcew_acquire.py  (Phase 1)
def acquire_qcew_levels(start_year: int = 2017, end_year: int | None = None) -> pl.DataFrame: ...
def acquire_qcew_size_native(start_year: int = 2017, end_year: int | None = None) -> pl.DataFrame: ...

# nfp_vintages/calendar.py  (Phase 3)
def advance_release_calendar() -> None: ...

# nfp_ingest/vintage_store.py  (existing; ukey now 10-col)
def read_vintage_store(store_path=VINTAGE_STORE_PATH, *, source=None, seasonally_adjusted=None, ...) -> pl.LazyFrame: ...
def append_to_vintage_store(new_rows, store_path=VINTAGE_STORE_PATH) -> int: ...
def compact_partition(store_path, source, seasonally_adjusted) -> None: ...
```

**Defined by this rewrite (keep these signatures so cross-phase consumers stay consistent):**

```python
# nfp_vintages/__main__.py  (Phase 5)
def _run_snapshot(as_of: date, grid_end: date | None = None) -> None: ...          # Task 5.1
def _run_update(as_of: date, *, only: str | None = None,
                refresh_calendar: bool = True, store_path=None) -> None: ...        # Task 5.2/5.3
@app.command() def update(--as-of, --only ces|qcew|indicators, --no-refresh-calendar): ...
@app.command() def snapshot(--as-of, --grid-end): ...   # day-12 enforced both paths (§4a)
@app.command() def status(--as-of, --store): ...        # Phase 7
@app.command() def watch(--source ces|qcew|all, --snapshot): ...  # Phase 8

# nfp_vintages/store_status.py  (Phase 7)
@dataclass(frozen=True)
class PartitionCoverage:
    source: str; seasonally_adjusted: bool; earliest_ref: date | None; latest_ref: date | None
    row_count: int; last_capture: date | None; distinct_vintages: int
@dataclass(frozen=True)
class StoreStatus:
    store_uri: str; is_remote: bool; is_canonical: bool
    per_partition: list[PartitionCoverage] = field(default_factory=list)
    uncaptured: list[str] = field(default_factory=list)
    missing_months: list[str] = field(default_factory=list)
    corrected: list[str] = field(default_factory=list)
def compute_status(store_path=VINTAGE_STORE_PATH, as_of: date | None = None) -> StoreStatus: ...
def format_status(status: StoreStatus) -> str: ...

# nfp_download/release_dates/feed.py  (Phase 8)
@dataclass
class FeedItem: ...                                     # carries pubDate + title/source
def parse_feed(xml: str) -> list[FeedItem]: ...
def fetch_feed(url: str, *, session=None) -> list[FeedItem]: ...

# scripts/bootstrap_store.py  (Phase 9) — one-time rebuild + promote (NOT a CLI command)
```

**Command surface after this plan (spec §4):** `update` / `snapshot` / `status` / `watch` are the
production surface; `download` / `download-indicators` / `process` / `current` / `build` /
`build-rebuild` and the bare-run chain are **retired** (Phase 9) — their reusable bodies move to
`scripts/bootstrap_store.py` (the rebuild lineage) or are already callable (`advance_release_calendar`).

**Cross-phase dependencies — PINNED seams (resolve drift the old plan left; do NOT diverge):**

- **QCEW wiring (5.2 ↔ 6.1 ↔ 6.2).** The old plan contradicts itself (5.2 already wires the QCEW
  call but 6.2 claims to "add the QCEW leg"). Canonical division: **Task 5.2's `_run_update` OWNS
  the QCEW call** (`if only in (None,"qcew"): capture_qcew_quarter(as_of, store_path=store_path)`),
  and 5.2's orchestration test asserts `["calendar","ces","qcew","indicators"]` (against the Phase-4
  stub, monkeypatched in the test). **Task 6.1** replaces the `capture_qcew_quarter` STUB with the
  real single-quarter capture. **Task 6.2** adds ONLY the `--only qcew` behavior + knowable-quarter
  steady-state-no-op (`skipped=1`) **test coverage** through `update` — it MUST NOT re-add the call
  to `_run_update` (5.2 already has it; re-adding double-wires it).
- **Phase 8 `watch` → Phase 5 helpers.** `watch` calls the plain helpers, never the Typer commands:
  `_run_update(as_of=<date>, only=<src>)` (pass a **`date`**, not `.isoformat()`; the canonical
  `_run_update(as_of: date, ...)` parses nothing) and, under `--snapshot`,
  `_run_snapshot(as_of=date(<refmonth>.year, <refmonth>.month, 12))` (the **day-12 anchor of the
  captured ref-month**, never the raw `pubDate` — `_run_snapshot` rejects non-12th, §4a). The old
  plan's `no_refresh_calendar=` kwarg and `as_of=pub.isoformat()` are **drift** — reconcile to the
  contract signatures.
- **Phase 8 `watch` → Phase 7 `compute_status`.** "Is this new?" is decided from the **store** via
  `compute_status(as_of=pub)` + `StoreStatus.uncaptured` (spec principle 3 — the store is the source
  of truth; the RSS feed only says "a release is out now"). Watch lets `compute_status` run for real
  against `tmp_path` in tests while monkeypatching the feed + `_run_update`/`_run_snapshot`.
- **Phase 9 bootstrap** consumes `acquire_qcew_levels` / `acquire_qcew_size_native` (Phase 1) +
  `advance_release_calendar` (Phase 3) + `write_rebuild_panel`/`compose_rebuild_panel`/
  `write_rebuild_store` (`rebuild_store.py`, unchanged) + the `_t8_promote.py` cutover. **Phase 9
  plans against this CONTRACT's command surface, NOT current `__main__.py`** (which has no
  `update`/`status`/`watch` yet — those are authored in this rewrite, not executed). Retiring
  `process` is just deleting the command (its `_build_release_calendar` helper is already gone,
  Phase 3).

**Stale-reference warning (the whole reason for this rewrite):** the original plan and the spec
predate plans/15 and Phases 1–4. **Every `__main__.py:NNN` / `rebuild_store.py:NNN` line-ref in the
old plan is suspect** — verify against current source before using it. The original plan's
reproduced impl code repeatedly used the **pre-plans/15 write pattern** (`df.write_parquet(PATH)`
without `str()`/`storage_options`/`is_remote`) — current source wins, container-safety is mandatory.

---

## Phase 5 — `update`/`snapshot` commands, `_run_*` helpers, day-12 fix, guardrail suite

**Spec:** §4a, §5, §5.0, §5.3, §6.2, §7. **Consumes (all DONE):** `advance_release_calendar`
(Phase 3), `capture_ces_print` + the `capture_qcew_quarter` STUB + `CaptureResult` (Phase 4),
`append_to_vintage_store`/`compact_partition` (10-col ukey, Phase 2). **Provides for Phase 8:** the
plain `_run_update`/`_run_snapshot` helpers `watch` calls (a Typer command is never called directly).

All command-body code lives in `packages/nfp-vintages/src/nfp_vintages/__main__.py`; tests in
`packages/nfp-vintages/src/nfp_vintages/tests/`. Every import inside a command body is **deferred**
(the `main` callback runs `load_dotenv()` before `VINTAGE_STORE_PATH` resolves at `paths.py` import).

> **Note — default `update` path is not end-to-end runnable until Phase 6.1.** `_run_update`'s QCEW
> leg calls `capture_qcew_quarter`, which is the Phase-4 `NotImplementedError` STUB until Phase 6.1
> replaces it. Every Phase-5 test monkeypatches the captures, so the suite is green; a *real* `update
> --as-of T` (no `--only`) raises `NotImplementedError` on the QCEW leg until Phase 6.1. `update
> --only ces` and `--only indicators` are fully runnable after Phase 5.

> **Task-ordering fix (vs the original plan):** the guardrail tests import shared fixtures
> (`make_ces_rows`, `make_benchmark_double_row`, `make_shutdown_sentinel_row` from
> `nfp_vintages/tests/_fixtures.py`). The original plan created those fixtures **last** (5.7), so the
> earlier red phases couldn't run. This rewrite builds the **fixtures FIRST** (Task 5.3) before any
> task that imports them (5.4 self-heal, 5.5–5.8 guardrails).

---

### Task 5.1: `_run_snapshot` helper + enforce day-12 on `snapshot --as-of` in BOTH paths (§4a)

The current `snapshot` command (`__main__.py:184-217`) validates `as_of.day == 12` **only when
`--grid-end` is None** (`:201-204`); the grid loop seeds `date(y, m, 12)` ignoring `as_of.day`
(`:210-213`), so `snapshot --as-of 2026-03-05 --grid-end 2026-06-12` silently snapshots `2026-03-12`
— a date *later* than the requested cutoff. Extract a plain `_run_snapshot` that validates **both**
paths; make the command a thin wrapper.

**Files:**
- Modify: `packages/nfp-vintages/src/nfp_vintages/__main__.py` (the `snapshot` command at `:184-217`; add `from datetime import date` at module top)
- Test: `packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py` (new)

**Interfaces — Produces:** `_run_snapshot(as_of: date, grid_end: date | None = None) -> None`.

- [ ] **Step 1: Write the failing test** — create `test_cli_update.py`:

```python
"""CLI tests for the production surface (snapshot day-12, update orchestration).

Phase 5 of specs/cli_production_workflow.md. Uses Typer's CliRunner with
deferred-import command bodies monkeypatched so no network/store/key is touched.
"""

from __future__ import annotations

from datetime import date

import pytest
from typer.testing import CliRunner

from nfp_vintages.__main__ import app

runner = CliRunner()


class TestSnapshotDay12:
    def test_grid_mode_rejects_non_12th_as_of(self):
        # Today this silently snapshots 2026-03-12; it must be rejected.
        result = runner.invoke(
            app, ["snapshot", "--as-of", "2026-03-05", "--grid-end", "2026-06-12"]
        )
        assert result.exit_code != 0
        assert "12th" in result.output or "day-12" in result.output

    def test_single_mode_rejects_non_12th_as_of(self):
        result = runner.invoke(app, ["snapshot", "--as-of", "2026-03-05"])
        assert result.exit_code != 0
```

- [ ] **Step 2: Run the test, verify it fails** — `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py::TestSnapshotDay12 -q --no-cov`. Expected: FAIL — `test_grid_mode_rejects_non_12th_as_of` returns exit_code 0 (the grid path accepts the non-12th `--as-of` today).

- [ ] **Step 3: Implement** — add `from datetime import date` to the module imports (top of file). Add `_run_snapshot` above the `snapshot` command and replace the command body:

```python
def _run_snapshot(as_of: date, grid_end: date | None = None) -> None:
    """Write hash-pinned ModelData snapshot(s); plain helper (no Typer types)."""
    from nfp_ingest.snapshots import snapshot_model_data

    if as_of.day != 12:
        raise ValueError("--as-of must fall on the 12th (day-12 convention)")

    if grid_end is None:
        dates = [as_of]
    else:
        dates = []
        y, m = as_of.year, as_of.month
        while date(y, m, 12) <= grid_end:
            dates.append(date(y, m, 12))
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)

    for d in dates:
        path, digest = snapshot_model_data(d)
        print(f"  {d}: {path} (hash {digest[:12]})")


@app.command()
def snapshot(
    as_of: str = typer.Option(..., "--as-of", help="Knowledge cutoff, YYYY-MM-DD (day-12)."),
    grid_end: str | None = typer.Option(
        None, "--grid-end", help="If set, snapshot every month's 12th from --as-of through here."
    ),
) -> None:
    """Write hash-pinned ModelData snapshot(s) for the given as-of date(s)."""
    from datetime import date as _date

    end = _date.fromisoformat(grid_end) if grid_end is not None else None
    try:
        _run_snapshot(_date.fromisoformat(as_of), end)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--as-of") from exc
```

- [ ] **Step 4: Run, verify pass** — `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py::TestSnapshotDay12 -q --no-cov` (PASS). Then `uv run pytest packages/nfp-vintages -q --no-cov -m "not network and not slow"` (full package suite — catches the snapshot-command refactor) and `uv run ruff check packages/nfp-vintages` (clean).

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/__main__.py \
        packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py
git commit -m "fix(cli): enforce day-12 on snapshot --as-of in both single and grid paths (§4a)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5.2: The `update` command + `_run_update` plain helper (§5, §5.0, §5.3)

`_run_update` orchestrates §5.0 (calendar advance, via the Phase-3 `advance_release_calendar`) →
§5.1 (CES capture, Phase-4 `capture_ces_print`) → §5.2 (QCEW, Phase-4 stub → Phase-6 real) → §5.3
(indicators refresh, existing `download_indicators` — **not** an append). The `update` command is a
thin wrapper. **The QCEW call lives HERE** (per the CONTRACT seam); Phase 6.1 only swaps the stub
for the real impl, Phase 6.2 only adds `--only qcew` test coverage.

**Files:**
- Modify: `packages/nfp-vintages/src/nfp_vintages/__main__.py`
- Test: `packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py`

**Interfaces — Produces:** `_run_update(as_of: date, *, only: str | None = None, refresh_calendar: bool = True, store_path=None) -> None`; the `update` Typer command.

- [ ] **Step 1: Write the failing test** — append to `test_cli_update.py`:

```python
class TestUpdateOrchestration:
    def test_update_runs_calendar_then_ces_then_qcew_then_indicators(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "nfp_vintages.calendar.advance_release_calendar",
            lambda: calls.append("calendar"),
        )

        class _Res:
            appended, corrected, skipped = 3, [], 0

        monkeypatch.setattr(
            "nfp_ingest.capture.capture_ces_print",
            lambda as_of, *, store_path=None: calls.append("ces") or _Res(),
        )
        monkeypatch.setattr(
            "nfp_ingest.capture.capture_qcew_quarter",
            lambda as_of, *, store_path=None: calls.append("qcew") or _Res(),
        )
        monkeypatch.setattr(
            "nfp_ingest.indicators.download_indicators",
            lambda: calls.append("indicators") or {},
        )

        result = runner.invoke(app, ["update", "--as-of", "2026-06-12"])
        assert result.exit_code == 0, result.output
        assert calls == ["calendar", "ces", "qcew", "indicators"]

    def test_only_ces_skips_qcew_and_indicators(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "nfp_vintages.calendar.advance_release_calendar", lambda: calls.append("calendar")
        )

        class _Res:
            appended, corrected, skipped = 1, [], 0

        monkeypatch.setattr(
            "nfp_ingest.capture.capture_ces_print",
            lambda as_of, *, store_path=None: calls.append("ces") or _Res(),
        )
        monkeypatch.setattr(
            "nfp_ingest.indicators.download_indicators",
            lambda: calls.append("indicators"),
        )
        result = runner.invoke(app, ["update", "--as-of", "2026-06-12", "--only", "ces"])
        assert result.exit_code == 0, result.output
        assert calls == ["calendar", "ces"]

    def test_no_refresh_calendar_skips_scrape(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "nfp_vintages.calendar.advance_release_calendar", lambda: calls.append("calendar")
        )

        class _Res:
            appended, corrected, skipped = 0, [], 1

        monkeypatch.setattr(
            "nfp_ingest.capture.capture_ces_print",
            lambda as_of, *, store_path=None: _Res(),
        )
        monkeypatch.setattr(
            "nfp_ingest.capture.capture_qcew_quarter",
            lambda as_of, *, store_path=None: _Res(),
        )
        monkeypatch.setattr("nfp_ingest.indicators.download_indicators", lambda: {})
        result = runner.invoke(
            app, ["update", "--as-of", "2026-06-12", "--no-refresh-calendar"]
        )
        assert result.exit_code == 0, result.output
        assert "calendar" not in calls

    def test_invalid_only_rejected(self):
        result = runner.invoke(app, ["update", "--as-of", "2026-06-12", "--only", "bogus"])
        assert result.exit_code != 0
```

- [ ] **Step 2: Run the test, verify it fails** — `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py::TestUpdateOrchestration -q --no-cov`. Expected: FAIL — `No such command 'update'`.

- [ ] **Step 3: Implement** — add `_run_update` and the `update` command. `--only` accepts `ces|qcew|indicators` (None ⇒ all). Each capture already appends+compacts its own touched partitions internally (Phase 4/6); `_run_update` prints the `CaptureResult` and surfaces corrected-level warnings.

```python
def _run_update(
    as_of: date,
    *,
    only: str | None = None,
    refresh_calendar: bool = True,
    store_path=None,
) -> None:
    """Capture month-T prints into the store; plain helper (no Typer types)."""
    from nfp_lookups.paths import VINTAGE_STORE_PATH

    store_path = store_path if store_path is not None else VINTAGE_STORE_PATH

    if refresh_calendar:
        from nfp_vintages.calendar import advance_release_calendar

        advance_release_calendar()

    if only in (None, "ces"):
        from nfp_ingest.capture import capture_ces_print

        res = capture_ces_print(as_of, store_path=store_path)
        print(f"  CES: appended {res.appended}, skipped {res.skipped}")
        for c in res.corrected:
            print(f"  CORRECTED-LEVEL ces {c.ref_date} {c.industry_code} "
                  f"rev{c.revision}/bmr{c.benchmark_revision}: "
                  f"{c.stored_employment} -> {c.incoming_employment}")

    if only in (None, "qcew"):
        from nfp_ingest.capture import capture_qcew_quarter

        res = capture_qcew_quarter(as_of, store_path=store_path)
        print(f"  QCEW: appended {res.appended}, skipped {res.skipped}")
        for c in res.corrected:
            print(f"  CORRECTED-LEVEL qcew {c.ref_date} {c.industry_code} "
                  f"rev{c.revision}/bmr{c.benchmark_revision}: "
                  f"{c.stored_employment} -> {c.incoming_employment}")

    if only in (None, "indicators"):
        from nfp_ingest.indicators import download_indicators

        results = download_indicators()
        total = sum(results.values()) if results else 0
        print(f"  Indicators: {total} rows across {len(results or {})} series")


@app.command()
def update(
    as_of: str = typer.Option(..., "--as-of", help="Knowledge cutoff, YYYY-MM-DD."),
    only: str | None = typer.Option(
        None, "--only", help="Limit to one source: ces | qcew | indicators."
    ),
    no_refresh_calendar: bool = typer.Option(
        False, "--no-refresh-calendar", help="Skip the release-calendar scrape (assume current)."
    ),
) -> None:
    """Advance the calendar, capture month-T prints, and append them to the store."""
    from datetime import date as _date

    if only is not None and only not in ("ces", "qcew", "indicators"):
        raise typer.BadParameter("must be ces, qcew, or indicators", param_hint="--only")
    _run_update(
        _date.fromisoformat(as_of), only=only, refresh_calendar=not no_refresh_calendar
    )
```

- [ ] **Step 4: Run, verify pass** — `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py -q --no-cov` (all `TestUpdateOrchestration` + `TestSnapshotDay12`). Then `uv run pytest packages/nfp-vintages -q --no-cov -m "not network and not slow"` and `uv run ruff check packages/nfp-vintages`.

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/__main__.py \
        packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py
git commit -m "feat(cli): add update command (calendar advance + CES/QCEW capture + indicators)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---
### Phase 5 — guardrail arm (Tasks 5.3–5.8)

This arm builds the `update` correctness guardrail on top of `_run_update` (authored in Task
5.2). It is **fixtures-first**: Task 5.3 lands `_fixtures.py`, then the guardrail tests import it.

**Pre-verified facts (do not re-derive — trust these, the old plan's names were wrong):**

- `first_print_changes` lives at `nfp_ingest/first_print.py:53`. Signature:
  `first_print_changes(*, store_path=VINTAGE_STORE_PATH, geographic_type="national",
  geographic_code="00", industry_type="total", industry_code="00")`. Returns columns
  **`ref_date, first_print_growth, first_print_change_k, vintage_date`** (NOT `period`/`change_k`
  — old plan was wrong). `ref_date` is the month truncated to day-1. It **drops the `-1.0`
  sentinel** via `.filter(pl.col("employment") > 0)` (`first_print.py:84`), so `'05'`-only
  fixtures must also carry a prior-month `rev1/bmr0` partner or `first_print_change_k` is null.
- `wedge_first_print_changes` lives at **`nfp_ingest/wedge_data.py:24`** (NOT `nfp_vintages`).
  Signature: `wedge_first_print_changes(*, store_path=VINTAGE_STORE_PATH)` — **no `industry_code=`
  kwarg** (the task prompt's assumption is wrong; see deviations). It inner-joins
  `first_print_changes(industry_type="total", industry_code="00")` against
  `(industry_code="05")` on `ref_date`, then `drop_nulls`, and **raises `ValueError`** if the two
  legs' `vintage_date` differ by more than 15 days. Returns
  **`ref_date, chg00, chg05, wedge_change_k`**. ⇒ the wedge guardrail (5.6) only tests anything if
  the fixture carries BOTH a `'00'/total` leg and a `'05'/private` leg, co-released.
- `VINTAGE_STORE_SCHEMA` (`nfp_lookups/schemas.py:128`) is the **15-key** dict, column order:
  `geographic_type, geographic_code, ownership, industry_type, industry_code, ref_date,
  vintage_date, revision, benchmark_revision, employment, size_class_type, size_class_code,
  source, seasonally_adjusted`. Types: `revision`/`benchmark_revision` are **`pl.UInt8`**,
  `employment` is `pl.Float64`, dates are `pl.Date`, `seasonally_adjusted` is `pl.Boolean`.
- `append_to_vintage_store(new_rows, store_path)` and `compact_partition(store_path, source,
  seasonally_adjusted)` (`vintage_store.py:678`, `:762`) dedup on the **10-col ukey**
  (`ref_date, industry_type, industry_code, geographic_type, geographic_code, revision,
  benchmark_revision, ownership, size_class_type, size_class_code`) — already merged. `append`
  keeps the **first-written** row per ukey; `compact` keeps **`MIN(vintage_date)`** per ukey.
  On-disk layout is `store/source=<src>/seasonally_adjusted=<true|false>/*.parquet`.

**Store-write test safety (HARD):** under pytest, `VINTAGE_STORE_PATH` resolves to the **canonical
MinIO store**. Every store-writing test in this arm pins `store_path=tmp_path` and every fixture
threads `store` explicitly. No test ever writes the default path.

**Lint:** dict-literals not `dict(...)` (C408); `zip(..., strict=True)` (B905); imports sorted at
module top (I001/E402); no unused imports (F401); ≤100 cols.

---

### Task 5.3: Guardrail fixtures — `_fixtures.py` (`make_ces_rows`, benchmark double-row, shutdown sentinel)

The guardrail tests (5.4–5.8) all import synthetic `VINTAGE_STORE_SCHEMA` rows from a shared
`_fixtures.py`. This task lands the module with the three named builders the spec §7 requires plus
a trivial shape test (so the task has an honest red→green cycle). The fixtures are **synthetic
store rows**, never capture output — no store, no network.

`make_ces_rows` carries optional kwargs (`employment`, `revision`, `benchmark_revision`,
`industry_code`, `seasonally_adjusted`) with defaults; 5.5/5.6/5.8 pass them. `make_first_print_window`
(the two-leg first-print builder) lands later in Task 5.6 where it is first consumed.

**Files:**
- Create: `packages/nfp-vintages/src/nfp_vintages/tests/_fixtures.py`
- Test: `packages/nfp-vintages/src/nfp_vintages/tests/test_guardrail_fixtures.py`

- [ ] **Step 1: Write the failing test** — `_fixtures` does not exist yet, so the import fails.

```python
"""Shape checks for the synthetic guardrail fixtures (§7 required fixtures)."""

from __future__ import annotations

import polars as pl
from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA

from nfp_vintages.tests._fixtures import (
    make_benchmark_double_row,
    make_ces_rows,
    make_shutdown_sentinel_row,
)

_COLS = list(VINTAGE_STORE_SCHEMA.keys())


def test_make_ces_rows_one_schema_row():
    df = make_ces_rows(ref_month="2026-01-12", vintage="2026-02-06")
    assert df.height == 1
    assert df.columns == _COLS
    assert df.schema == VINTAGE_STORE_SCHEMA
    # the headline default targets private '05' (ownership=private)
    assert df["industry_code"].item() == "05"
    assert df["ownership"].item() == "private"


def test_make_benchmark_double_row_two_coherent_tracks():
    df = make_benchmark_double_row(ref_month="2025-12-12")
    assert df.height == 2
    assert df.columns == _COLS
    keys = set(zip(df["revision"].to_list(), df["benchmark_revision"].to_list(), strict=True))
    assert keys == {(1, 0), (2, 1)}
    # both rows are the same ref_date
    assert df["ref_date"].n_unique() == 1


def test_make_shutdown_sentinel_row_literal_minus_one():
    df = make_shutdown_sentinel_row(ref_month="2025-10-12")
    assert df.height == 1
    assert df.columns == _COLS
    assert df["employment"].item() == -1.0
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_guardrail_fixtures.py -q --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'nfp_vintages.tests._fixtures'` (the module
is created in Step 3). This is the honest red: the deliverable doesn't exist yet.

- [ ] **Step 3: Implement** — create `_fixtures.py`. `make_ces_rows` builds one
  `VINTAGE_STORE_SCHEMA`-conforming row; the two danger-edge builders compose it. The
  `pl.DataFrame([...], schema=VINTAGE_STORE_SCHEMA)` construction pins column order and dtypes
  (UInt8 revisions, Float64 employment) so downstream `append_to_vintage_store` accepts it without
  a cast.

```python
"""Synthetic VINTAGE_STORE_SCHEMA rows for the update guardrail tests.

These are hand-built store rows (no store I/O, no network) used to exercise the
dangerous edges of append/compact and the first-print consumers. The capture path
itself never produces a -1.0 sentinel; ``make_shutdown_sentinel_row`` fabricates
one to test that the overlap diagnostic excludes it (§7).
"""

from __future__ import annotations

from datetime import date

import polars as pl
from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA


def make_ces_rows(
    *,
    ref_month: str,
    vintage: str,
    employment: float = 150_000.0,
    industry_code: str = "05",
    revision: int = 0,
    benchmark_revision: int = 0,
    seasonally_adjusted: bool = True,
) -> pl.DataFrame:
    """One CES headline row in the rebuilt-store schema.

    Defaults target the modeled private aggregate (``industry_code='05'`` ⇒
    ``ownership='private'``). Pass ``industry_code='00'`` for the total leg
    (⇒ ``ownership='total'``).
    """
    ownership = "private" if industry_code == "05" else "total"
    row = {
        "geographic_type": "national",
        "geographic_code": "00",
        "ownership": ownership,
        "industry_type": "total",
        "industry_code": industry_code,
        "ref_date": date.fromisoformat(ref_month),
        "vintage_date": date.fromisoformat(vintage),
        "revision": revision,
        "benchmark_revision": benchmark_revision,
        "employment": employment,
        "size_class_type": None,
        "size_class_code": None,
        "source": "ces",
        "seasonally_adjusted": seasonally_adjusted,
    }
    cols = list(VINTAGE_STORE_SCHEMA.keys())
    return pl.DataFrame([{c: row[c] for c in cols}], schema=VINTAGE_STORE_SCHEMA)


def make_benchmark_double_row(*, ref_month: str) -> pl.DataFrame:
    """One ref_date co-published as BOTH (rev1,bmr0) and (rev2,bmr1) on a benchmark.

    The February benchmark restamps a month under the new benchmark revision while
    the pre-benchmark track still exists. Both ukeys are distinct (differ in
    benchmark_revision and revision), so append/compact must keep both rows.
    """
    a = make_ces_rows(
        ref_month=ref_month, vintage="2026-02-06",
        revision=1, benchmark_revision=0, employment=149_500.0,
    )
    b = make_ces_rows(
        ref_month=ref_month, vintage="2026-02-06",
        revision=2, benchmark_revision=1, employment=149_900.0,
    )
    return pl.concat([a, b])


def make_shutdown_sentinel_row(*, ref_month: str) -> pl.DataFrame:
    """The literal ``employment = -1.0`` 'no print' sentinel for a shutdown-skipped slot.

    This is the *value* the rebuilt store writes for a skipped release slot (e.g.
    Oct-2025 rev0); ``first_print_changes`` drops it via ``employment > 0``
    (``first_print.py:84``). Distinct from the *date* quirk
    ``CES_OCT_2025_RELEASED_WITH_NOV_REF``.
    """
    return make_ces_rows(
        ref_month=ref_month, vintage="2025-11-12",
        revision=0, benchmark_revision=0, employment=-1.0,
    )
```

- [ ] **Step 4: Run, verify pass**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_guardrail_fixtures.py -q --no-cov`
Expected: PASS (3 tests). Then the full touched-package suite and lint:

```bash
uv run pytest packages/nfp-vintages -m "not network and not slow" --no-cov -q
uv run ruff check packages/nfp-vintages
```

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/tests/_fixtures.py \
        packages/nfp-vintages/src/nfp_vintages/tests/test_guardrail_fixtures.py
git commit -m "test(cli): synthetic guardrail fixtures (ces rows, benchmark double-row, shutdown sentinel)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5.4: `_run_update` self-healing compaction of fragmented partitions (§6.2)

If a prior `update` crashed between `append_to_vintage_store` and `compact_partition`, a partition
is left with >1 fragment (order-sensitive, read-amplifying). Per §6.2, `_run_update` must compact
**any** `(source, seasonally_adjusted)` partition holding more than one parquet file — regardless
of whether this run appended — because it is cheap and idempotent, self-healing a crash on the next
run.

**Assumptions about Task 5.2's `_run_update` (authored upstream — do not redefine):** signature is
`_run_update(as_of, *, only=None, refresh_calendar=True, store_path=None)`; when `store_path is
None` it resolves the effective store path from `VINTAGE_STORE_PATH` internally (the heal pass must
glob the **resolved** path, never `None`); and capture functions are imported **deferred inside the
body** (per the global constraint) so a `monkeypatch.setattr("nfp_ingest.capture.capture_ces_print",
...)` lands. The heal pass is appended as the final step of `_run_update`.

**Files:**
- Modify: `packages/nfp-vintages/src/nfp_vintages/__main__.py`
- Test: `packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py`

- [ ] **Step 1: Write the failing test** — two disjoint appends leave two fragment files in the
  `(ces, true)` partition; with capture stubbed to append nothing, `_run_update` must still collapse
  them to one. This is a **real behavioral red** (the heal pass doesn't exist yet).

```python
"""§6.2: _run_update self-heals partitions a crashed prior run left fragmented."""

from __future__ import annotations

from datetime import date

from nfp_ingest.vintage_store import append_to_vintage_store

from nfp_vintages.tests._fixtures import make_ces_rows


class TestUpdateSelfHealingCompaction:
    def test_update_compacts_pre_existing_fragments(self, tmp_path, monkeypatch):
        store = tmp_path / "store"
        # Two disjoint appends → two fragment files in the same (ces, true) partition.
        append_to_vintage_store(
            make_ces_rows(ref_month="2026-01-12", vintage="2026-02-06"), store
        )
        append_to_vintage_store(
            make_ces_rows(ref_month="2026-02-12", vintage="2026-03-06"), store
        )
        part = store / "source=ces" / "seasonally_adjusted=true"
        assert len(list(part.glob("*.parquet"))) == 2

        # Stub everything except the heal pass; capture appends nothing.
        monkeypatch.setattr(
            "nfp_vintages.calendar.advance_release_calendar", lambda: None
        )

        class _Res:
            appended, corrected, skipped = 0, [], 1

        monkeypatch.setattr(
            "nfp_ingest.capture.capture_ces_print", lambda a, *, store_path=None: _Res()
        )
        monkeypatch.setattr(
            "nfp_ingest.capture.capture_qcew_quarter",
            lambda a, *, store_path=None: _Res(),
        )
        # only=None runs the indicators leg too; stub it so the heal pass — not a
        # real download — is what Step 2's red exercises. (Reconcile the exact symbol
        # against Task 5.2's _run_update indicators leg; the old plan used
        # `nfp_ingest.indicators.download_indicators`. If 5.2 imports it under a
        # different name/module, patch that one instead.)
        monkeypatch.setattr("nfp_ingest.indicators.download_indicators", lambda: {})

        from nfp_vintages.__main__ import _run_update

        _run_update(date(2026, 6, 12), store_path=store)
        assert len(list(part.glob("*.parquet"))) == 1
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py::TestUpdateSelfHealingCompaction -q --no-cov`
Expected: FAIL — assertion fires with **two** parquet files still present in the partition (no heal
pass yet). (If `_run_update`/its capture imports aren't deferred, the monkeypatch won't land and you
will instead see a real capture attempt — that is a Task 5.2 contract violation to flag, not a 5.4
bug.)

- [ ] **Step 3: Implement** — append a heal pass at the end of `_run_update`. Resolve the effective
  store path (the body already does this for the capture calls; reuse that local, e.g. `store`),
  then compact every partition holding >1 parquet file. Keep the `is_remote` **local-store guard**:
  the directory glob is a local-filesystem operation; object-store self-heal is a documented
  follow-on (remote partitions are enumerated via `UPath` + `storage_options_for`, and
  `compact_partition` already handles remote deletes — wired in a later iteration).

```python
    # --- self-healing compaction (§6.2) -------------------------------------
    # A crash between append_to_vintage_store and compact_partition leaves a
    # partition with >1 fragment. Compact any such partition on the next run —
    # cheap and idempotent (compact is a no-op on a single-file partition).
    # FOLLOW-ON: remote (s3://) self-heal — enumerate partitions via UPath +
    # storage_options_for(store); compact_partition already deletes remote
    # fragments. Guarded out here so the local test stays hermetic.
    from nfp_ingest.vintage_store import compact_partition
    from nfp_lookups.paths import is_remote

    if not is_remote(store):
        for source_dir in sorted(store.glob("source=*")):
            source = source_dir.name.split("=", 1)[1]
            for sa_dir in sorted(source_dir.glob("seasonally_adjusted=*")):
                if len(list(sa_dir.glob("*.parquet"))) > 1:
                    sa = sa_dir.name.split("=", 1)[1] == "true"
                    compact_partition(store, source, sa)
```

(`store` is the effective store path local already bound earlier in `_run_update`. If Task 5.2
named it differently, rename to match; do **not** glob the bare `store_path` parameter, which may be
`None`.)

- [ ] **Step 4: Run, verify pass**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py -q --no-cov`
Expected: PASS. Then the full touched-package suite and lint:

```bash
uv run pytest packages/nfp-vintages -m "not network and not slow" --no-cov -q
uv run ruff check packages/nfp-vintages
```

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/__main__.py \
        packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py
git commit -m "feat(cli): self-heal fragmented partitions in update (compact any >1-file partition)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5.5: Guardrail — idempotence (§7.1)

A month-T capture must be idempotent: appending the same rows twice adds 0 the second time, a
second compact is a no-op, and the `(ukey → employment)` relation is unchanged. The §7.1 landmine —
a same-ukey/**different-vintage** row — exercises the append (first-written) vs compact
(min-vintage) tie-break: compact keeps `MIN(vintage_date)`, so the earlier real-time level wins.

This is a **characterization** test — the 10-col ukey and the min-vintage compact rule are already
merged, so the property already holds. There is no new artifact and the import (`_fixtures` from
5.3) already exists, so the honest red is a **deliberately-wrong assertion** that the executor flips
once they have read the true value. Label it as characterization in the test docstring.

**Files:**
- Test: `packages/nfp-vintages/src/nfp_vintages/tests/test_update_guardrail.py`

- [ ] **Step 1: Write the failing test** — write the second assert as `assert added == 1` (the
  deliberate-wrong value) for the red phase; Step 4 flips it to `== 0`.

```python
"""§7 guardrail: month-T capture is idempotent and tie-breaks via min(vintage_date).

Characterization tests — the 10-col ukey + min-vintage compact already give these
properties. The first run pins them so a future change to the dedup rules trips here.
"""

from __future__ import annotations

from nfp_ingest.vintage_store import (
    append_to_vintage_store,
    compact_partition,
    read_vintage_store,
)

from nfp_vintages.tests._fixtures import make_ces_rows

_UKEY = [
    "ref_date", "industry_type", "industry_code", "geographic_type",
    "geographic_code", "revision", "benchmark_revision", "ownership",
    "size_class_type", "size_class_code",
]


def _relation(store) -> dict:
    """Map the 10-col dedup ukey -> employment for the (ces, true) partition."""
    df = read_vintage_store(store, source="ces", seasonally_adjusted=True).collect()
    return {
        tuple(r[c] for c in _UKEY): r["employment"]
        for r in df.iter_rows(named=True)
    }


class TestIdempotence:
    def test_capture_append_compact_twice_same_relation(self, tmp_path):
        store = tmp_path / "store"
        rows = make_ces_rows(ref_month="2026-01-12", vintage="2026-02-06")

        append_to_vintage_store(rows, store)
        compact_partition(store, "ces", True)
        first = _relation(store)

        # Second run: identical rows must add 0; compact must be a no-op.
        added = append_to_vintage_store(rows, store)
        compact_partition(store, "ces", True)
        second = _relation(store)

        assert added == 1  # DELIBERATE-WRONG (red); flip to 0 in Step 4
        assert first == second

    def test_same_ukey_later_vintage_keeps_min_vintage_level(self, tmp_path):
        import polars as pl

        store = tmp_path / "store"
        early = make_ces_rows(
            ref_month="2026-01-12", vintage="2026-02-06", employment=150_000.0
        )
        late = make_ces_rows(
            ref_month="2026-01-12", vintage="2026-03-06", employment=151_000.0
        )
        # Append BOTH in one batch: append does no within-batch ukey dedup, so both
        # land in the partition and compact is what actually chooses the survivor
        # (the §7.1 first-written-vs-min-vintage flip — a two-append sequence would
        # instead have append's anti-join drop `late` before compact ever saw it).
        append_to_vintage_store(pl.concat([early, late]), store)
        pre = read_vintage_store(store, source="ces", seasonally_adjusted=True).collect()
        assert pre.height == 2  # both rows present pre-compact (append keeps both)
        compact_partition(store, "ces", True)
        rel = _relation(store)
        # compact keeps MIN(vintage_date) per ukey → the early real-time level wins
        assert set(rel.values()) == {150_000.0}
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_update_guardrail.py::TestIdempotence -q --no-cov`
Expected: FAIL — `test_capture_append_compact_twice_same_relation` asserts `added == 1` but the
re-append is fully anti-joined out, so the real value is `0`. (`test_same_ukey_later_vintage_keeps_min_vintage_level`
should already pass — confirm it does; if it fails, the compact min-vintage rule regressed and that
is a real bug to surface.)

- [ ] **Step 3: Implement** — no production code (characterization). Flip the deliberate-wrong
  assert.

```python
        assert added == 0  # re-append of identical rows is fully deduped
```

- [ ] **Step 4: Run, verify pass**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_update_guardrail.py::TestIdempotence -q --no-cov`
Expected: PASS (2 tests). Then the full touched-package suite and lint:

```bash
uv run pytest packages/nfp-vintages -m "not network and not slow" --no-cov -q
uv run ruff check packages/nfp-vintages
```

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/tests/test_update_guardrail.py
git commit -m "test(cli): guardrail idempotence (re-append adds 0; compact keeps min-vintage level)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5.6: Guardrail — first-print-unchanged (§7.3)

A capture must not perturb `first_print_changes()` (`first_print.py:53`) **or**
`wedge_first_print_changes()` (`wedge_data.py:24`) for already-present months. §7.3 is a hard
equality pin (not the "flag don't assert" of the overlap test).

**Vacuous-pin landmine (do not skip):** `wedge_first_print_changes` inner-joins
`first_print_changes(industry_code="00")` against `("05")` on `ref_date`, then `drop_nulls`. If the
fixture has only `'05'` rows the wedge frame is **empty** and the before/after `.all()` passes
vacuously — the guardrail tests nothing. So `make_first_print_window` (added here) must emit **both**
legs: a `'05'/private` track and a `'00'/total` track, **co-released** (same vintage stamps, within
the 15-day window) across two consecutive months so each month has a prior-month `rev1/bmr0`
partner. The test asserts `wedge_first_print_changes(store_path=store).height >= 1` **before**
trusting the pin — a discriminating check against the vacuous-empty trap.

**Documented limitation:** this proves a capture is non-destructive for existing months; it
**cannot** catch a dropped same-revision correction (§6.3) — that is the runtime `CORRECTED-LEVEL`
warning's job (Phase 4).

**Files:**
- Modify: `packages/nfp-vintages/src/nfp_vintages/tests/_fixtures.py` (add `make_first_print_window`)
- Test: `packages/nfp-vintages/src/nfp_vintages/tests/test_update_guardrail.py`

- [ ] **Step 1: Write the failing test** — `make_first_print_window` is not in `_fixtures` yet, so
  the import fails (honest red).

```python
class TestFirstPrintUnchanged:
    def test_capture_does_not_move_existing_first_prints(self, tmp_path):
        from nfp_ingest.first_print import first_print_changes
        from nfp_ingest.wedge_data import wedge_first_print_changes

        from nfp_vintages.tests._fixtures import make_first_print_window

        store = tmp_path / "store"
        make_first_print_window(store)  # two months × {00 total, 05 private}, co-released

        # Discriminating guard against a vacuous (empty-frame) pin: the wedge must
        # actually resolve at least one ref_date, else the .all() below is empty.
        wedge_before = wedge_first_print_changes(store_path=store)
        assert wedge_before.height >= 1
        fp05_before = first_print_changes(store_path=store, industry_code="05")
        assert fp05_before.filter(
            pl.col("first_print_change_k").is_not_null()
        ).height >= 1

        # A NEW, later month's capture must not move earlier months' first prints.
        append_to_vintage_store(
            make_ces_rows(
                ref_month="2026-03-12", vintage="2026-04-03",
                revision=0, employment=152_000.0, industry_code="05",
            ),
            store,
        )
        append_to_vintage_store(
            make_ces_rows(
                ref_month="2026-03-12", vintage="2026-04-03",
                revision=0, employment=303_000.0, industry_code="00",
            ),
            store,
        )
        compact_partition(store, "ces", True)

        fp05_after = first_print_changes(store_path=store, industry_code="05")
        wedge_after = wedge_first_print_changes(store_path=store)

        common = fp05_before.join(fp05_after, on="ref_date", suffix="_after", how="inner")
        assert (
            common["first_print_change_k"] == common["first_print_change_k_after"]
        ).all()

        wcommon = wedge_before.join(
            wedge_after, on="ref_date", suffix="_after", how="inner"
        )
        assert (wcommon["wedge_change_k"] == wcommon["wedge_change_k_after"]).all()
```

(`pl`, `append_to_vintage_store`, `compact_partition`, `make_ces_rows` are already imported at the
top of `test_update_guardrail.py` from Task 5.5 — add `import polars as pl` to that module's top
imports if Task 5.5 didn't, keeping I001 order.)

- [ ] **Step 2: Run the test, verify it fails**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_update_guardrail.py::TestFirstPrintUnchanged -q --no-cov`
Expected: FAIL — `ImportError: cannot import name 'make_first_print_window' from
'nfp_vintages.tests._fixtures'` (the builder lands in Step 3).

- [ ] **Step 3: Implement** — add `make_first_print_window` to `_fixtures.py`. It writes, for each
  of `'05'` and `'00'`: month-A `rev0/bmr0` + `rev1/bmr0`, and month-B `rev0/bmr0`, so month-B's
  first print has month-A's `rev1/bmr0` partner (⇒ a non-null `first_print_change_k`). The `'00'`
  and `'05'` legs share vintage stamps so the wedge's same-release check passes. Compacts once.

```python
def make_first_print_window(store) -> None:
    """Seed a two-month first-print window for BOTH the 05 (private) and 00 (total) legs.

    For each industry leg: month-A gets rev0/bmr0 (first print) and rev1/bmr0
    (second print = next month's prior-month partner); month-B gets rev0/bmr0.
    The 00 and 05 legs share vintage stamps so wedge_first_print_changes' same-release
    check passes and it resolves a non-empty frame. Levels differ across legs so the
    wedge is non-trivial. Compacts the partition once at the end.
    """
    # 05 (private) leg
    append_to_vintage_store(
        make_ces_rows(ref_month="2026-01-12", vintage="2026-02-06", revision=0,
                      employment=150_000.0, industry_code="05"), store)
    append_to_vintage_store(
        make_ces_rows(ref_month="2026-01-12", vintage="2026-03-06", revision=1,
                      employment=150_300.0, industry_code="05"), store)
    append_to_vintage_store(
        make_ces_rows(ref_month="2026-02-12", vintage="2026-03-06", revision=0,
                      employment=150_800.0, industry_code="05"), store)
    # 00 (total) leg — co-released vintages, larger levels (total > private)
    append_to_vintage_store(
        make_ces_rows(ref_month="2026-01-12", vintage="2026-02-06", revision=0,
                      employment=300_000.0, industry_code="00"), store)
    append_to_vintage_store(
        make_ces_rows(ref_month="2026-01-12", vintage="2026-03-06", revision=1,
                      employment=300_500.0, industry_code="00"), store)
    append_to_vintage_store(
        make_ces_rows(ref_month="2026-02-12", vintage="2026-03-06", revision=0,
                      employment=301_400.0, industry_code="00"), store)
    compact_partition(store, "ces", True)
```

Add the store-writer import to the top of `_fixtures.py` (keep I001 order; it currently imports only
`date`, `pl`, and `VINTAGE_STORE_SCHEMA`):

```python
from nfp_ingest.vintage_store import append_to_vintage_store, compact_partition
```

- [ ] **Step 4: Run, verify pass**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_update_guardrail.py::TestFirstPrintUnchanged -q --no-cov`
Expected: PASS. If `wedge_before.height >= 1` fails, the two legs' vintage stamps drifted past the
15-day window — re-check the fixture's vintage dates are identical across legs. Then the full
touched-package suite and lint:

```bash
uv run pytest packages/nfp-vintages -m "not network and not slow" --no-cov -q
uv run ruff check packages/nfp-vintages
```

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/tests/_fixtures.py \
        packages/nfp-vintages/src/nfp_vintages/tests/test_update_guardrail.py
git commit -m "test(cli): guardrail first-print + wedge unchanged across a later-month capture

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5.7: Guardrail — calendar-not-advanced ⇒ loud failure (§7.4)

If the release calendar does not cover `T`, the CES tag join is empty and (without a guard) every
row censors out — a silent empty capture. `update --as-of T` must **error loudly** instead. The
matching raise lives in `capture_ces_print` (Phase 4); this test pins the end-to-end behaviour
through the `update` command: `_run_update` must let the capture exception **propagate**, not
swallow-and-continue.

**Red-phase is conditional on Task 5.2's error handling (not visible from this arm).** Write the
test to give an honest red either way:
- If `update`/`_run_update` already propagates (no `try/except` around CES capture), this is a
  **characterization** test — use a deliberate-wrong assert (`assert result.exit_code == 0`) for the
  red, then flip to `!= 0` in Step 3 and add a note that no production change was needed.
- If `_run_update` swallows the capture exception (prints-and-continues), the test fails naturally
  (`exit_code == 0`); Step 3 removes the swallowing so the exception reaches Typer.

**Files:**
- Test: `packages/nfp-vintages/src/nfp_vintages/tests/test_update_guardrail.py`
- Possibly modify: `packages/nfp-vintages/src/nfp_vintages/__main__.py` (only if it swallows)

- [ ] **Step 1: Write the failing test** — start with the deliberate-wrong assert
  (`exit_code == 0`).

```python
class TestCalendarNotAdvancedLoudFailure:
    def test_update_errors_when_calendar_missing_target(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from nfp_vintages.__main__ import app

        # advance_release_calendar is a no-op (stale/missing calendar); capture_ces_print
        # raises because the tag join is empty for T (the Phase 4 loud-failure contract).
        monkeypatch.setattr(
            "nfp_vintages.calendar.advance_release_calendar", lambda: None
        )

        def _raise(as_of, *, store_path=None):
            raise RuntimeError(
                f"no vintage calendar rows for {as_of}; advance the calendar first"
            )

        monkeypatch.setattr("nfp_ingest.capture.capture_ces_print", _raise)

        result = CliRunner().invoke(
            app,
            ["update", "--as-of", "2026-06-12", "--only", "ces",
             "--no-refresh-calendar"],
        )
        # DELIBERATE-WRONG (red): flip to `!= 0` in Step 3.
        assert result.exit_code == 0
        assert "calendar" in (result.output + str(result.exception)).lower()
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_update_guardrail.py::TestCalendarNotAdvancedLoudFailure -q --no-cov`
Expected: FAIL on `assert result.exit_code == 0` **if** `update` correctly propagates (the runner
reports a non-zero exit) — confirm the failure message shows a non-zero exit and the `RuntimeError`
in `result.exception`. If instead it **passes** at this step, `_run_update` is swallowing the
exception (the real bug): proceed to Step 3 to remove the swallow, then the corrected assert fails
until fixed.

- [ ] **Step 3: Implement**

  Flip the assert to the real expectation:

```python
        assert result.exit_code != 0
        assert "calendar" in (result.output + str(result.exception)).lower()
```

  Then inspect `_run_update`'s CES capture call. If it is wrapped so the exception is caught and
  printed (`try: capture_ces_print(...) except Exception: ...continue`), **remove that swallowing**
  for the capture call so the error reaches Typer and yields a non-zero exit. If `_run_update`
  already lets it propagate, no production change is needed — record that in the commit body.
  (`--no-refresh-calendar` keeps `advance_release_calendar` from running; combined with the no-op
  monkeypatch this simulates a stale calendar. The monkeypatch on
  `nfp_ingest.capture.capture_ces_print` lands only if `_run_update` imports capture **deferred** in
  its body per the global constraint — Task 5.2's contract.)

- [ ] **Step 4: Run, verify pass**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_update_guardrail.py::TestCalendarNotAdvancedLoudFailure -q --no-cov`
Expected: PASS. Then the full touched-package suite and lint:

```bash
uv run pytest packages/nfp-vintages -m "not network and not slow" --no-cov -q
uv run ruff check packages/nfp-vintages
```

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/tests/test_update_guardrail.py \
        packages/nfp-vintages/src/nfp_vintages/__main__.py
git commit -m "test(cli): guardrail loud-failure when the release calendar lacks the target month

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

(If no `__main__.py` change was needed, drop it from the `git add` and note 'no production change —
capture exception already propagates' in the commit body.)

---

### Task 5.8: Guardrail — overlap-window divergence diagnostic (§7.2, fixture-only)

Over a synthetic bootstrap∩capture window, compare the **score-relevant levels** — `rev0/bmr0` and
`rev1/bmr0` rows — between a bootstrap reconstruction and a capture, **excluding `-1.0` sentinel
rows** (else a future real-level capture false-flags against a bootstrap `-1`). Per §7.2 the store
is "replaceable, not *identical*": this is **flagged, not asserted-zero**. It is a fixture
diagnostic, **not** a runtime monitor (no persisted reconstruction post-promotion, §6.3).

This task adds the comparator helper `overlap_level_divergence(bootstrap, capture)` to `_fixtures.py`
and a bootstrap/capture pair of fixtures, then asserts only that (a) sentinel rows are excluded from
the comparison and (b) a divergence record is **produced**. It does **not** assert the divergence is
zero.

**Files:**
- Modify: `packages/nfp-vintages/src/nfp_vintages/tests/_fixtures.py`
- Test: `packages/nfp-vintages/src/nfp_vintages/tests/test_update_guardrail.py`

- [ ] **Step 1: Write the failing test** — `overlap_level_divergence` does not exist yet, so the
  import fails (honest red).

```python
class TestOverlapDivergence:
    def test_overlap_diagnostic_excludes_sentinel_and_flags(self, tmp_path):
        from nfp_vintages.tests._fixtures import (
            make_shutdown_sentinel_row,
            overlap_level_divergence,
        )

        store = tmp_path / "store"
        # Bootstrap leg: a real rev0 + a -1 sentinel slot.
        append_to_vintage_store(
            make_ces_rows(ref_month="2025-11-12", vintage="2025-12-05",
                          revision=0, employment=150_800.0, industry_code="05"),
            store,
        )
        append_to_vintage_store(
            make_shutdown_sentinel_row(ref_month="2025-10-12"), store
        )
        compact_partition(store, "ces", True)
        bootstrap = read_vintage_store(
            store, source="ces", seasonally_adjusted=True
        ).collect()

        # Capture leg: the same Nov row at a *diverged* level (replaceable not identical),
        # and crucially NO -1 sentinel (the real path never emits one).
        capture = make_ces_rows(
            ref_month="2025-11-12", vintage="2025-12-05",
            revision=0, employment=151_100.0, industry_code="05",
        )

        report = overlap_level_divergence(bootstrap, capture)

        # (a) the -1 sentinel ref_date is excluded from the scored comparison
        assert -1.0 not in report["bootstrap_employment"].to_list()
        assert -1.0 not in report["capture_employment"].to_list()
        # (b) a divergence record is produced (flag, NOT asserted zero — §7.2)
        assert report.height >= 1
        assert "abs_diff" in report.columns
        assert (report["abs_diff"] >= 0).all()
```

(`read_vintage_store`, `append_to_vintage_store`, `compact_partition`, `make_ces_rows`, `pl` are
already imported at the top of `test_update_guardrail.py`.)

- [ ] **Step 2: Run the test, verify it fails**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_update_guardrail.py::TestOverlapDivergence -q --no-cov`
Expected: FAIL — `ImportError: cannot import name 'overlap_level_divergence' from
'nfp_vintages.tests._fixtures'`.

- [ ] **Step 3: Implement** — add `overlap_level_divergence` to `_fixtures.py`. It filters both
  frames to the **score-relevant rows** (`(rev0/bmr0) ∨ (rev1/bmr0)`), **excludes the `-1.0`
  sentinel** with `employment > 0`, inner-joins on the score ukey, and returns a per-row divergence
  frame (`abs_diff`). The caller decides what to do with a non-zero `abs_diff` — the helper only
  *measures*, never asserts.

```python
def overlap_level_divergence(
    bootstrap: pl.DataFrame, capture: pl.DataFrame
) -> pl.DataFrame:
    """Per-row level divergence on score-relevant rows over a bootstrap∩capture window.

    Compares ``employment`` on the rows that drive the A5 score — first print
    (``rev0/bmr0``) and its prior-month partner (``rev1/bmr0``) — between a bootstrap
    reconstruction and a capture. The ``-1.0`` shutdown sentinel is EXCLUDED
    (``employment > 0``) so a real-level capture is not false-flagged against a
    bootstrap ``-1``. Per §7.2 this is a *diagnostic*: it returns the divergence; it
    does not assert it is zero ("replaceable, not identical").

    Returns one row per overlapping score-key with ``bootstrap_employment``,
    ``capture_employment``, ``abs_diff``.
    """
    score_key = [
        "ref_date", "industry_type", "industry_code", "geographic_type",
        "geographic_code", "revision", "benchmark_revision", "ownership",
    ]

    def _scored(df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(
            (pl.col("employment") > 0)
            & (pl.col("benchmark_revision") == 0)
            & (pl.col("revision").is_in([0, 1]))
        )

    b = _scored(bootstrap).select([*score_key, pl.col("employment").alias("bootstrap_employment")])
    c = _scored(capture).select([*score_key, pl.col("employment").alias("capture_employment")])
    return (
        b.join(c, on=score_key, how="inner", nulls_equal=True)
        .with_columns(
            (pl.col("capture_employment") - pl.col("bootstrap_employment")).abs().alias("abs_diff")
        )
        .sort("ref_date", "revision")
    )
```

- [ ] **Step 4: Run, verify pass**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_update_guardrail.py::TestOverlapDivergence -q --no-cov`
Expected: PASS. Then the full touched-package suite and lint:

```bash
uv run pytest packages/nfp-vintages -m "not network and not slow" --no-cov -q
uv run ruff check packages/nfp-vintages
```

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/tests/_fixtures.py \
        packages/nfp-vintages/src/nfp_vintages/tests/test_update_guardrail.py
git commit -m "test(cli): guardrail overlap-divergence diagnostic (score rows, sentinel-excluded, flag-only)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Arm exit criteria

- `_fixtures.py` exports: `make_ces_rows`, `make_benchmark_double_row`, `make_shutdown_sentinel_row`
  (5.3), `make_first_print_window` (5.6), `overlap_level_divergence` (5.8).
- `_run_update` self-heals fragmented local partitions (5.4); remote self-heal is a documented
  follow-on.
- `test_update_guardrail.py` covers idempotence (5.5), first-print + wedge unchanged (5.6),
  calendar-not-advanced loud-failure (5.7), overlap-divergence diagnostic (5.8).
- Every store-writing test pins `store_path=tmp_path`; no test writes canonical MinIO.
- Full `packages/nfp-vintages -m "not network and not slow"` suite + `ruff check packages/nfp-vintages`
  green after every task.

## Phase 6 — QCEW conditional capture

> **Spec:** `specs/cli_production_workflow.md` §5.2 (QCEW conditional capture), §5.0 (calendar-advance
> dependency), §6.1 (ukey now keys `size_class_*`), §6.3 (corrected-level runtime warning). Build-order
> item 6. **Global Constraints** (plan §"Global Constraints") apply to every task below verbatim.
>
> **Prerequisites already landed in earlier phases — consume, do NOT re-implement:**
> - **Phase 1** — `nfp_ingest/qcew_acquire.py` exposes the **public** `acquire_qcew_levels(start_year,
>   end_year=None)` and `acquire_qcew_size_native(start_year, end_year=None)` (verbatim relocation of
>   `rebuild_store._acquire_qcew_levels` / `_acquire_qcew_size_native`). Both loop over **all quarters of
>   every year** in `[start_year, end_year]` and tag `revision=0`. (`acquire_qcew_levels` at
>   `qcew_acquire.py:151`.)
> - **Phase 2** — `append_to_vintage_store` / `compact_partition` ukey lists now include `ownership`,
>   `size_class_type`, `size_class_code` (`vintage_store.py`, §6.1), so distinct QCEW Q1 size buckets no
>   longer collapse on append.
> - **Phase 4** — `nfp_ingest/capture.py` already defines `@dataclass CaptureResult(appended, corrected,
>   skipped)`, `@dataclass CorrectedLevel(...)`, `capture_ces_print(...)`, `_detect_corrected_levels(
>   new_rows, store_path, source, seasonally_adjusted)`, **and a `capture_qcew_quarter` STUB** (raises
>   `NotImplementedError`, `capture.py:284`). Phase 6 **replaces that stub body** and **adds**
>   `_knowable_qcew_quarter` to the same module — it does **not** redefine the Phase-4 symbols.
>   `capture.py` already imports `VINTAGE_STORE_SCHEMA` (L22) and `append_to_vintage_store`/
>   `compact_partition`/`read_vintage_store` (L25-29) at module top.
> - **Phase 5** — `nfp_vintages/__main__.py` already defines `_run_update(as_of: date, *, only=None,
>   refresh_calendar=True, store_path=None)` and the `update` Typer command (`--as-of` →
>   `_date.fromisoformat` → `_run_update`). **`_run_update` ALREADY calls `capture_qcew_quarter`** —
>   `if only in (None, "qcew"): _capture.capture_qcew_quarter(as_of, store_path=...)` — against the Phase-4
>   stub, and 5.2's orchestration test asserts the call order `["calendar","ces","qcew","indicators"]`.
>   **Phase 6 MUST NOT re-add that call** (re-adding double-wires it — see the PINNED SEAM below).
>
> **PINNED SEAM — QCEW wiring (5.2 ↔ 6.1 ↔ 6.2)** (plan Interface Contract, "Cross-phase dependencies"):
> The old plan contradicted itself (5.2 wired the call *and* 6.2 claimed to "add the QCEW leg"). Canonical
> division: **5.2 OWNS** the `capture_qcew_quarter` call in `_run_update`, the `--only ces|qcew|indicators`
> option, and the order-assertion test. **6.1** replaces the `capture_qcew_quarter` STUB with the real
> single-quarter capture. **6.2** adds **ONLY** the `--only qcew` behavior coverage + the knowable-quarter
> steady-state-no-op (`skipped=1`) integration test *through* `update` — it is **TEST-ONLY** (no
> `__main__.py` production delta) and **MUST NOT** re-add the call to `_run_update`.
>
> **Key code facts grounding the implementation (verified read-only):**
> - `get_qcew_vintage_date(ref_quarter, ref_year, revision=0)` takes `ref_quarter` as the **string
>   `'Q1'..'Q4'`** (`revision_schedules.py:299`). With the §5.0 calendar advanced it returns the exact
>   `vintage_date`; without it the lag fallback (`revision_schedules.py:358-365`) assigns a day-1
>   approximation.
> - `build_qcew_panel(raw)` (`qcew_crosswalk.py:174`) returns 14-of-16 store columns but its final
>   `.select(...)` **omits `size_class_type`/`size_class_code`** — `capture_qcew_quarter` must null-fill
>   those two before append. It stamps `source='qcew'`, `seasonally_adjusted=False`, `benchmark_revision=0`,
>   `revision` from the raw rows.
> - `build_size_class_panel(native)` (`size_class.py:61`) returns the full store schema **with non-null**
>   `size_class_type`/`size_class_code`; Q1-only.
> - `acquire_qcew_levels` / `acquire_qcew_size_native` fetch **whole years**, so the single-quarter wrapper
>   fetches the containing year then **filters** to the one quarter (`year`/`qtr` survive into the raw frame
>   the crosswalk consumes; the size endpoint URL is Q1-only by path).
> - `append_to_vintage_store` is anti-join idempotent; `compact_partition` keeps `MIN(vintage_date)` per
>   ukey. Both are container-safe (thread `storage_options_for`); reuse them — **no new raw `write_parquet`
>   in production code**.

---

### Task 6.1: `capture_qcew_quarter` — knowable-quarter selection + single-quarter capture

**Files:**
- Modify: `packages/nfp-ingest/src/nfp_ingest/capture.py` (replace the Phase-4 stub; add `_knowable_qcew_quarter`)
- Test: `packages/nfp-ingest/src/nfp_ingest/tests/test_capture.py` (extend — created in Phase 4)

This task **replaces** the Phase-4 `capture_qcew_quarter` STUB body (the `NotImplementedError` at
`capture.py:284`) with the real single-quarter capture, and **adds** `_knowable_qcew_quarter(as_of)` (the
§5.2 knowable test). It reuses the Phase-4 `CaptureResult`, `CorrectedLevel`, and `_detect_corrected_levels`
symbols already in the module. New imports are merged into the existing sorted import groups (I001) — they
are **not** dumped at the top, and the already-present `VINTAGE_STORE_SCHEMA` / `append_to_vintage_store` /
`compact_partition` imports are **not** duplicated.

- [ ] **Step 1: Write the failing test** — `_knowable_qcew_quarter` picks the most recent quarter whose
  rev-0 `vintage_date ≤ as_of`, returns `None` when nothing is knowable yet; `capture_qcew_quarter` is a
  `skipped=1` no-op when no new quarter is knowable / already stored, and appends a rev-0 row (with the size
  columns null-filled) when a new quarter becomes knowable.

  **Store-write test safety (HARD).** Both `capture_qcew_quarter` tests pin `store_path=tmp_path`. Under
  pytest the root `conftest.py` `load_dotenv()`s before any `nfp_*` import, so the default
  `VINTAGE_STORE_PATH` resolves to the **canonical MinIO store** — a test that let the real append/read
  reach the default path would write/wipe canonical MinIO. Every store-touching call here is pinned to
  `tmp_path`, and the unit tests monkeypatch `acquire_qcew_levels` (no network — a `@pytest.mark.network`
  test is not in scope for this task).

  Append to `packages/nfp-ingest/src/nfp_ingest/tests/test_capture.py` (imports merged into that file's
  existing top-of-module import block — I001/E402; add only names not already imported there):

  ```python
  from datetime import date

  import polars as pl

  from nfp_ingest import capture as _cap
  from nfp_ingest.capture import (
      CaptureResult,
      _knowable_qcew_quarter,
      capture_qcew_quarter,
  )
  from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA


  # --- helpers -------------------------------------------------------------

  def _qcew_store_row(
      ref_date: date,
      vintage_date: date,
      *,
      industry_code: str = "05",
      industry_type: str = "total",
      ownership: str = "private",
      revision: int = 0,
      employment: float = 130_000.0,
      size_class_type: str | None = None,
      size_class_code: str | None = None,
  ) -> dict:
      """One VINTAGE_STORE_SCHEMA-conformant QCEW row (NSA)."""
      return {
          "geographic_type": "national",
          "geographic_code": "00",
          "ownership": ownership,
          "industry_type": industry_type,
          "industry_code": industry_code,
          "ref_date": ref_date,
          "vintage_date": vintage_date,
          "revision": revision,
          "benchmark_revision": 0,
          "employment": employment,
          "size_class_type": size_class_type,
          "size_class_code": size_class_code,
          "source": "qcew",
          "seasonally_adjusted": False,
      }


  def _write_qcew_partition(rows: list[dict], store_path) -> None:
      """Seed a tmp_path QCEW partition directly (a fixture, not capture output).

      This raw write is test-fixture scaffolding to a tmp_path store ONLY — never
      production code (production appends go through append_to_vintage_store).
      """
      df = pl.DataFrame(rows, schema=VINTAGE_STORE_SCHEMA)
      pdir = store_path / "source=qcew" / "seasonally_adjusted=false"
      pdir.mkdir(parents=True, exist_ok=True)
      df.drop(["source", "seasonally_adjusted"]).write_parquet(pdir / "data.parquet")


  def _qcew_panel_rows(ref_year: int, qtr: int, employment: float) -> pl.DataFrame:
      """Stand-in for build_qcew_panel output: 14-of-16 cols, NO size_class_*.

      Mirrors qcew_crosswalk.build_qcew_panel's final .select (which omits
      size_class_type/size_class_code) so capture_qcew_quarter's null-fill is
      exercised. The test monkeypatches the schedule lookup, so any placeholder
      vintage_date that satisfies vintage_date <= as_of is fine here.
      """
      ref_month = (qtr - 1) * 3 + 1
      ref = date(ref_year, ref_month, 1)
      cols = [
          c for c in VINTAGE_STORE_SCHEMA
          if c not in ("size_class_type", "size_class_code")
      ]
      return pl.DataFrame(
          {
              "geographic_type": ["national"],
              "geographic_code": ["00"],
              "ownership": ["private"],
              "industry_type": ["total"],
              "industry_code": ["05"],
              "ref_date": [ref],
              "vintage_date": [date(ref_year, ref_month + 4, 1)],
              "revision": [0],
              "benchmark_revision": [0],
              "employment": [employment],
              "source": ["qcew"],
              "seasonally_adjusted": [False],
          }
      ).select(cols)


  # --- _knowable_qcew_quarter ---------------------------------------------

  class TestKnowableQcewQuarter:
      def test_picks_most_recent_knowable_quarter(self, monkeypatch):
          # Q1-2024 rev0 published 2024-05-01; Q2-2024 rev0 published 2024-08-01.
          def fake_vdate(ref_quarter, ref_year, revision):
              table = {
                  ("Q1", 2024): date(2024, 5, 1),
                  ("Q2", 2024): date(2024, 8, 1),
                  ("Q3", 2024): date(2024, 11, 1),
              }
              return table.get((ref_quarter, ref_year), date(2099, 1, 1))

          monkeypatch.setattr(_cap, "get_qcew_vintage_date", fake_vdate)
          # As of 2024-06-01: Q1-2024 is knowable, Q2-2024 is not yet.
          assert _knowable_qcew_quarter(date(2024, 6, 1)) == ("Q1", 2024)

      def test_returns_none_when_no_quarter_knowable(self, monkeypatch):
          # Every candidate publishes in the far future ⇒ nothing knowable.
          monkeypatch.setattr(
              _cap,
              "get_qcew_vintage_date",
              lambda ref_quarter, ref_year, revision: date(2099, 1, 1),
          )
          assert _knowable_qcew_quarter(date(2024, 6, 1)) is None


  # --- capture_qcew_quarter -----------------------------------------------

  class TestCaptureQcewQuarter:
      def test_no_new_quarter_returns_skipped_no_append(self, tmp_path, monkeypatch):
          # Store already holds Q1-2024; as-of makes Q1-2024 the newest knowable.
          _write_qcew_partition(
              [_qcew_store_row(date(2024, 1, 1), date(2024, 5, 1))], tmp_path
          )

          monkeypatch.setattr(
              _cap,
              "get_qcew_vintage_date",
              lambda ref_quarter, ref_year, revision: (
                  date(2024, 5, 1)
                  if (ref_quarter, ref_year) == ("Q1", 2024)
                  else date(2099, 1, 1)
              ),
          )

          def _boom(*a, **k):  # acquire must NOT be called on a no-op
              raise AssertionError("acquire_qcew_levels called on a no-op month")

          monkeypatch.setattr(_cap, "acquire_qcew_levels", _boom)

          result = capture_qcew_quarter(date(2024, 6, 1), store_path=tmp_path)

          assert isinstance(result, CaptureResult)
          assert result.appended == 0
          assert result.skipped == 1
          assert result.corrected == []

      def test_knowable_new_quarter_appends_rev0(self, tmp_path, monkeypatch):
          # Empty store; Q1-2024 becomes knowable as of 2024-06-01.
          monkeypatch.setattr(
              _cap,
              "get_qcew_vintage_date",
              lambda ref_quarter, ref_year, revision: (
                  date(2024, 5, 1)
                  if (ref_quarter, ref_year) == ("Q1", 2024)
                  else date(2099, 1, 1)
              ),
          )
          # acquire returns a raw frame; build_qcew_panel / build_size_class_panel
          # are monkeypatched to the test panel (no real crosswalk / network).
          monkeypatch.setattr(
              _cap,
              "acquire_qcew_levels",
              lambda start_year, end_year=None: pl.DataFrame(
                  {"year": [2024], "qtr": [1]}
              ),
          )
          monkeypatch.setattr(
              _cap,
              "acquire_qcew_size_native",
              lambda start_year, end_year=None: pl.DataFrame({"year": [2024]}),
          )
          monkeypatch.setattr(
              _cap, "build_qcew_panel", lambda raw: _qcew_panel_rows(2024, 1, 130_000.0)
          )
          # Size leg disabled for this test (return an empty Q1 size frame).
          empty_size = pl.DataFrame(schema=VINTAGE_STORE_SCHEMA).filter(pl.lit(False))
          monkeypatch.setattr(_cap, "build_size_class_panel", lambda native: empty_size)

          result = capture_qcew_quarter(date(2024, 6, 1), store_path=tmp_path)

          assert result.skipped == 0
          assert result.appended == 1
          assert result.corrected == []

          stored = pl.read_parquet(
              str(tmp_path / "source=qcew" / "seasonally_adjusted=false" / "*.parquet")
          )
          assert stored.height == 1
          assert stored["revision"].to_list() == [0]
          assert stored["industry_code"].to_list() == ["05"]
          # null-fill of the missing size cols held:
          assert stored["size_class_type"].to_list() == [None]
          assert stored["size_class_code"].to_list() == [None]
  ```

- [ ] **Step 2: Run the test, confirm it fails for the stated reason**
  - `uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/test_capture.py::TestKnowableQcewQuarter -q --no-cov`
  - `uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/test_capture.py::TestCaptureQcewQuarter -q --no-cov`
  - **Expected (exact reason):** collection-time
    `ImportError: cannot import name '_knowable_qcew_quarter' from 'nfp_ingest.capture'`.
    `capture_qcew_quarter` already resolves (the Phase-4 stub), so the *only* unresolved import is
    `_knowable_qcew_quarter`. If the failure is instead `NotImplementedError` or any other error, **stop** —
    the test did not fail for the stated reason.

- [ ] **Step 3: Implement** — add `_knowable_qcew_quarter` and replace the stub body in `capture.py`.

  **Imports (merge into existing sorted groups — do NOT add a new block at the top, do NOT duplicate L22 /
  L25-29).** Add these names to the already-present import groups of `capture.py`:
  - to the `nfp_ingest.*` group (after the existing `from nfp_ingest.releases import ...` /
    `from nfp_ingest.vintage_store import ...` lines), add:
    ```python
    from nfp_ingest.qcew_acquire import acquire_qcew_levels, acquire_qcew_size_native
    from nfp_ingest.qcew_crosswalk import build_qcew_panel
    from nfp_ingest.size_class import build_size_class_panel
    ```
  - to the `nfp_lookups.*` group (alongside the existing `from nfp_lookups.industry import ...` /
    `from nfp_lookups.paths import ...` / `from nfp_lookups.schemas import ...`), add:
    ```python
    from nfp_lookups.revision_schedules import get_qcew_vintage_date
    ```
  > `VINTAGE_STORE_SCHEMA` (L22), `append_to_vintage_store` / `compact_partition` / `read_vintage_store`
  > (L25-29), `VINTAGE_STORE_PATH`, `Path`, `logger`, `CaptureResult`, `CorrectedLevel`, and
  > `_detect_corrected_levels` are **already** present in `capture.py` (Phase 4) — reuse them; do not
  > re-import or redefine. Keep imports sorted within each group (I001) and at module top (E402).

  **Add `_knowable_qcew_quarter` and replace the stub body.** Replace the existing Phase-4 stub —

  ```python
  def capture_qcew_quarter(
      as_of: date,
      *,
      store_path: Path = VINTAGE_STORE_PATH,
  ) -> CaptureResult:
      """Stub — QCEW quarterly capture is implemented in Phase 6 (spec §6.3).
      ...
      """
      raise NotImplementedError(
          "capture_qcew_quarter is implemented in Phase 6 (spec §6.3)"
      )
  ```

  — with the real implementation below (the `_knowable_qcew_quarter` helper goes immediately above it):

  ```python
  # ---------------------------------------------------------------------------
  # QCEW conditional quarter capture (spec §5.2)
  # ---------------------------------------------------------------------------

  # How far back to scan for a knowable quarter. QCEW rev-0 lags the reference
  # quarter by ~5 months, so 8 candidate quarters (2 years) always covers the
  # newest-knowable quarter for any monthly as-of.
  _QCEW_CANDIDATE_QUARTERS = 8


  def _knowable_qcew_quarter(as_of: date) -> tuple[str, int] | None:
      """Most recent QCEW quarter whose rev-0 ``vintage_date`` is ``<= as_of``.

      Iterates candidate ``(ref_quarter, ref_year)`` pairs newest-first and returns
      the first whose ``get_qcew_vintage_date(..., revision=0)`` is on or before
      ``as_of``. Returns ``None`` when no candidate is knowable yet (the steady-state
      monthly no-op — QCEW is quarterly, §5.2).

      Requires the §5.0 calendar to be advanced so the schedule returns real release
      dates rather than the day-1 lag fallback (``revision_schedules.py:358-365``).

      Parameters
      ----------
      as_of : date
          Knowability cutoff.

      Returns
      -------
      tuple[str, int] | None
          ``(ref_quarter, ref_year)`` for the newest knowable quarter (e.g.
          ``("Q1", 2024)``), or ``None`` when none is knowable as of ``as_of``.
      """
      # The quarter containing ``as_of`` cannot have been published yet, so start
      # from the previous quarter and walk back.
      q = (as_of.month - 1) // 3 + 1
      year = as_of.year
      q -= 1
      if q == 0:
          q = 4
          year -= 1

      for _ in range(_QCEW_CANDIDATE_QUARTERS):
          ref_quarter = f"Q{q}"
          rev0_vdate = get_qcew_vintage_date(ref_quarter, year, 0)
          if rev0_vdate <= as_of:
              return (ref_quarter, year)
          q -= 1
          if q == 0:
              q = 4
              year -= 1
      return None


  def capture_qcew_quarter(
      as_of: date,
      *,
      store_path: Path = VINTAGE_STORE_PATH,
  ) -> CaptureResult:
      """Capture the newest knowable QCEW quarter and append it to the store.

      Most months this is a **no-op** (QCEW is quarterly): if no new quarter is
      knowable as of ``as_of``, or the newest knowable quarter is already in the
      store, returns ``CaptureResult(appended=0, corrected=[], skipped=1)``.

      Otherwise (spec §5.2) fetches the containing **year** via the relocated public
      acquire helpers, filters to the single knowable quarter, runs the crosswalk
      (``build_qcew_panel`` for levels, ``build_size_class_panel`` for the Q1 size
      cross-product), null-fills the ``size_class_*`` columns the levels builder
      omits, censors ``vintage_date <= as_of``, runs the §6.3 corrected-level
      comparison, then appends + compacts the ``(qcew, seasonally_adjusted=False)``
      partition. QCEW is NSA-only, so every row is tagged ``revision=0`` /
      ``seasonally_adjusted=False`` by the builders.

      Parameters
      ----------
      as_of : date
          Knowability cutoff. No row with ``vintage_date > as_of`` is appended.
      store_path : Path
          Root of the Hive-partitioned vintage store.

      Returns
      -------
      CaptureResult
          ``skipped=1`` (and ``appended=0``) when there is no new quarter to
          capture; otherwise the append count and any corrected-level warnings.
      """
      knowable = _knowable_qcew_quarter(as_of)
      if knowable is None:
          logger.info("QCEW: no knowable quarter as of %s — skipping", as_of)
          return CaptureResult(appended=0, corrected=[], skipped=1)

      ref_quarter, ref_year = knowable
      qtr = int(ref_quarter[1])

      # Fetch the containing YEAR (the helpers loop over full years), then filter to
      # the one knowable quarter. The levels endpoint carries year+qtr; the size
      # endpoint is Q1-only by URL path, so the size leg only runs for Q1.
      raw_levels = acquire_qcew_levels(ref_year, ref_year)
      raw_levels_q = raw_levels.filter(
          (pl.col("year").cast(pl.Int64) == ref_year)
          & (pl.col("qtr").cast(pl.Int64) == qtr)
      )
      levels = build_qcew_panel(raw_levels_q)
      # build_qcew_panel's .select omits size_class_* (qcew_crosswalk.py); the store
      # schema requires them, so null-fill before append.
      levels = levels.with_columns(
          size_class_type=pl.lit(None, pl.Utf8),
          size_class_code=pl.lit(None, pl.Utf8),
      )

      parts: list[pl.DataFrame] = [levels]
      if qtr == 1:
          raw_size = acquire_qcew_size_native(ref_year, ref_year)
          size = build_size_class_panel(raw_size)
          if size.height:
              parts.append(size)

      new_rows = (
          pl.concat(parts, how="diagonal_relaxed")
          .select(list(VINTAGE_STORE_SCHEMA))
          .cast(VINTAGE_STORE_SCHEMA)
      )

      # Censor to the knowability cutoff.
      new_rows = new_rows.filter(pl.col("vintage_date") <= as_of)
      if new_rows.height == 0:
          logger.info(
              "QCEW: %s %d knowable but no rows survive vintage_date <= %s",
              ref_quarter,
              ref_year,
              as_of,
          )
          return CaptureResult(appended=0, corrected=[], skipped=1)

      # §6.3 corrected-level comparison BEFORE the append anti-join.
      corrected = _detect_corrected_levels(
          new_rows, store_path, source="qcew", seasonally_adjusted=False
      )
      for c in corrected:
          logger.warning(
              "CORRECTED-LEVEL qcew %s rev=%d bmr=%d: stored=%.1f incoming=%.1f",
              c.ref_date,
              c.revision,
              c.benchmark_revision,
              c.stored_employment,
              c.incoming_employment,
          )

      appended = append_to_vintage_store(new_rows, store_path)
      compact_partition(store_path, source="qcew", seasonally_adjusted=False)

      skipped = 0 if appended else 1
      logger.info(
          "QCEW: captured %s %d — appended %d rows (%d corrected)",
          ref_quarter,
          ref_year,
          appended,
          len(corrected),
      )
      return CaptureResult(appended=appended, corrected=corrected, skipped=skipped)
  ```

  > **`.select(list(VINTAGE_STORE_SCHEMA)).cast(VINTAGE_STORE_SCHEMA)`** mirrors the in-file
  > `_remap_ces_to_store_schema` pattern (`capture.py:121-122`) — `VINTAGE_STORE_SCHEMA` (an
  > `OrderedDict`-like mapping) is both the column-order source (`list(...)`) and the dtype map
  > (`.cast(...)`); do not wrap it in `dict(...)` or `list(....keys())`.

- [ ] **Step 4: Run, confirm pass + full suite + lint**
  - `uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/test_capture.py::TestKnowableQcewQuarter -q --no-cov` → **PASS** (2).
  - `uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/test_capture.py::TestCaptureQcewQuarter -q --no-cov` → **PASS** (2).
  - Full touched-package suite (mandatory — catches stale fixtures the narrow test misses):
    `uv run pytest packages/nfp-ingest -m "not network and not slow" --no-cov` → **PASS**.
  - `uv run ruff check packages/nfp-ingest/src/nfp_ingest/capture.py packages/nfp-ingest/src/nfp_ingest/tests/test_capture.py` → **clean**.

- [ ] **Step 5: Commit** (scoped pathspec — never `-A`/`.`)
  ```bash
  git add packages/nfp-ingest/src/nfp_ingest/capture.py \
          packages/nfp-ingest/src/nfp_ingest/tests/test_capture.py
  git commit -m "feat(ingest): implement capture_qcew_quarter conditional quarter capture

Replace the Phase-4 stub: iterate candidate quarters, pick the most recent
whose rev-0 vintage_date <= as_of; fetch the containing year via
acquire_qcew_levels then filter to the one quarter (Q1-only for the size
cross-product). Null-fill the size_class_* columns build_qcew_panel omits, run
the §6.3 corrected-level check, then append + compact the (qcew, False)
partition. Returns skipped=1 when no new quarter is knowable (the steady-state
monthly no-op). Spec §5.2.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
  ```

---

### Task 6.2: `--only qcew` behavior + knowable-quarter no-op coverage through `update`

**Files:**
- Test **only**: `packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py` (extend — created in Phase 5)

> **PINNED SEAM — this task is TEST-ONLY. Do NOT edit `__main__.py`.** Phase 5 (Task 5.2) already wired
> the QCEW call into `_run_update` (`if only in (None, "qcew"): _capture.capture_qcew_quarter(as_of,
> store_path=...)`), already added the `--only ces|qcew|indicators` option, and already asserted the call
> order `["calendar","ces","qcew","indicators"]` against the Phase-4 stub. Re-adding the leg here would
> **double-wire** it (the contradiction the seam resolves). Task 6.1 made the capture **real**; this task
> adds the one genuinely-new piece 5.2 left open: an integration test proving `update --only qcew` drives
> the **real** `capture_qcew_quarter` no-op end-to-end and **gates the CES leg off**.

The `update` command exposes **no `--store` flag** (production surface, plan §"Command surface"). So a test
that let the real `capture_qcew_quarter` reach its **append** path through the CliRunner would write
`VINTAGE_STORE_PATH` = **canonical MinIO** under pytest — the exact canonical-wipe scar
(`store-write-test-safety`). The safe, real no-op exercises the **None-knowable** branch of 6.1's
`_knowable_qcew_quarter`: patch its dependency `get_qcew_vintage_date → date(2099, 1, 1)` so
`_knowable_qcew_quarter` returns `None`, and `capture_qcew_quarter` returns `skipped=1` **before any
`acquire_qcew_levels` / `read_vintage_store` / `append_to_vintage_store` call** — it never touches the
store. That is *why* no `tmp_path` pin is needed here (vs the old 6.2, which faked `capture_qcew_quarter`
itself and asserted nothing real).

> **HARD constraint for any future variant:** a test that wants the **knowable/append** path must call
> `_run_update(as_of=..., only="qcew", store_path=tmp_path)` **directly** (pinning `store_path`), never the
> CliRunner `update` default — the CLI has no `--store` and would target canonical MinIO. The
> `--no-refresh-calendar` flag keeps the calendar leg fully offline.

- [ ] **Step 1: Write the failing test** — `update --only qcew` drives the real `capture_qcew_quarter`
  no-op (`skipped=1`, exit 0, the `QCEW: ... skipped 1` echo) and the CES leg does **not** run; the calendar
  scrape is sidestepped via `--no-refresh-calendar`, and `get_qcew_vintage_date` is patched far-future so the
  capture stays a store-free no-op.

  Append to `packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py` (imports merged into that
  file's existing top-of-module import block — I001/E402; add only names not already imported there):

  ```python
  from datetime import date

  from typer.testing import CliRunner

  import nfp_vintages.__main__ as cli

  runner = CliRunner()


  class TestUpdateOnlyQcew:
      def test_only_qcew_drives_real_noop_and_skips_ces(self, monkeypatch):
          """--only qcew runs the REAL capture_qcew_quarter no-op; CES leg gated off.

          No --store flag exists on `update`, so the capture must NOT reach the
          store. Patching get_qcew_vintage_date far-future makes
          _knowable_qcew_quarter return None ⇒ capture_qcew_quarter returns
          skipped=1 before any acquire/append/read (store-safe under pytest, where
          VINTAGE_STORE_PATH is canonical MinIO).
          """
          import nfp_ingest.capture as _cap
          import nfp_vintages.calendar as _cal

          # Offline calendar (defensive; --no-refresh-calendar also skips it).
          monkeypatch.setattr(_cal, "advance_release_calendar", lambda: None)

          # Far-future schedule ⇒ no knowable quarter ⇒ real no-op, no store touch.
          monkeypatch.setattr(
              _cap,
              "get_qcew_vintage_date",
              lambda ref_quarter, ref_year, revision: date(2099, 1, 1),
          )

          # acquire MUST NOT be reached on the no-op path.
          def _boom(*a, **k):
              raise AssertionError("acquire_qcew_levels reached on a no-op")

          monkeypatch.setattr(_cap, "acquire_qcew_levels", _boom)

          # CES leg must be gated OFF by --only qcew; flag if it runs.
          def _ces_boom(*a, **k):
              raise AssertionError("capture_ces_print ran under --only qcew")

          monkeypatch.setattr(_cap, "capture_ces_print", _ces_boom)

          result = runner.invoke(
              cli.app,
              ["update", "--as-of", "2024-06-12", "--only", "qcew", "--no-refresh-calendar"],
          )

          # HARD assertion: the real no-op path ran to completion under --only qcew.
          assert result.exit_code == 0, result.output
          # SOFT assertion: 5.2 emits a QCEW outcome line reporting the skip. The
          # exact wording is OWNED BY 5.2 (not yet authored) — if it lands
          # differently, reconcile THESE strings to 5.2's actual echo, never the
          # command body.
          assert "QCEW" in result.output
          assert "skipped 1" in result.output
  ```

  > This asserts behavior, not just "was qcew called" (5.2 already covers the call + ordering). It pins:
  > (a) the real 6.1 no-op path returns `skipped=1`, (b) `update` surfaces it (exit 0 + a QCEW outcome
  > line), and (c) `--only qcew` gates the CES leg off. The `_boom` / `_ces_boom` guards genuinely catch a
  > buggy 5.2 gate or a 6.1 no-op regression — they prove the store-free path is taken and the CES leg is
  > skipped — so no `tmp_path` is needed (nothing touches the store). **`exit_code == 0` is the hard
  > assertion; the echo strings are soft** — they depend on 5.2's not-yet-authored echo wording, so if the
  > `QCEW:`/`skipped 1` text differs, reconcile the **assertion** to 5.2's actual echo (5.2 owns the
  > wording) — do **not** change `__main__.py`.

- [ ] **Step 2: Run the test, confirm it PASSES** — *(this is the one task where failing-first does not
  apply; see the note)*
  - `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py::TestUpdateOnlyQcew -q --no-cov`
  - **Why no red phase:** the seam assigns 6.2 **no production code** (test-only). The red→green work for
    the QCEW leg already happened — in **5.2** (the leg wired against the Phase-4 stub) and **6.1** (the
    real capture). By the time you reach 6.2, both are merged, so this cross-seam **integration/regression**
    test passes **by construction**. Run it and expect green.
  - **If it fails, do NOT make it pass by editing `__main__.py`.** Read the failure as a diagnostic:
    - an `AssertionError` on `exit_code != 0` / a `_boom` or `_ces_boom` `AssertionError` ⇒ a defect in 5.2's
      wiring (call missing, `--only` gate wrong) or in 6.1's no-op branch ⇒ **fix it in that phase's scope**;
    - an `AssertionError` on the soft `"skipped 1"` / `"QCEW"` strings only ⇒ 5.2's echo wording differs ⇒
      **reconcile the test assertion** to 5.2's actual echo (Step 3 note), never the command body;
    - `ImportError: cannot import name '_knowable_qcew_quarter'` ⇒ 6.1 is not yet merged ⇒ land 6.1 first.

- [ ] **Step 3: Implement** — **none.** This is the seam's payoff: 5.2 already wired `_run_update`'s QCEW
  leg + the `--only qcew` gate + the echo, and 6.1 made `capture_qcew_quarter` real, so the test passes with
  **no `__main__.py` edit**. Do not duplicate the calendar / CES / indicators / QCEW steps; do not add a
  `capture_qcew_quarter` call to `__main__.py` (that would double-wire the seam).

  > The only adjustment ever permitted in this task is to the **test assertion**: if 5.2's echo wording
  > differs (e.g. it prints `QCEW: appended 0, skipped 1, corrected 0`), update the soft string assertions
  > to match 5.2's actual output — never the command body.

- [ ] **Step 4: Run, confirm pass + full suite + lint**
  - `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py::TestUpdateOnlyQcew -q --no-cov` → **PASS**.
  - Full update-CLI file (catches interaction with 5.2's CES/indicator/ordering tests):
    `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py -q --no-cov` → **PASS**.
  - Full touched-package suite:
    `uv run pytest packages/nfp-vintages -m "not network and not slow" --no-cov` → **PASS**.
  - `uv run ruff check packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py` → **clean**.

- [ ] **Step 5: Commit** (scoped pathspec — only the test file changed)
  ```bash
  git add packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py
  git commit -m "test(cli): cover --only qcew real no-op through update

Drive the real capture_qcew_quarter (Phase 6.1) no-op end-to-end via
\`update --only qcew\`: patch get_qcew_vintage_date far-future so the capture
returns skipped=1 before touching the store (no --store flag => canonical-MinIO
safe under pytest), assert exit 0 + the QCEW skipped echo, and confirm the CES
leg is gated off. The QCEW call into _run_update is owned by Phase 5 (Task 5.2)
— no __main__.py change here. Spec §5.2.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
  ```

---

**Phase 6 done-when:** `capture_qcew_quarter` (replacing the Phase-4 stub at `capture.py:284`) selects the
newest knowable quarter via `_knowable_qcew_quarter` (rev-0 `vintage_date ≤ as_of`), fetches the containing
year and filters to that one quarter, null-fills the `size_class_*` columns the levels builder omits, runs
the §6.3 corrected-level comparison, and appends + compacts the `(qcew, False)` partition — returning
`skipped=1` on the steady-state monthly no-op. `update --only qcew` drives that real no-op end-to-end (exit
0, QCEW skipped echo, CES leg gated off), with the `_run_update` QCEW call **owned by Phase 5** (no
`__main__.py` change in this phase). No firewall path (`transform_to_panel`, `build_model_data`, A1/A2/A3
goldens, `nfp-model/*`) is touched; no test writes the canonical store.

---
## Phase 7 — `status` command + `nfp_vintages/store_status.py`

**Spec:** §8 (`specs/cli_production_workflow.md`). **Consumes (must be DONE):** Phase 5's
`_run_update`/`_run_snapshot` helpers and the `nfp_ingest.vintage_store.read_vintage_store`
primitive (Phases 1–4). **Provides for Phase 8:** `compute_status(store_path, as_of)` +
`StoreStatus.uncaptured` (the `watch` command's idempotence check).

**What this phase builds:** a cheap, read-only health + knowability report. Three deliverables:

1. **Task 7.1** — `PartitionCoverage` + `StoreStatus` dataclasses and `compute_status`
   per-`(source, seasonally_adjusted)` coverage (raw row presence, no `employment > 0` filter
   so the Oct-2025 `-1` sentinel counts as present).
2. **Task 7.2** — the forward **UNCAPTURED** alarm + **missing-month** list folded into
   `compute_status`, plus `format_status` rendering incl. the resolved-URI header and the
   LOCAL-FALLBACK warning when the store is not remote.
3. **Task 7.3** — the `status` Typer command in `__main__.py` with **all imports deferred**
   inside the body (so `load_dotenv()` in the app callback runs before `VINTAGE_STORE_PATH`
   resolves at `paths.py` import-time).

**Firewall:** this phase never imports or calls `transform_to_panel`, `build_model_data`,
`first_print.py`, or any A1/A2/A3 golden path. Store **writing** is not done here; all tests
build a synthetic `tmp_path` store with `pl.DataFrame.write_parquet` into the Hive layout and
read it back — never a real MinIO store (conftest auto-loads prod creds via `.env`).

### Cross-phase contract (pinned — do NOT deviate)

```python
# Phase 7 (nfp_vintages/store_status.py) — define exactly:

@dataclass(frozen=True)
class PartitionCoverage:
    source: str
    seasonally_adjusted: bool
    earliest_ref: date | None
    latest_ref: date | None
    row_count: int
    last_capture: date | None
    distinct_vintages: int

@dataclass(frozen=True)
class StoreStatus:
    store_uri: str
    is_remote: bool
    is_canonical: bool
    per_partition: list[PartitionCoverage] = field(default_factory=list)
    uncaptured: list[str] = field(default_factory=list)   # consumed by Phase 8 watch
    missing_months: list[str] = field(default_factory=list)
    corrected: list[str] = field(default_factory=list)

def compute_status(store_path=VINTAGE_STORE_PATH, as_of: date | None = None) -> StoreStatus: ...
def format_status(status: StoreStatus) -> str: ...
```

**`uncaptured` entry format (Phase 8 parses this — do NOT change):**

| Source | Format | Example |
|---|---|---|
| CES | `"ces:<ISO-date>"` — first of the ref-month | `"ces:2025-09-01"` |
| QCEW | `"qcew:<YYYY>-Q<n>"` — year-quarter | `"qcew:2025-Q4"` |

Phase 8's `watch` command parses entries as `u.startswith(f"{src}:")` then
`ref_token = u.split(":", 1)[1]`, where `_watch_snapshot_anchor` distinguishes the two by
`"-Q" in ref_token`. **Any other separator or field ordering breaks Phase 8.**

---

### Task 7.1: `PartitionCoverage`/`StoreStatus` dataclasses + `compute_status` per-partition coverage

**Files:**

- Create: `packages/nfp-vintages/src/nfp_vintages/store_status.py`
- Create: `packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py`

**Interfaces — Produces:**

- `PartitionCoverage` (frozen dataclass)
- `StoreStatus` (frozen dataclass)
- `compute_status(store_path=VINTAGE_STORE_PATH, as_of: date | None = None) -> StoreStatus`
  (fills only `per_partition` in this task; `uncaptured`/`missing_months`/`corrected` seeded
  empty — Task 7.2 populates them)

---

- [ ] **Step 1: Write the failing test** — create `test_store_status.py`:

```python
# packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py
"""Tests for the read-only ``status`` report (spec §8).

All store I/O is against a synthetic Hive-partitioned tmp_path store built by
``_write_store_rows`` below — NEVER a real MinIO store (conftest auto-loads prod
creds). ``compute_status`` reads via ``read_vintage_store`` and must never call
``transform_to_panel``.
"""

from __future__ import annotations

from datetime import date

import polars as pl
from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA
from nfp_vintages.store_status import (
    PartitionCoverage,
    StoreStatus,
    compute_status,
)


def _row(
    *,
    source: str,
    sa: bool,
    ref_date: date,
    vintage_date: date,
    revision: int = 0,
    benchmark_revision: int = 0,
    employment: float = 100.0,
    industry_code: str = "00",
    geographic_code: str = "00",
) -> dict:
    """One VINTAGE_STORE_SCHEMA row as a dict (defaults = national total headline)."""
    return {
        "geographic_type": "national",
        "geographic_code": geographic_code,
        "ownership": "total",
        "industry_type": "total",
        "industry_code": industry_code,
        "ref_date": ref_date,
        "vintage_date": vintage_date,
        "revision": revision,
        "benchmark_revision": benchmark_revision,
        "employment": employment,
        "size_class_type": None,
        "size_class_code": None,
        "source": source,
        "seasonally_adjusted": sa,
    }


def _write_store_rows(store_path, rows: list[dict]) -> None:
    """Write rows into the Hive layout the store reader expects.

    Partitions on (source, seasonally_adjusted); the partition columns are
    encoded in the directory names (Hive), so they are dropped from the file.
    """
    df = pl.DataFrame(rows, schema=VINTAGE_STORE_SCHEMA)
    for (source, sa), part in df.group_by(["source", "seasonally_adjusted"]):
        part_dir = (
            store_path
            / f"source={source}"
            / f"seasonally_adjusted={str(sa).lower()}"
        )
        part_dir.mkdir(parents=True, exist_ok=True)
        part.drop("source", "seasonally_adjusted").write_parquet(
            part_dir / "part-0.parquet"
        )


def test_compute_status_partition_coverage(tmp_path):
    """One PartitionCoverage per (source, sa); raw row presence, sentinel counts."""
    rows = [
        # CES SA: two months, the second is the Oct-2025 -1 sentinel slot.
        _row(
            source="ces",
            sa=True,
            ref_date=date(2025, 9, 1),
            vintage_date=date(2025, 11, 20),
            employment=159000.0,
        ),
        _row(
            source="ces",
            sa=True,
            ref_date=date(2025, 10, 1),
            vintage_date=date(2025, 12, 16),
            employment=-1.0,
        ),
        # QCEW NSA: one quarter.
        _row(
            source="qcew",
            sa=False,
            ref_date=date(2025, 1, 1),
            vintage_date=date(2025, 9, 1),
            employment=140000.0,
        ),
    ]
    _write_store_rows(tmp_path, rows)

    status = compute_status(tmp_path, as_of=date(2026, 1, 12))

    assert isinstance(status, StoreStatus)
    parts = {(p.source, p.seasonally_adjusted): p for p in status.per_partition}
    assert set(parts) == {("ces", True), ("qcew", False)}

    ces = parts[("ces", True)]
    assert isinstance(ces, PartitionCoverage)
    # Both CES rows counted — the -1 sentinel is NOT filtered out.
    assert ces.row_count == 2
    assert ces.earliest_ref == date(2025, 9, 1)
    assert ces.latest_ref == date(2025, 10, 1)
    assert ces.last_capture == date(2025, 12, 16)
    assert ces.distinct_vintages == 2

    qcew = parts[("qcew", False)]
    assert qcew.row_count == 1
    assert qcew.latest_ref == date(2025, 1, 1)
    assert qcew.last_capture == date(2025, 9, 1)
```

- [ ] **Step 2: Run the test, verify it fails** —
  ```bash
  uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py::test_compute_status_partition_coverage -q --no-cov
  ```
  Expected: **FAIL** with `ModuleNotFoundError: No module named 'nfp_vintages.store_status'`
  (the module does not exist yet).

- [ ] **Step 3: Implement** — create `store_status.py` with the two dataclasses and a
  `compute_status` that fills only `per_partition`. Coverage is one `read_vintage_store`
  lazy scan per `(source, sa)` partition, aggregated with Polars — **no** `transform_to_panel`,
  **no** `employment > 0` filter.

```python
# packages/nfp-vintages/src/nfp_vintages/store_status.py
"""Read-only store health + knowability report (spec §8).

Built on ``read_vintage_store`` (partition-prune + projection pushdown,
LazyFrame) ONLY — never ``transform_to_panel`` (the expensive growth/censoring
path) and never ``views.py`` (panel-grain, post-transform). Coverage is raw
row presence (no ``employment > 0`` filter) so the Oct-2025 ``-1`` "no print"
sentinel (``first_print.py:79-84``) counts as present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import polars as pl
from nfp_ingest.vintage_store import read_vintage_store
from nfp_lookups.paths import VINTAGE_STORE_PATH, is_canonical_store, is_remote

# (source, seasonally_adjusted) partitions present in the rebuilt store.
_PARTITIONS: tuple[tuple[str, bool], ...] = (
    ("ces", True),
    ("ces", False),
    ("qcew", False),
)


@dataclass(frozen=True)
class PartitionCoverage:
    """Coverage of one ``(source, seasonally_adjusted)`` store partition."""

    source: str
    seasonally_adjusted: bool
    earliest_ref: date | None
    latest_ref: date | None
    row_count: int
    last_capture: date | None
    distinct_vintages: int


@dataclass(frozen=True)
class StoreStatus:
    """The full ``status`` report — header flags, coverage, and alarms."""

    store_uri: str
    is_remote: bool
    is_canonical: bool
    per_partition: list[PartitionCoverage] = field(default_factory=list)
    uncaptured: list[str] = field(default_factory=list)
    missing_months: list[str] = field(default_factory=list)
    corrected: list[str] = field(default_factory=list)


def _partition_coverage(store_path, source: str, sa: bool) -> PartitionCoverage | None:
    """Aggregate one partition via ``read_vintage_store``; None if empty/absent."""
    lf = read_vintage_store(store_path, source=source, seasonally_adjusted=sa)
    agg = lf.select(
        pl.len().alias("row_count"),
        pl.col("ref_date").min().alias("earliest_ref"),
        pl.col("ref_date").max().alias("latest_ref"),
        pl.col("vintage_date").max().alias("last_capture"),
        pl.col("vintage_date").n_unique().alias("distinct_vintages"),
    ).collect()
    row_count = int(agg.item(0, "row_count"))
    if row_count == 0:
        return None
    return PartitionCoverage(
        source=source,
        seasonally_adjusted=sa,
        earliest_ref=agg.item(0, "earliest_ref"),
        latest_ref=agg.item(0, "latest_ref"),
        row_count=row_count,
        last_capture=agg.item(0, "last_capture"),
        distinct_vintages=int(agg.item(0, "distinct_vintages")),
    )


def compute_status(
    store_path=VINTAGE_STORE_PATH,
    as_of: date | None = None,
) -> StoreStatus:
    """Read-only coverage + knowability report for the vintage store.

    Reads via ``read_vintage_store`` only. ``as_of`` (default: today) bounds the
    forward UNCAPTURED alarm (Task 7.2). Never calls ``transform_to_panel``.
    """
    if as_of is None:
        as_of = date.today()

    per_partition: list[PartitionCoverage] = []
    for source, sa in _PARTITIONS:
        cov = _partition_coverage(store_path, source, sa)
        if cov is not None:
            per_partition.append(cov)

    return StoreStatus(
        store_uri=str(store_path),
        is_remote=is_remote(store_path),
        is_canonical=is_canonical_store(store_path),
        per_partition=per_partition,
        uncaptured=[],
        missing_months=[],
        corrected=[],
    )
```

- [ ] **Step 4: Run, verify pass** —
  ```bash
  uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py::test_compute_status_partition_coverage -q --no-cov
  ```
  Expected: **PASS**. Then run the file suite and lint:
  ```bash
  uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py -q --no-cov
  uv run ruff check packages/nfp-vintages/src/nfp_vintages/store_status.py \
      packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py
  ```
  Then run the **full nfp-vintages suite**:
  ```bash
  uv run pytest packages/nfp-vintages/ -m "not network and not slow" --no-cov -q
  ```

- [ ] **Step 5: Commit** — stage only the two new files:
  ```bash
  git add packages/nfp-vintages/src/nfp_vintages/store_status.py \
      packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py
  git commit -m "$(cat <<'EOF'
  feat(status): per-partition store coverage via read_vintage_store

  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 7.2: forward UNCAPTURED alarm + missing-month list + `format_status`

**Files:**

- Modify: `packages/nfp-vintages/src/nfp_vintages/store_status.py`
- Modify: `packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py`

**Interfaces — Produces:**

- `compute_status` extended: `uncaptured` and `missing_months` now populated
- `format_status(status: StoreStatus) -> str`

**Key design decisions:**

- CES uncaptured entries: `"ces:<YYYY-MM-DD>"` where the date is the first of the ref-month
  (e.g. `"ces:2025-09-01"`). Phase 8 parses `ref_token = u.split(":", 1)[1]` and calls
  `date.fromisoformat(ref_token)` on non-QCEW tokens.
- QCEW uncaptured entries: `"qcew:<YYYY>-Q<n>"` (e.g. `"qcew:2025-Q4"`). Phase 8 detects
  `"-Q" in ref_token` then `year_str, q_str = ref_token.split("-Q")`.
- `_missing_headline_months` is raw row presence over CES SA headline (`geo 00`,
  `industry_code in {"00", "05"}`). No `employment > 0` filter — the Oct-2025 `-1` sentinel
  row counts present. Interior gaps (months in `[earliest_ref, latest_ref]` with no row) are
  reported; known shutdown months get an annotation suffix rather than a bare flag.
- The `as_of` default in `compute_status` is `date.today()`.
- `format_status` renders a LOCAL-FALLBACK warning when `status.is_remote` is False, covering
  the `.env`-not-loaded gotcha.

---

- [ ] **Step 1: Write the failing tests** — extend `test_store_status.py`. Add the new imports
  to the **top-of-file import block** (sorted into the correct groups):

```python
# full updated import block for test_store_status.py after Task 7.2
from __future__ import annotations

from datetime import date

import polars as pl
from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA
from nfp_vintages.store_status import (
    PartitionCoverage,
    StoreStatus,
    compute_status,
    format_status,
)
```

Then append these tests after the existing `test_compute_status_partition_coverage`:

```python
def test_compute_status_flags_uncaptured_ces_month(tmp_path):
    """Store lags the calendar: published-but-absent CES ref-months are flagged."""
    # Store stops at Aug-2025; as-of 2026-01-12 → Sep/Oct/Nov rev0 are out by then.
    rows = [
        _row(
            source="ces",
            sa=True,
            ref_date=date(2025, 7, 1),
            vintage_date=date(2025, 8, 1),
            employment=158000.0,
        ),
        _row(
            source="ces",
            sa=True,
            ref_date=date(2025, 8, 1),
            vintage_date=date(2025, 9, 5),
            employment=158200.0,
        ),
    ]
    _write_store_rows(tmp_path, rows)

    status = compute_status(tmp_path, as_of=date(2026, 1, 12))

    joined = " ".join(status.uncaptured)
    # Entries use "ces:<ISO-date>" format (Phase 8 contract).
    assert "ces:" in joined
    # At least Sep-2025 should be reported uncaptured (rev0 published ~Oct-2025).
    assert "2025-09-01" in joined


def test_compute_status_uncaptured_uses_colon_format(tmp_path):
    """uncaptured entries must be 'src:<ref_token>' (Phase 8 parse contract)."""
    rows = [
        _row(
            source="ces",
            sa=True,
            ref_date=date(2025, 8, 1),
            vintage_date=date(2025, 9, 5),
            employment=158200.0,
        ),
    ]
    _write_store_rows(tmp_path, rows)
    status = compute_status(tmp_path, as_of=date(2026, 1, 12))
    for entry in status.uncaptured:
        src, _, ref = entry.partition(":")
        assert src in {"ces", "qcew"}, f"bad source in {entry!r}"
        assert ref, f"empty ref_token in {entry!r}"


def test_oct_2025_sentinel_not_flagged_missing(tmp_path):
    """A -1 sentinel row at Oct-2025 counts as present (raw row presence)."""
    rows = [
        _row(
            source="ces",
            sa=True,
            ref_date=date(2025, 9, 1),
            vintage_date=date(2025, 11, 20),
            employment=159000.0,
        ),
        # The shutdown "no print" sentinel: literal -1.0 at the Oct-2025 slot.
        _row(
            source="ces",
            sa=True,
            ref_date=date(2025, 10, 1),
            vintage_date=date(2025, 12, 16),
            employment=-1.0,
        ),
        _row(
            source="ces",
            sa=True,
            ref_date=date(2025, 11, 1),
            vintage_date=date(2025, 12, 16),
            employment=159100.0,
        ),
    ]
    _write_store_rows(tmp_path, rows)

    status = compute_status(tmp_path, as_of=date(2026, 1, 12))

    # Oct-2025 has a (sentinel) row → NOT an interior hole.
    missing = " ".join(status.missing_months)
    assert "2025-10" not in missing


def test_format_status_local_fallback_warning(tmp_path):
    """A local (non-remote) store renders the .env LOCAL-FALLBACK warning."""
    rows = [
        _row(
            source="ces",
            sa=True,
            ref_date=date(2025, 9, 1),
            vintage_date=date(2025, 11, 20),
            employment=159000.0,
        ),
    ]
    _write_store_rows(tmp_path, rows)

    text = format_status(compute_status(tmp_path, as_of=date(2025, 12, 12)))

    assert "LOCAL FALLBACK" in text
    assert "NFP_STORE_URI" in text
    assert "ces" in text
```

- [ ] **Step 2: Run the tests, verify they fail** —
  ```bash
  uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py -q --no-cov \
      -k "uncaptured or sentinel or format"
  ```
  Expected: **FAIL** — `test_compute_status_flags_uncaptured_ces_month` fails because
  `uncaptured` is still empty; `test_format_status_local_fallback_warning` fails with
  `ImportError: cannot import name 'format_status'`.

- [ ] **Step 3: Implement** — add imports, helpers, and `format_status` to `store_status.py`.

  **3a. Add imports at module top** (after the existing `from nfp_lookups.paths …` line,
  sorted into the same `nfp_lookups` import group):

```python
from nfp_lookups.revision_schedules import get_ces_vintage_date, get_qcew_vintage_date
```

  **3b. Add module-level helpers** (insert before `compute_status`):

```python
# Known shutdown months (employment -1 sentinel in the store): these are expected
# interior "gaps" caused by BLS shutdown delays, not missing captures.
_KNOWN_SHUTDOWN_MONTHS: frozenset[date] = frozenset({
    date(2025, 10, 1),  # Oct-2025: BLS shutdown delayed Sep+Oct CES prints
})


def _next_month(d: date) -> date:
    """First of the month after *d*."""
    return date(d.year + 1, 1, 1) if d.month == 12 else date(d.year, d.month + 1, 1)


def _ces_uncaptured(latest_ref: date | None, as_of: date) -> list[str]:
    """CES ref-months whose rev0 was published <= as_of but are absent from the store.

    Returns entries in the format ``"ces:<YYYY-MM-DD>"`` (Phase 8 contract).
    The date is the first of the ref-month so Phase 8 can reconstruct it via
    ``date.fromisoformat(ref_token)``.
    """
    if latest_ref is None:
        return []
    out: list[str] = []
    candidate = _next_month(latest_ref)
    while candidate <= as_of:
        try:
            v0 = get_ces_vintage_date(candidate, 0)
        except ValueError:
            break
        if v0 <= as_of:
            out.append(f"ces:{candidate.isoformat()}")
        candidate = _next_month(candidate)
    return out


def _qcew_uncaptured(latest_ref: date | None, as_of: date) -> list[str]:
    """QCEW quarters whose rev0 was published <= as_of but are absent from the store.

    Returns entries in the format ``"qcew:<YYYY>-Q<n>"`` (Phase 8 contract).
    Phase 8 detects ``"-Q" in ref_token`` then splits on ``"-Q"`` to recover
    year and quarter number.
    """
    if latest_ref is None:
        return []
    out: list[str] = []
    # Advance to the first month of the quarter after latest_ref's quarter.
    q_start_month = ((latest_ref.month - 1) // 3) * 3 + 1
    year, month = latest_ref.year, q_start_month + 3
    if month > 12:
        year, month = year + 1, month - 12
    while date(year, month, 1) <= as_of:
        q_num = (month - 1) // 3 + 1
        ref_quarter = f"Q{q_num}"
        try:
            v0 = get_qcew_vintage_date(ref_quarter, year, 0)
        except ValueError:
            break
        if v0 <= as_of:
            out.append(f"qcew:{year}-Q{q_num}")
        month += 3
        if month > 12:
            year, month = year + 1, month - 12
    return out


def _missing_headline_months(store_path) -> list[str]:
    """Interior CES-SA ref-month gaps over the headline series (geo 00, ind 00/05).

    Raw row presence (no ``employment > 0`` filter) so a -1 sentinel row counts
    as present and known shutdown months are annotated rather than flagged as errors.
    A month is "present" if either the total (``00``) or private (``05``) headline
    row exists. Returns ``"YYYY-MM"`` strings; shutdown months are suffixed with
    ``" [known-shutdown]"`` instead of a bare flag.
    """
    lf = read_vintage_store(
        store_path,
        source="ces",
        seasonally_adjusted=True,
        geographic_type="national",
        geographic_code="00",
    )
    present = (
        lf.filter(pl.col("industry_code").is_in(["00", "05"]))
        .select(pl.col("ref_date").dt.truncate("1mo"))
        .unique()
        .collect()
        .get_column("ref_date")
        .sort()
        .to_list()
    )
    if len(present) < 2:
        return []
    have = set(present)
    out: list[str] = []
    cursor = present[0]
    last = present[-1]
    while cursor < last:
        cursor = _next_month(cursor)
        if cursor not in have:
            label = f"{cursor:%Y-%m}"
            if cursor in _KNOWN_SHUTDOWN_MONTHS:
                label += " [known-shutdown]"
            out.append(label)
    return out


def format_status(status: StoreStatus) -> str:
    """Render a StoreStatus as a human-readable multi-line report."""
    lines: list[str] = []
    flags = []
    if status.is_remote:
        flags.append("REMOTE")
    else:
        flags.append("LOCAL")
    if status.is_canonical:
        flags.append("CANONICAL")
    lines.append(f"store: {status.store_uri}  [{'/'.join(flags)}]")
    if not status.is_remote:
        lines.append(
            "  WARNING: LOCAL FALLBACK — NFP_STORE_URI unset; reading the "
            "local data/store, not the canonical S3 store."
        )

    lines.append("")
    lines.append("coverage (source, seasonally_adjusted):")
    for p in status.per_partition:
        lines.append(
            f"  {p.source:<5} sa={str(p.seasonally_adjusted):<5} "
            f"rows={p.row_count:>8,} "
            f"ref=[{p.earliest_ref}..{p.latest_ref}] "
            f"last_capture={p.last_capture} vintages={p.distinct_vintages}"
        )

    if status.uncaptured:
        lines.append("")
        lines.append("UNCAPTURED (published per calendar, absent from store):")
        lines.extend(f"  {u}" for u in status.uncaptured)

    if status.missing_months:
        lines.append("")
        lines.append("missing headline months (interior gaps):")
        lines.extend(f"  {m}" for m in status.missing_months)

    if status.corrected:
        lines.append("")
        lines.append("CORRECTED-LEVEL (incoming != stored employment):")
        lines.extend(f"  {c}" for c in status.corrected)

    return "\n".join(lines)
```

  **3c. Replace the `return StoreStatus(...)` block** in `compute_status` with alarm
  computation (the final block of the function body, replacing the seed-empty return from
  Task 7.1):

```python
    # --- Task 7.2: alarm computation ---
    by_key = {(p.source, p.seasonally_adjusted): p for p in per_partition}
    ces_sa = by_key.get(("ces", True))
    qcew_nsa = by_key.get(("qcew", False))

    uncaptured: list[str] = []
    uncaptured.extend(_ces_uncaptured(ces_sa.latest_ref if ces_sa else None, as_of))
    uncaptured.extend(_qcew_uncaptured(qcew_nsa.latest_ref if qcew_nsa else None, as_of))

    missing_months = _missing_headline_months(store_path)

    return StoreStatus(
        store_uri=str(store_path),
        is_remote=is_remote(store_path),
        is_canonical=is_canonical_store(store_path),
        per_partition=per_partition,
        uncaptured=uncaptured,
        missing_months=missing_months,
        corrected=[],
    )
```

- [ ] **Step 4: Run, verify pass** —
  ```bash
  uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py -q --no-cov
  ```
  Expected: **PASS** (all tests in the file). Then lint:
  ```bash
  uv run ruff check packages/nfp-vintages/src/nfp_vintages/store_status.py \
      packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py
  ```
  Then run the **full nfp-vintages suite**:
  ```bash
  uv run pytest packages/nfp-vintages/ -m "not network and not slow" --no-cov -q
  ```

- [ ] **Step 5: Commit** — stage only the two modified files:
  ```bash
  git add packages/nfp-vintages/src/nfp_vintages/store_status.py \
      packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py
  git commit -m "$(cat <<'EOF'
  feat(status): forward UNCAPTURED alarm, missing-month list, format_status

  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 7.3: `status` Typer command (deferred imports)

**Files:**

- Modify: `packages/nfp-vintages/src/nfp_vintages/__main__.py`
- Modify: `packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py`

**Interfaces — Produces:**

- `alt-nfp status [--as-of YYYY-MM-DD] [--store URI]` command
  - `--as-of`: knowability cutoff for the UNCAPTURED alarm (defaults to today inside
    `compute_status`)
  - `--store`: override the store URI/path (default: `VINTAGE_STORE_PATH`); accepts both
    local paths and `s3://` / `s3a://` URIs

**Critical: all imports inside the command body must be deferred.** The `main` callback
(`__main__.py:24-33`) runs `load_dotenv()` before any subcommand body executes. Because
`paths.py` binds `VINTAGE_STORE_PATH` at import time (`paths.py:152`), importing `paths` or
`store_status` (which imports `paths`) at module top would resolve `VINTAGE_STORE_PATH` before
`.env` is loaded — causing the command to read the empty local `data/store` even when
`NFP_STORE_URI` is set. `Path` (stdlib) does not read env and may stay at module top.

---

- [ ] **Step 1: Write the failing test** — extend `test_store_status.py`. Add the new imports
  to the **top-of-file import block** (append at the end of the third-party block; note
  `typer.testing` is third-party, `nfp_vintages.__main__` is first-party):

```python
# full updated import block for test_store_status.py after Task 7.3
from __future__ import annotations

from datetime import date

import polars as pl
from typer.testing import CliRunner
from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA
from nfp_vintages.__main__ import app
from nfp_vintages.store_status import (
    PartitionCoverage,
    StoreStatus,
    compute_status,
    format_status,
)
```

Then append this test:

```python
def test_status_command_renders_report(tmp_path):
    """`alt-nfp status --store <tmp> --as-of D` prints the coverage report."""
    rows = [
        _row(
            source="ces",
            sa=True,
            ref_date=date(2025, 9, 1),
            vintage_date=date(2025, 11, 20),
            employment=159000.0,
        ),
        _row(
            source="qcew",
            sa=False,
            ref_date=date(2025, 1, 1),
            vintage_date=date(2025, 9, 1),
            employment=140000.0,
        ),
    ]
    _write_store_rows(tmp_path, rows)

    result = CliRunner().invoke(
        app,
        ["status", "--store", str(tmp_path), "--as-of", "2025-12-12"],
    )

    assert result.exit_code == 0, result.output
    assert "coverage" in result.output
    assert "ces" in result.output
    assert "qcew" in result.output
```

- [ ] **Step 2: Run the test, verify it fails** —
  ```bash
  uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py::test_status_command_renders_report -q --no-cov
  ```
  Expected: **FAIL** — Typer exits non-zero, output contains `No such command 'status'`.

- [ ] **Step 3: Implement** — insert the `status` command into `__main__.py` immediately before
  the existing `snapshot` command (currently at line 184). All imports inside the body are
  **deferred**. `Path` (stdlib) stays at module top and may be used in the body without a
  deferred import.

```python
@app.command()
def status(
    as_of: str | None = typer.Option(
        None, "--as-of", help="Knowability cutoff for the UNCAPTURED alarm (YYYY-MM-DD)."
    ),
    store: str | None = typer.Option(
        None, "--store", help="Override the store URI/path (default: VINTAGE_STORE_PATH)."
    ),
) -> None:
    """Read-only store coverage + 'what's uncaptured' report (spec §8)."""
    from datetime import date as _date

    from nfp_lookups.paths import VINTAGE_STORE_PATH
    from nfp_vintages.store_status import compute_status, format_status

    if store is not None:
        if store.startswith(("s3://", "s3a://")):
            from upath import UPath

            store_path = UPath(store)
        else:
            store_path = Path(store)
    else:
        store_path = VINTAGE_STORE_PATH

    as_of_date = _date.fromisoformat(as_of) if as_of is not None else None
    report = compute_status(store_path, as_of=as_of_date)
    print(format_status(report))
```

  Note: `Path` is already imported at module top in `__main__.py` (`from pathlib import Path`
  at line 15 — verify before adding a duplicate). No new module-top imports are needed for
  this command.

- [ ] **Step 4: Run, verify pass** —
  ```bash
  uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py::test_status_command_renders_report -q --no-cov
  ```
  Expected: **PASS**. Then run the full test file + the existing CLI snapshot test + lint:
  ```bash
  uv run pytest \
      packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py \
      packages/nfp-vintages/src/nfp_vintages/tests/test_cli_snapshot.py \
      -q --no-cov
  uv run ruff check packages/nfp-vintages/src/nfp_vintages/__main__.py
  ```
  Then the **full nfp-vintages suite**:
  ```bash
  uv run pytest packages/nfp-vintages/ -m "not network and not slow" --no-cov -q
  ```

- [ ] **Step 5: Commit** — stage only the two modified files:
  ```bash
  git add packages/nfp-vintages/src/nfp_vintages/__main__.py \
      packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py
  git commit -m "$(cat <<'EOF'
  feat(cli): add alt-nfp status command (deferred imports)

  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Phase 7 completion checklist

After all three tasks are committed:

- [ ] `packages/nfp-vintages/src/nfp_vintages/store_status.py` exists with correct signatures
- [ ] `packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py` has ≥ 6 tests
      (7.1: 1, 7.2: 4, 7.3: 1)
- [ ] `packages/nfp-vintages/src/nfp_vintages/__main__.py` has `status` command registered
- [ ] `uv run pytest packages/nfp-vintages/ -m "not network and not slow" --no-cov` passes
- [ ] `uv run ruff check packages/nfp-vintages/` clean
- [ ] `uncaptured` entry format is `"<src>:<ref_token>"` — Phase 8 `watch` can parse it
- [ ] No write to a live MinIO store in any test path
## Phase 8 — `watch` command + `nfp_download/release_dates/feed.py`

**Spec:** §9 (`specs/cli_production_workflow.md`). **Consumes (must be DONE before Task 8.3):**
Phase 5's `_run_update`/`_run_snapshot` helpers and Phase 7's `compute_status`/`StoreStatus`.
**Provides:** `nfp_download/release_dates/feed.py` (Tasks 8.1–8.2) + the `alt-nfp watch` Typer
command in `nfp_vintages/__main__.py` (Task 8.3).

**What this phase builds:** a feed-driven trigger for a daily cron. The BLS publishes RSS feeds
at fixed URLs announcing each release; `watch` fetches the appropriate feed, asks the store
whether that release's ref-period is already captured, and — if it is not — calls `_run_update`.
With `--snapshot` it additionally runs `_run_snapshot` at the day-12 anchor of the captured
ref-month (never the raw pubDate, because `_run_snapshot` enforces day-12, §4a).

### Cross-phase contract (pinned — do NOT deviate)

```python
# Phase 5 (nfp_vintages/__main__.py) — consume these exact signatures:
def _run_update(as_of: date, *, only: str | None = None,
                refresh_calendar: bool = True, store_path=None) -> None: ...
def _run_snapshot(as_of: date, grid_end: date | None = None) -> None: ...

# Phase 7 (nfp_vintages/store_status.py) — consume:
def compute_status(store_path=VINTAGE_STORE_PATH, as_of: date | None = None) -> StoreStatus: ...
# StoreStatus.uncaptured: list[str]  — entries like "ces:2025-05-01" or "qcew:2025-Q1"

# Phase 8 (nfp_download/release_dates/feed.py) — define:
@dataclass
class FeedItem:
    title: str
    pub_date: date    # calendar date only (not datetime), parsed from RFC-822 pubDate
    guid: str

def parse_feed(xml: str) -> list[FeedItem]: ...
def fetch_feed(url: str, *, session=None) -> list[FeedItem]: ...
```

**Feed URLs (hard-coded constants in `feed.py`):**

```
EMPSIT_FEED_URL = "https://www.bls.gov/feed/empsit.rss"   # CES
CEWQTR_FEED_URL = "https://www.bls.gov/feed/cewqtr.rss"   # QCEW
```

**Impersonating session factory (current source — verified):**

- `packages/nfp-download/src/nfp_download/release_dates/scraper.py:191–202`
  `create_session(timeout=30.0) -> AsyncSession` — returns a `curl_cffi.requests.AsyncSession`
  with `impersonate='chrome'`. This is an **async** session used as `async with ... await
  session.get(...)`.
- `packages/nfp-download/src/nfp_download/client.py:67–92`
  `create_impersonating_session(*, timeout=...) -> curl_cffi.requests.Session` — the **sync**
  counterpart used by nfp-vintages. **Do NOT use** this in `feed.py` — `feed.py` reuses the
  **async** `create_session` from `scraper.py` so it rides the same session that the calendar
  scraper uses (correct pairing per CLAUDE.md).

**Drift reconciliation (old plan → new plan):**

| Old plan | Contract | Fix |
|---|---|---|
| `_run_update(as_of=pub.isoformat(), only=src)` | `as_of: date` | Pass `pub` (a `date`), not `.isoformat()` |
| `_run_snapshot(as_of=anchor)` where anchor is a string | `as_of: date` | Build a `date(y, m, 12)`, pass it directly |
| `_snapshot_anchor()` returns `str` (.isoformat()) | helper returns `date` | Return `date(y, m, 12)` |
| Test monkeypatches assert `as_of == "2025-06-06"` (str) | `as_of` is `date` | Assertions use `date(2025, 6, 6)` |
| Test asserts `snaps[0]["as_of"] == "2025-05-12"` (str) | `as_of` is `date` | Assertion uses `date(2025, 5, 12)` |
| `no_refresh_calendar=` kwarg | `refresh_calendar: bool = True` | Use `refresh_calendar=False` to skip |

---

### Task 8.1: `feed.py` — pure `parse_feed(xml) -> list[FeedItem]`

**Files:**
- Create: `packages/nfp-download/src/nfp_download/release_dates/feed.py`
- Create: `packages/nfp-download/src/nfp_download/tests/release_dates/test_feed.py`
  (note: the `release_dates/` test directory already exists — add this file)

`parse_feed` is the pure, no-network half: parse a BLS RSS 2.0 feed XML string into
`FeedItem`s. `pubDate` is RFC-822 (e.g. `Fri, 06 Jun 2025 08:30:00 -0400`), parsed with
`email.utils.parsedate_to_datetime(...).date()` — **not** a hand-rolled regex. The fixture
below is a realistic synthetic RSS 2.0 document matching the BLS empsit/cewqtr feed shape (BLS
RSS items carry `<title>`, `<pubDate>`, and `<guid>`; `<link>` is also present but not part of
`FeedItem`). We cannot live-fetch in red phase — `www.bls.gov` intermittently 403s a plain GET
(the Akamai TLS-fingerprint block that forces the curl_cffi session in Task 8.2).

- [ ] **Step 1: Write the failing test** — create
  `packages/nfp-download/src/nfp_download/tests/release_dates/test_feed.py`:

```python
"""Unit tests for feed.parse_feed (pure, no network).

Fixture is standard RSS 2.0 matching the BLS empsit/cewqtr feed shape: each
<item> carries <title>, an RFC-822 <pubDate>, and a <guid>. We could not
live-capture in red phase — www.bls.gov intermittently 403s a plain GET (the
Akamai TLS block that forces fetch_feed's curl_cffi session; Task 8.2).
pubDate format is pinned to RFC-822.
"""

from __future__ import annotations

from datetime import date

from nfp_download.release_dates.feed import FeedItem, parse_feed

EMPSIT_RSS = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Employment Situation</title>
    <link>https://www.bls.gov/news.release/empsit.htm</link>
    <item>
      <title>Employment Situation Summary</title>
      <link>https://www.bls.gov/news.release/archives/empsit_06062025.htm</link>
      <pubDate>Fri, 06 Jun 2025 08:30:00 -0400</pubDate>
      <guid>https://www.bls.gov/news.release/archives/empsit_06062025.htm</guid>
    </item>
    <item>
      <title>Employment Situation Summary</title>
      <link>https://www.bls.gov/news.release/archives/empsit_05022025.htm</link>
      <pubDate>Fri, 02 May 2025 08:30:00 -0400</pubDate>
      <guid>https://www.bls.gov/news.release/archives/empsit_05022025.htm</guid>
    </item>
  </channel>
</rss>
"""


class TestParseFeed:
    def test_returns_feed_items(self):
        items = parse_feed(EMPSIT_RSS)
        assert len(items) == 2
        assert all(isinstance(it, FeedItem) for it in items)

    def test_first_item_fields(self):
        items = parse_feed(EMPSIT_RSS)
        first = items[0]
        assert first.title == "Employment Situation Summary"
        assert first.pub_date == date(2025, 6, 6)
        assert first.guid == "https://www.bls.gov/news.release/archives/empsit_06062025.htm"

    def test_pubdate_parsed_as_date_object(self):
        items = parse_feed(EMPSIT_RSS)
        assert all(isinstance(it.pub_date, date) for it in items)
        assert items[1].pub_date == date(2025, 5, 2)

    def test_items_in_feed_order_newest_first(self):
        items = parse_feed(EMPSIT_RSS)
        assert items[0].pub_date >= items[1].pub_date

    def test_empty_channel_returns_empty_list(self):
        empty = '<?xml version="1.0"?><rss version="2.0"><channel/></rss>'
        assert parse_feed(empty) == []

    def test_item_missing_pubdate_is_skipped(self):
        no_date = """\
<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item><title>No date</title><guid>g1</guid></item>
</channel></rss>"""
        assert parse_feed(no_date) == []

    def test_item_missing_guid_is_skipped(self):
        no_guid = """\
<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>No guid</title>
    <pubDate>Fri, 06 Jun 2025 08:30:00 -0400</pubDate>
  </item>
</channel></rss>"""
        assert parse_feed(no_guid) == []
```

- [ ] **Step 2: Run the test, verify it fails** —
  ```
  uv run pytest packages/nfp-download/src/nfp_download/tests/release_dates/test_feed.py \
    -q --no-cov
  ```
  Expected failure: `ModuleNotFoundError: No module named 'nfp_download.release_dates.feed'`
  (the file does not exist yet).

- [ ] **Step 3: Implement** `feed.py` with `FeedItem`, `parse_feed`, and the URL constants
  (`fetch_feed` lands in Task 8.2). Create
  `packages/nfp-download/src/nfp_download/release_dates/feed.py`:

```python
"""BLS release RSS feed — fetch + parse.

The feed answers the production question the calendar can only predict and
shutdowns can delay: "is the release out *now*?". ``parse_feed`` is pure (no
network); ``fetch_feed`` reuses the scraper's curl_cffi Chrome-impersonating
session (www.bls.gov/Akamai 403s a plain httpx GET — memory
``bls-akamai-blocking-intermittent``).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from email.utils import parsedate_to_datetime

EMPSIT_FEED_URL = "https://www.bls.gov/feed/empsit.rss"
CEWQTR_FEED_URL = "https://www.bls.gov/feed/cewqtr.rss"


@dataclass
class FeedItem:
    """One RSS <item>: release title, publication date, and unique id."""

    title: str
    pub_date: date
    guid: str


def _text(item: ET.Element, tag: str) -> str | None:
    """Return the stripped text of the first child ``tag``, or None."""
    child = item.find(tag)
    if child is None or child.text is None:
        return None
    return child.text.strip()


def parse_feed(xml: str) -> list[FeedItem]:
    """Parse a BLS RSS 2.0 feed into FeedItems (pure, no network).

    ``pubDate`` is RFC-822 (e.g. ``Fri, 06 Jun 2025 08:30:00 -0400``); the
    calendar date is extracted. Items missing a title, a parseable pubDate, or
    a guid are skipped — a malformed item should not sink the poll.

    Parameters
    ----------
    xml : str
        Raw RSS feed body.

    Returns
    -------
    list[FeedItem]
        One per well-formed <item>, in feed order (BLS lists newest first).
    """
    root = ET.fromstring(xml)
    items: list[FeedItem] = []
    for item in root.iter("item"):
        title = _text(item, "title")
        raw_date = _text(item, "pubDate")
        guid = _text(item, "guid")
        if title is None or raw_date is None or guid is None:
            continue
        try:
            pub = parsedate_to_datetime(raw_date).date()
        except (TypeError, ValueError):
            continue
        items.append(FeedItem(title=title, pub_date=pub, guid=guid))
    return items
```

- [ ] **Step 4: Run, verify pass** —
  ```
  uv run pytest packages/nfp-download/src/nfp_download/tests/release_dates/test_feed.py \
    -q --no-cov
  ```
  Expected: **7 tests pass**. Then run the full release_dates suite and lint:
  ```
  uv run pytest packages/nfp-download/src/nfp_download/tests/release_dates/ \
    -q --no-cov -m "not network"
  uv run ruff check \
    packages/nfp-download/src/nfp_download/release_dates/feed.py \
    packages/nfp-download/src/nfp_download/tests/release_dates/test_feed.py
  ```

- [ ] **Step 5: Commit** —
  ```
  git add \
    packages/nfp-download/src/nfp_download/release_dates/feed.py \
    packages/nfp-download/src/nfp_download/tests/release_dates/test_feed.py
  git commit -m "$(cat <<'EOF'
  feat(download): add release-feed parse_feed + FeedItem (RSS 2.0, RFC-822 pubDate)

  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 8.2: `feed.py` — networked `fetch_feed(url, *, session=None)`

**Files:**
- Modify: `packages/nfp-download/src/nfp_download/release_dates/feed.py`
- Modify: `packages/nfp-download/src/nfp_download/tests/release_dates/test_feed.py` (extend)

`fetch_feed` is the network half. It reuses `create_session` from `scraper.py` — an **async**
`curl_cffi.requests.AsyncSession` factory (`scraper.py:191–202`). The contract declares
`fetch_feed` as **sync** (`-> list[FeedItem]`), so it wraps an async helper in `asyncio.run`,
mirroring how the old calendar CLI drove the async scraper. The session parameter is typed as
`curl_cffi.requests.AsyncSession | None` and is optional; when `None`, the function opens and
closes its own session.

There are two tests for this task:

1. A **`@pytest.mark.network` live smoke test** — confirms the real feed is reachable and
   returns parseable `FeedItem`s. Deselected in the default suite (`-m "not network"`).
2. A **monkeypatched unit test** — patches `create_session` with a fake async context manager
   whose `get()` returns a minimal RSS response; exercises the call path without the network.
   This test runs in the default suite.

- [ ] **Step 1: Write the failing tests** — extend `test_feed.py` with:

```python
# ── at the top of test_feed.py, add these imports (sorted with existing): ──
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nfp_download.release_dates.feed import EMPSIT_FEED_URL, FeedItem, fetch_feed

# ── network test ──────────────────────────────────────────────────────────

@pytest.mark.network
class TestFetchFeedNetwork:
    def test_fetch_empsit_returns_feed_items(self):
        items = fetch_feed(EMPSIT_FEED_URL)
        assert isinstance(items, list)
        assert items, "empsit feed should publish at least one item"
        assert all(isinstance(it, FeedItem) for it in items)
        # BLS lists newest first.
        assert items[0].pub_date >= items[-1].pub_date


# ── monkeypatched unit test (no network) ──────────────────────────────────

_MINIMAL_RSS = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>Employment Situation Summary</title>
      <pubDate>Fri, 06 Jun 2025 08:30:00 -0400</pubDate>
      <guid>https://www.bls.gov/news.release/archives/empsit_06062025.htm</guid>
    </item>
  </channel>
</rss>
"""


class TestFetchFeedUnit:
    def test_fetch_feed_calls_create_session(self):
        """fetch_feed drives an async session; monkeypatch create_session."""
        fake_resp = MagicMock()
        fake_resp.text = _MINIMAL_RSS
        fake_resp.raise_for_status = MagicMock()

        fake_session = AsyncMock()
        fake_session.get = AsyncMock(return_value=fake_resp)
        fake_session.__aenter__ = AsyncMock(return_value=fake_session)
        fake_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "nfp_download.release_dates.feed.create_session",
            return_value=fake_session,
        ):
            items = fetch_feed(EMPSIT_FEED_URL)

        assert len(items) == 1
        assert items[0].pub_date == date(2025, 6, 6)
        fake_session.get.assert_awaited_once_with(EMPSIT_FEED_URL)
```

  Note: the `date` import is already at the top of the test file (added in Task 8.1). The new
  imports (`asyncio`, `AsyncMock`, `MagicMock`, `patch`, `pytest`, `EMPSIT_FEED_URL`,
  `fetch_feed`) must be added in sorted import order (I001) at the top of the file, not
  mid-file. The `TestFetchFeedNetwork` and `TestFetchFeedUnit` classes are new class blocks
  after the existing `TestParseFeed` class.

- [ ] **Step 2: Run the test, verify it fails** —
  ```
  uv run pytest \
    "packages/nfp-download/src/nfp_download/tests/release_dates/test_feed.py::TestFetchFeedUnit" \
    -q --no-cov
  ```
  Expected failure: `ImportError: cannot import name 'fetch_feed'` (only `parse_feed` exists
  after Task 8.1). The network test class is not run here (no `-m network`).

- [ ] **Step 3: Implement** — add the async helper and `fetch_feed` to `feed.py`. Insert
  these additions after the `parse_feed` function (new imports must go at the module top, after
  the existing `from __future__` and stdlib imports, in sorted order):

  **New imports to add at the top of `feed.py`** (after `from email.utils import ...`):

  ```python
  import asyncio

  from nfp_download.release_dates.scraper import create_session
  ```

  **New functions to append after `parse_feed`**:

  ```python
  async def _fetch_feed_async(url: str, session=None) -> list[FeedItem]:
      """Fetch + parse one feed URL; reuse ``session`` if given, else open one.

      Parameters
      ----------
      url : str
          Feed URL (``EMPSIT_FEED_URL`` or ``CEWQTR_FEED_URL``).
      session : curl_cffi.requests.AsyncSession or None
          Reuse an open async session (e.g. polling both feeds in one run);
          when ``None``, a session is opened and closed for this call.
      """
      if session is not None:
          resp = await session.get(url)
          resp.raise_for_status()
          return parse_feed(resp.text)
      async with create_session() as owned:
          resp = await owned.get(url)
          resp.raise_for_status()
          return parse_feed(resp.text)


  def fetch_feed(url: str, *, session=None) -> list[FeedItem]:
      """Fetch and parse a BLS RSS feed (requires network).

      Transport is the scraper's curl_cffi Chrome-impersonating
      :func:`nfp_download.release_dates.scraper.create_session` — www.bls.gov
      sits behind Akamai TLS fingerprinting, so a plain httpx GET intermittently
      403s (memory ``bls-akamai-blocking-intermittent``). The session is an async
      curl_cffi ``AsyncSession``; this sync wrapper drives it via ``asyncio.run``.

      Parameters
      ----------
      url : str
          Feed URL (``EMPSIT_FEED_URL`` or ``CEWQTR_FEED_URL``).
      session : curl_cffi.requests.AsyncSession or None
          Reuse an open async session (e.g. polling both feeds in one invocation);
          when ``None``, a session is opened and closed for this call.

      Returns
      -------
      list[FeedItem]
          One per well-formed ``<item>``, newest first (BLS feed order).
      """
      return asyncio.run(_fetch_feed_async(url, session=session))
  ```

- [ ] **Step 4: Run, verify pass** — unit test (no network):
  ```
  uv run pytest \
    "packages/nfp-download/src/nfp_download/tests/release_dates/test_feed.py::TestFetchFeedUnit" \
    -q --no-cov
  ```
  Expected: **1 test passes**. If network is available, also confirm the live test:
  ```
  uv run pytest \
    "packages/nfp-download/src/nfp_download/tests/release_dates/test_feed.py::TestFetchFeedNetwork" \
    -q --no-cov -m network
  ```
  Then run the full non-network release_dates suite and lint:
  ```
  uv run pytest packages/nfp-download/src/nfp_download/tests/release_dates/ \
    -q --no-cov -m "not network"
  uv run ruff check \
    packages/nfp-download/src/nfp_download/release_dates/feed.py \
    packages/nfp-download/src/nfp_download/tests/release_dates/test_feed.py
  ```

- [ ] **Step 5: Commit** —
  ```
  git add \
    packages/nfp-download/src/nfp_download/release_dates/feed.py \
    packages/nfp-download/src/nfp_download/tests/release_dates/test_feed.py
  git commit -m "$(cat <<'EOF'
  feat(download): add fetch_feed reusing scraper curl_cffi async session (network)

  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 8.3: `alt-nfp watch` command — trigger-on-new + day-12 snapshot anchor

**Files:**
- Modify: `packages/nfp-vintages/src/nfp_vintages/__main__.py`
- Create: `packages/nfp-vintages/src/nfp_vintages/tests/test_cli_watch.py`

**Prerequisites:** Phase 5's `_run_update(as_of: date, *, only, refresh_calendar, store_path)`
and `_run_snapshot(as_of: date, ...)` are in `__main__.py`; Phase 7's `compute_status` +
`StoreStatus` (with `uncaptured: list[str]`) are in `nfp_vintages/store_status.py`.

`watch` polls the BLS RSS feed per `--source`, takes the newest `FeedItem`, and calls
`compute_status(as_of=pub)` to decide whether that source's latest ref-period is already
captured. If `StoreStatus.uncaptured` contains a `"<src>:..."` entry, the release is new;
`watch` calls `_run_update(as_of=pub, only=src)`. With `--snapshot`, it derives the **day-12
anchor of the captured ref-month** and calls `_run_snapshot(as_of=date(y, m, 12))`. A clean
no-op when nothing is uncaptured.

**Key design points:**
- `_run_update` receives a `date` object (`pub`), not `pub.isoformat()`.
- `_run_snapshot` receives a `date` object (e.g. `date(2025, 5, 12)`), not a string.
- The private helper `_watch_snapshot_anchor(ref_token: str) -> date` returns a `date`.
- All imports inside the `watch` command body are **deferred** (the `main` callback's
  `load_dotenv()` must run before `VINTAGE_STORE_PATH` binds via `paths.py` import).
- The `--source` flag maps: `all → ["ces","qcew"]`, `ces → ["ces"]`, `qcew → ["qcew"]`.
- Feed URL routing: `"ces" → EMPSIT_FEED_URL`, `"qcew" → CEWQTR_FEED_URL`.

**Watch test strategy:** the test seeds a `tmp_path` store and a `vintage_dates.parquet` so
that `compute_status` runs **for real** (no mock of it). The feed is monkeypatched
(`nfp_download.release_dates.feed.fetch_feed`) so no network is hit. `_run_update` and
`_run_snapshot` are monkeypatched to record calls without touching the store. The
`NFP_BASE_DIR` env var points at `tmp_path`; `NFP_STORE_URI` is deleted so the local fallback
under `tmp_path` is used — avoiding canonical MinIO (which conftest auto-loads via `.env`).

**Store schema for the fixture:** the `_seed_store` helper writes a minimal CES SA partition
with the 12-column `VINTAGE_STORE_SCHEMA` shape the real store uses. The `vintage_dates.parquet`
seed uses the 5-column schema that `compute_status` reads (publication, ref_date, vintage_date,
revision, benchmark_revision). Exact schema is inferred from the `_seed_store` / `_seed_calendar`
helpers below — if `compute_status`'s actual implementation (Phase 7) requires a different
schema, update the helpers to match; the test intent (present vs absent ref-month flips
trigger-vs-no-op) does not change.

- [ ] **Step 1: Write the failing test** — create
  `packages/nfp-vintages/src/nfp_vintages/tests/test_cli_watch.py`:

```python
"""CLI tests for `alt-nfp watch` (feed-driven trigger).

Monkeypatches feed.fetch_feed and the _run_update/_run_snapshot helpers; lets
compute_status run for real against a tmp store + vintage_dates.parquet so the
present/absent ref-month decides trigger-vs-no-op. Store-write-free: we seed
rows to a tmp_path partition and read them. NEVER points at real MinIO (conftest
loads prod creds; NFP_STORE_URI is deleted so the local fallback under tmp_path
is used).
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from typer.testing import CliRunner

from nfp_download.release_dates.feed import FeedItem

runner = CliRunner()


def _seed_store(store_root, *, ref_dates, vintage_date):
    """Write a minimal CES SA partition with the given headline ref_dates."""
    part = store_root / "source=ces" / "seasonally_adjusted=true"
    part.mkdir(parents=True, exist_ok=True)
    n = len(ref_dates)
    rows = {
        "ref_date": list(ref_dates),
        "industry_type": ["total"] * n,
        "industry_code": ["00"] * n,
        "ownership": ["total"] * n,
        "size_class_type": [None] * n,
        "size_class_code": [None] * n,
        "geographic_type": ["national"] * n,
        "geographic_code": ["00"] * n,
        "revision": [0] * n,
        "benchmark_revision": [0] * n,
        "vintage_date": [vintage_date] * n,
        "employment": [150_000.0 + i for i in range(n)],
    }
    pl.DataFrame(rows).write_parquet(str(part / "part-0.parquet"))


def _seed_calendar(intermediate_dir, rows):
    """Write a minimal vintage_dates.parquet (publication/ref/vintage/rev)."""
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        rows,
        schema={
            "publication": pl.Utf8,
            "ref_date": pl.Date,
            "vintage_date": pl.Date,
            "revision": pl.Int64,
            "benchmark_revision": pl.Int64,
        },
        orient="row",
    ).write_parquet(str(intermediate_dir / "vintage_dates.parquet"))


@pytest.fixture
def watch_env(tmp_path, monkeypatch):
    """Point store + intermediate dirs at tmp_path; return (store_root, intermediate)."""
    store_root = tmp_path / "store"
    intermediate = tmp_path / "intermediate"
    monkeypatch.setenv("NFP_BASE_DIR", str(tmp_path))
    monkeypatch.delenv("NFP_STORE_URI", raising=False)
    return store_root, intermediate


def _patch_feed(monkeypatch, pub_date: date):
    """Make fetch_feed return one empsit item published on ``pub_date``."""
    item = FeedItem(
        title="Employment Situation Summary",
        pub_date=pub_date,
        guid=f"empsit_{pub_date.isoformat()}",
    )
    import nfp_download.release_dates.feed as feed_mod

    monkeypatch.setattr(feed_mod, "fetch_feed", lambda url, **kw: [item])


def test_triggers_update_when_refmonth_uncaptured(watch_env, monkeypatch):
    """A feed release whose ref-month is NOT in the store triggers update."""
    store_root, intermediate = watch_env
    # Store has CES through 2025-04; calendar says 2025-05 rev0 published 2025-06-06.
    _seed_store(
        store_root,
        ref_dates=[date(2025, 3, 1), date(2025, 4, 1)],
        vintage_date=date(2025, 5, 2),
    )
    _seed_calendar(
        intermediate,
        rows=[
            ("ces", date(2025, 4, 1), date(2025, 5, 2), 0, 0),
            ("ces", date(2025, 5, 1), date(2025, 6, 6), 0, 0),
        ],
    )
    _patch_feed(monkeypatch, date(2025, 6, 6))

    calls = []
    import nfp_vintages.__main__ as cli

    monkeypatch.setattr(cli, "_run_update", lambda as_of, **kw: calls.append({"as_of": as_of, **kw}))
    monkeypatch.setattr(cli, "_run_snapshot", lambda as_of, **kw: calls.append(("snap", as_of, kw)))

    result = runner.invoke(cli.app, ["watch", "--source", "ces"])
    assert result.exit_code == 0, result.output
    update_calls = [c for c in calls if isinstance(c, dict)]
    assert len(update_calls) == 1
    # as_of is a date object, not a string (contract: _run_update(as_of: date, ...))
    assert update_calls[0]["as_of"] == date(2025, 6, 6)
    assert update_calls[0]["only"] == "ces"


def test_no_op_when_refmonth_already_present(watch_env, monkeypatch):
    """A feed release whose ref-month IS captured triggers nothing."""
    store_root, intermediate = watch_env
    _seed_store(
        store_root,
        ref_dates=[date(2025, 4, 1), date(2025, 5, 1)],
        vintage_date=date(2025, 6, 6),
    )
    _seed_calendar(
        intermediate,
        rows=[
            ("ces", date(2025, 4, 1), date(2025, 5, 2), 0, 0),
            ("ces", date(2025, 5, 1), date(2025, 6, 6), 0, 0),
        ],
    )
    _patch_feed(monkeypatch, date(2025, 6, 6))

    calls = []
    import nfp_vintages.__main__ as cli

    monkeypatch.setattr(cli, "_run_update", lambda as_of, **kw: calls.append({"as_of": as_of, **kw}))
    monkeypatch.setattr(cli, "_run_snapshot", lambda as_of, **kw: calls.append(("snap", as_of, kw)))

    result = runner.invoke(cli.app, ["watch", "--source", "ces"])
    assert result.exit_code == 0, result.output
    assert calls == []  # nothing uncaptured → clean no-op


def test_snapshot_uses_day12_anchor_not_pubdate(watch_env, monkeypatch):
    """With --snapshot, snapshot as-of is date(refmonth.year, refmonth.month, 12), not pubDate."""
    store_root, intermediate = watch_env
    _seed_store(
        store_root,
        ref_dates=[date(2025, 3, 1), date(2025, 4, 1)],
        vintage_date=date(2025, 5, 2),
    )
    _seed_calendar(
        intermediate,
        rows=[
            ("ces", date(2025, 4, 1), date(2025, 5, 2), 0, 0),
            ("ces", date(2025, 5, 1), date(2025, 6, 6), 0, 0),
        ],
    )
    _patch_feed(monkeypatch, date(2025, 6, 6))

    snaps = []
    import nfp_vintages.__main__ as cli

    monkeypatch.setattr(cli, "_run_update", lambda as_of, **kw: None)
    monkeypatch.setattr(cli, "_run_snapshot", lambda as_of, **kw: snaps.append(as_of))

    result = runner.invoke(cli.app, ["watch", "--source", "ces", "--snapshot"])
    assert result.exit_code == 0, result.output
    assert len(snaps) == 1
    # Captured ref-month is 2025-05 → anchor date(2025, 5, 12), NOT pubDate date(2025, 6, 6).
    assert snaps[0] == date(2025, 5, 12)
```

- [ ] **Step 2: Run the test, verify it fails** —
  ```
  uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_cli_watch.py -q --no-cov
  ```
  Expected failure: the `watch` command does not exist yet, so
  `runner.invoke(cli.app, ["watch", ...])` exits non-zero (`No such command 'watch'`),
  failing the `exit_code == 0` assertions.

- [ ] **Step 3: Implement** — add `watch` and the private `_watch_snapshot_anchor` helper to
  `packages/nfp-vintages/src/nfp_vintages/__main__.py`. All imports inside the command body
  are **deferred** (the `main` callback runs `load_dotenv()` before `VINTAGE_STORE_PATH`
  resolves). Insert the two functions after the existing `status` command:

```python
def _watch_snapshot_anchor(ref_token: str) -> "date":
    """Day-12 anchor for an uncaptured ref token — never the raw pubDate.

    ``ref_token`` is the string after the source prefix in ``StoreStatus.uncaptured``
    (e.g. ``"2025-05-01"`` for CES or ``"2025-Q2"`` for QCEW). Returns the 12th of
    the ref-period's closing month — the convention ``_run_snapshot`` enforces (§4a).

    Parameters
    ----------
    ref_token : str
        ISO ref date (CES, ``"2025-05-01"``) or QCEW quarter token (``"2025-Q2"``).

    Returns
    -------
    date
        The day-12 anchor as a ``datetime.date``.
    """
    from datetime import date as _date

    if "-Q" in ref_token:
        year_str, q_str = ref_token.split("-Q")
        month = int(q_str) * 3  # Q1→Mar, Q2→Jun, Q3→Sep, Q4→Dec
        return _date(int(year_str), month, 12)
    ref = _date.fromisoformat(ref_token)
    return _date(ref.year, ref.month, 12)


@app.command()
def watch(
    source: str = typer.Option(
        "all", "--source", help="Which feed(s) to poll: all | ces | qcew."
    ),
    snapshot_after: bool = typer.Option(
        False, "--snapshot", help="Also bake a ModelData snapshot for each new release."
    ),
) -> None:
    """Poll the BLS release feed; trigger ``update`` on a newly-published release.

    Designed for a daily cron. The feed answers only "a release is out *now*" and
    supplies the publication day (``pubDate``); the **store** (via ``compute_status``)
    is the source of truth for which ref-month/quarter is still uncaptured. A clean
    no-op on days with nothing new. A same-day CES + QCEW co-release triggers both.
    """
    from nfp_download.release_dates.feed import (
        CEWQTR_FEED_URL,
        EMPSIT_FEED_URL,
        fetch_feed,
    )
    from nfp_vintages.store_status import compute_status

    _feeds = {"ces": EMPSIT_FEED_URL, "qcew": CEWQTR_FEED_URL}
    if source == "all":
        wanted = ["ces", "qcew"]
    elif source in _feeds:
        wanted = [source]
    else:
        raise typer.BadParameter("must be one of: all, ces, qcew", param_hint="--source")

    for src in wanted:
        items = fetch_feed(_feeds[src])
        if not items:
            print(f"  {src}: feed empty — skipping")
            continue
        # BLS lists newest first; the top item is the latest release.
        latest = items[0]
        pub = latest.pub_date  # type: date — already a date object from parse_feed

        # The store decides whether this release's ref-period is captured.
        status = compute_status(as_of=pub)
        uncaptured = [u for u in status.uncaptured if u.startswith(f"{src}:")]
        if not uncaptured:
            print(f"  {src}: latest release ({pub}) already captured — no-op")
            continue

        # ref_token is the part after "src:" — ISO date (CES) or YYYY-Qn (QCEW).
        ref_token = uncaptured[0].split(":", 1)[1]
        print(f"  {src}: NEW release {pub} (uncaptured {ref_token}) — updating")
        _run_update(pub, only=src)

        if snapshot_after:
            anchor = _watch_snapshot_anchor(ref_token)
            print(f"  {src}: snapshot at day-12 anchor {anchor}")
            _run_snapshot(anchor)
```

  **Lint notes:**
  - No `dict(a=1)` style (C408): the `_feeds` dict uses `{...}` literal. ✓
  - `zip(..., strict=True)` (B905): no zip calls here. ✓
  - Imports at module top **within the deferred body** only (the deferred-import rule for
    commands; E402 applies to module-level imports, not body-local imports). ✓
  - Line length ≤100 cols. ✓

- [ ] **Step 4: Run, verify pass** —
  ```
  uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_cli_watch.py -q --no-cov
  ```
  Expected: **3 tests pass**.

  > **Note on cross-phase prerequisites:** if Phase 7 (`compute_status` / `StoreStatus`) is
  > not yet implemented, the `watch` command body's deferred `from nfp_vintages.store_status
  > import compute_status` will raise `ImportError` at invoke time, failing all three tests.
  > Phase 7 must be done before this step is green. If running this phase in isolation, stub
  > `store_status.py` with the minimal `StoreStatus`/`compute_status` contract from the
  > Interface Contract section of plan16-new.md (the `watch` test seeds a real store and relies
  > on real `compute_status` logic to flip trigger-vs-no-op — a stub must at least read the
  > store partitions to return a meaningful `uncaptured` list).

  Then run the full nfp-vintages non-network suite and lint:
  ```
  uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/ \
    -q --no-cov -m "not network and not slow"
  uv run ruff check \
    packages/nfp-vintages/src/nfp_vintages/__main__.py \
    packages/nfp-vintages/src/nfp_vintages/tests/test_cli_watch.py
  ```

- [ ] **Step 5: Commit** —
  ```
  git add \
    packages/nfp-vintages/src/nfp_vintages/__main__.py \
    packages/nfp-vintages/src/nfp_vintages/tests/test_cli_watch.py
  git commit -m "$(cat <<'EOF'
  feat(cli): add alt-nfp watch — feed-driven update trigger + day-12 snapshot anchor

  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Phase 8 dependency graph

```
Task 8.1 (parse_feed, FeedItem)
    └── Task 8.2 (fetch_feed, async wrapper)
            └── Task 8.3 (watch command) ← also needs Phase 5 + Phase 7
```

Phase 8 is unblocked by Phase 5 for Tasks 8.1–8.2 (pure `nfp-download`). Task 8.3 requires
both Phase 5 (`_run_update`/`_run_snapshot` in `__main__.py`) and Phase 7 (`compute_status` +
`StoreStatus.uncaptured` in `store_status.py`) to be done before the test can go green.
## Phase 9 — `scripts/bootstrap_store.py` + legacy CLI retirement + docs

> **Spec:** `specs/cli_production_workflow.md` §10 (bootstrap script + legacy retirement),
> §11 (firewall/file map), §14 step 9. **Last phase in the build order.** It depends on
> Phase 1 (the relocated public `nfp_ingest.qcew_acquire.acquire_qcew_levels` /
> `acquire_qcew_size_native`), Phase 3 (the lifted `nfp_vintages.calendar.advance_release_calendar`),
> and Phases 5–8 (the `update`/`status`/`watch` commands that the retirement leaves in place).
> It (a) writes the one-time bootstrap script generalizing the `_t8_promote.py` cutover, (b)
> deletes the retired legacy commands from `__main__.py` while keeping `snapshot` + the Phase 5–8
> production surface, and (c) updates four CLAUDE.md command banners/maps. **Retirement MUST keep
> the suite green** (no orphaned imports, no test that invokes a deleted command).

> **PINNED — plan against the CONTRACT surface, NOT current `__main__.py`.** When this phase
> executes, Phases 5/7/8 will already have authored `update`/`status`/`watch` and re-shaped
> `snapshot` into the thin `_run_snapshot` wrapper (Task 5.1). So the **KEEP** list is the
> contract's production surface — `update`/`snapshot`/`status`/`watch` — and the **RETIRE** list is
> `download`/`download-indicators`/`process`/`current`/`build`/`build-rebuild` + the
> `invoke_without_command=True` bare-run chain. **Verify the live file before deleting** (the line
> numbers below are for the *pre-Phase-5* file shown in the verification appendix; the Phase 5–8
> edits land on top first, so re-grep, do not trust spans).

> **Container-safety (the #1 risk in this phase) — three corrections to the old plan.** Every
> `write_parquet`/`mkdir` in the new code threads `str(path)` + `storage_options_for(path)` + an
> `is_remote(path)` mkdir guard. The bootstrap delegates its store write entirely to
> `write_rebuild_store` (which is already container-safe — `rebuild_store.py:253,262,283-286`), so
> the bootstrap itself writes **no** parquet directly. But the **acquisition seams leak to `./data`
> unless wired to the tempdir**, which the old plan's reproduced code never actually did:
> 1. `download_ces(data_dir=None)` writes to `DATA_DIR/downloads/ces/cesvinall/` (`bls/bulk.py:79`)
>    — a `./data` write. The bootstrap MUST pass `data_dir=Path(tmp)`.
> 2. `build_ces_panel(cesvinall_dir=None)` defaults to `CESVINALL_DIR = DOWNLOADS_DIR/ces/cesvinall`
>    (`ces_builder.py:67,304`) — a `./data` read of files that only exist under the tempdir. The
>    bootstrap MUST pass `cesvinall_dir=Path(tmp)/"downloads"/"ces"/"cesvinall"` so it reads what
>    `download_ces` just extracted there.
> 3. `write_rebuild_store(panel, store_path, *, allow_canonical=False)` is `panel`-positional-first
>    (`rebuild_store.py:212-217`). The old plan's `write_rebuild_store(_store_path(args.scratch),
>    panel=panel)` puts the **path in the `panel` slot** and re-passes `panel=` — wrong. Use the
>    explicit `write_rebuild_store(panel, _store_path(args.scratch), allow_canonical=False)`.

---

### Task 9.1: `scripts/bootstrap_store.py` — one-time rebuild + promote

The bootstrap script lifts the **rebuild** lineage (not the legacy `build_store` one), ordered
exactly per §10: `download_ces(data_dir=tmp)` → `build_ces_panel(cesvinall_dir=tmp/…)` (run with
`vintage_dates.parquet` present via `advance_release_calendar()` so bootstrap & the §5.1 capture
agree on `(revision, benchmark_revision)` for overlap months) → `acquire_qcew_levels` →
`build_qcew_panel`; `acquire_qcew_size_native` (Q1-only) → `build_size_class_panel` →
`compose_rebuild_panel` → `write_rebuild_store(panel, scratch, allow_canonical=False)` to a
**scratch** prefix → **promote** via a generalized copy-then-delete cutover (the
`_t8_promote.py:cutover` flow). The promote keeps the `is_canonical_store` refusal: a write or
overwrite-mirror straight to `…/store` is the exact hazard CLAUDE.md warns about (filenames encode
vintage ranges, so an overwrite leaves stale fragments). `scripts/mirror_store.py` is **not** reused
— it is overwrite-only.

> **Tier-C tempdir (absorbs plans/15 Phase 2 Tasks 7 & 9).** This is the one place a run-scoped
> temp dir works (a single process). Wrap the rebuild in
> `with tempfile.TemporaryDirectory(prefix="altnfp-bootstrap-") as tmp:` and thread `Path(tmp)` as
> the scratch root for every byproduct that would otherwise land under `./data`: the raw
> `download_ces()` `cesvinall/` extract and any scraped release HTML. Nothing the script writes may
> persist under `./data` on Bloomberg's container — only the rebuilt **store** (S3 via the scratch
> `NFP_STORE_URI`) survives the run. (plans/15 Tasks 8 + 10 — the HTTP-cache and SAE-checkpoint
> tempfile defaults — are already done and need nothing here.)

It is a **script**, not a CLI command (§4 table row, §10): invoked as
`uv run python scripts/bootstrap_store.py …`, `argparse` for flags.

**Files:**
- Create: `/Users/lowell/Projects/alt-nfp/scripts/bootstrap_store.py`
- Test: `/Users/lowell/Projects/alt-nfp/packages/nfp-vintages/src/nfp_vintages/tests/test_bootstrap_store.py`

> Why the test lives under `packages/nfp-vintages/.../tests/`: `pyproject.toml` sets
> `testpaths = ["packages"]` — pytest does **not** collect from `scripts/`. The test imports the
> script by file path via `importlib.util.spec_from_file_location`, anchored at
> `nfp_lookups.paths.BASE_DIR / "scripts" / "bootstrap_store.py"`.

**Interfaces — Produces:** `main(argv: list[str] | None = None) -> None` plus module-level
`download_ces` / `build_ces_panel` / `acquire_qcew_levels` / `acquire_qcew_size_native` /
`advance_release_calendar` bindings (imported at module scope so the test monkeypatches them).

- [ ] **Step 1: Write the failing test** — a no-network, no-real-store smoke test. It monkeypatches
  the acquisition seams (`download_ces`, `advance_release_calendar`, `build_ces_panel`,
  `acquire_qcew_levels`, `acquire_qcew_size_native`) to return tiny synthetic inputs, points
  `--scratch` + `--canonical` at two **local** `tmp_path` prefixes (both
  `is_canonical_store(...) == False`, so the guard is satisfied and `write_rebuild_store`'s own
  canonical guard passes), runs `main()`, and asserts the canonical prefix ends up populated. A
  second test asserts the `is_canonical_store` refusal fires when `--scratch` is the canonical
  `s3://alt-nfp/store`.

```python
"""Smoke test for scripts/bootstrap_store.py (no network, no real store).

Phase 9 of specs/cli_production_workflow.md. The bootstrap orchestration is
exercised with every heavy/network step monkeypatched and both store prefixes
pinned to tmp_path — the real bootstrap is NEVER run against MinIO here.
"""

from __future__ import annotations

import importlib.util

import polars as pl
import pytest
from nfp_lookups.paths import BASE_DIR


def _load_bootstrap():
    """Import scripts/bootstrap_store.py by path (scripts/ is not on testpaths)."""
    path = BASE_DIR / "scripts" / "bootstrap_store.py"
    spec = importlib.util.spec_from_file_location("bootstrap_store", path)
    assert spec and spec.loader, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_ces_panel() -> pl.DataFrame:
    """One CES row in VINTAGE_STORE_SCHEMA (already remapped to total taxonomy)."""
    from datetime import date

    from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA

    row = {
        "geographic_type": "national",
        "geographic_code": "00",
        "ownership": "total",
        "industry_type": "total",
        "industry_code": "00",
        "ref_date": date(2024, 1, 12),
        "vintage_date": date(2024, 2, 2),
        "revision": 0,
        "benchmark_revision": 0,
        "employment": 158000.0,
        "size_class_type": None,
        "size_class_code": None,
        "source": "ces",
        "seasonally_adjusted": True,
    }
    return pl.DataFrame([row], schema=dict(VINTAGE_STORE_SCHEMA))


def _tiny_qcew_levels_panel() -> pl.DataFrame:
    """One QCEW-levels row in VINTAGE_STORE_SCHEMA (post-build_qcew_panel shape)."""
    from datetime import date

    from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA

    row = {
        "geographic_type": "national",
        "geographic_code": "00",
        "ownership": "private",
        "industry_type": "total",
        "industry_code": "05",
        "ref_date": date(2024, 3, 12),
        "vintage_date": date(2024, 9, 1),
        "revision": 0,
        "benchmark_revision": 0,
        "employment": 130000.0,
        "size_class_type": None,
        "size_class_code": None,
        "source": "qcew",
        "seasonally_adjusted": False,
    }
    return pl.DataFrame([row], schema=dict(VINTAGE_STORE_SCHEMA))


def _patch_seams(monkeypatch, boot):
    """Replace every heavy/network seam with a zero-network synthetic stub."""
    monkeypatch.setattr(boot, "download_ces", lambda *a, **k: None)
    monkeypatch.setattr(boot, "advance_release_calendar", lambda *a, **k: None)
    monkeypatch.setattr(boot, "build_ces_panel", lambda *a, **k: _tiny_ces_panel())
    monkeypatch.setattr(boot, "acquire_qcew_levels", lambda *a, **k: pl.DataFrame())
    monkeypatch.setattr(boot, "acquire_qcew_size_native", lambda *a, **k: pl.DataFrame())
    # build_qcew_panel/build_size_class_panel are the crosswalk consumers; with
    # empty raw frames they would error, so stub the *panel* builders too. The
    # QCEW-levels panel carries the qcew partition; size returns empty (no Q1
    # coverage in this synthetic fixture).
    monkeypatch.setattr(boot, "build_qcew_panel", lambda *a, **k: _tiny_qcew_levels_panel())
    monkeypatch.setattr(boot, "build_size_class_panel", lambda *a, **k: pl.DataFrame())


def test_bootstrap_builds_scratch_then_promotes_to_canonical(monkeypatch, tmp_path):
    boot = _load_bootstrap()
    _patch_seams(monkeypatch, boot)

    scratch = tmp_path / "store-rebuild"
    canonical = tmp_path / "store"

    boot.main(
        argv=[
            "--scratch", str(scratch),
            "--canonical", str(canonical),
            "--start-year", "2024",
            "--end-year", "2024",
        ]
    )

    # Promotion left the canonical prefix populated with the composed partitions.
    canon_files = sorted(canonical.glob("**/*.parquet"))
    assert canon_files, "canonical store has no parquet partitions after bootstrap"
    ces_part = canonical / "source=ces" / "seasonally_adjusted=true"
    assert ces_part.exists(), "expected source=ces/seasonally_adjusted=true partition"
    df = pl.read_parquet(canon_files[0])
    assert df.height >= 1


def test_bootstrap_refuses_canonical_uri_as_scratch(monkeypatch, tmp_path):
    """--scratch must not be the canonical store (is_canonical_store guard)."""
    boot = _load_bootstrap()
    _patch_seams(monkeypatch, boot)
    with pytest.raises(SystemExit):
        boot.main(
            argv=[
                "--scratch", "s3://alt-nfp/store",
                "--canonical", str(tmp_path / "store"),
                "--start-year", "2024",
                "--end-year", "2024",
            ]
        )
```

- [ ] **Step 2: Run the test, verify it fails** —
  `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_bootstrap_store.py -q --no-cov`
  Expected: **FAIL** — `scripts/bootstrap_store.py` does not exist yet, so the `_load_bootstrap()`
  helper's `spec_from_file_location` either returns `None` (assertion `spec and spec.loader` trips)
  or `exec_module` raises `FileNotFoundError`. Confirm the failure names the missing script, not a
  different error.

- [ ] **Step 3: Implement** — `scripts/bootstrap_store.py`. The promote step generalizes
  `_t8_promote.py:cutover` (`scripts/_t8_promote.py:123-145`): copy every scratch file into the
  canonical prefix under its rebuilt name, then delete any canonical orphan not in the new set, then
  verify the keyset. Module-level `download_ces` / `build_ces_panel` / `acquire_qcew_levels` /
  `acquire_qcew_size_native` / `advance_release_calendar` / `build_qcew_panel` /
  `build_size_class_panel` bindings are imported at module scope so the test can monkeypatch them.
  Local `Path` prefixes use plain filesystem ops; `s3://` prefixes use the `_t8_promote.py` `_fs()`
  s3fs pattern. The `is_canonical_store` guard is checked on `--scratch` **first**, before any work.
  **Container-safety: thread `Path(tmp)` into `download_ces`/`build_ces_panel` so no `./data` write
  happens; the store write is delegated to the already-safe `write_rebuild_store`.**

```python
#!/usr/bin/env python3
"""One-time historical store rebuild + promote (NOT a CLI command).

Lifts the **rebuild** lineage (spec cli_production_workflow.md §10), ordered::

    download_ces(data_dir=tmp)                    # extract cesvinall/ triangular CSVs
    advance_release_calendar()                    # vintage_dates.parquet present for overlap parity
    build_ces_panel(cesvinall_dir=tmp/...)        # CES NSA+SA store-schema rows
    acquire_qcew_levels(...)      -> build_qcew_panel(...)
    acquire_qcew_size_native(...) -> build_size_class_panel(...)   # Q1-only
    compose_rebuild_panel(...)
    write_rebuild_store(panel, scratch, allow_canonical=False)     # scratch prefix
    promote(scratch -> canonical)                 # copy-then-delete cutover (_t8_promote flow)

Usage::

    NFP_STORE_URI=s3://alt-nfp/store-rebuild \\
      uv run python scripts/bootstrap_store.py \\
      --scratch s3://alt-nfp/store-rebuild --canonical s3://alt-nfp/store

Scope is national-only, 2017+ (the intended canonical scope). QCEW is fetched
live from the CEW API (not the bulk ZIPs), so only ``download_ces`` is wired.

Container-safety (plans/15): every byproduct that the legacy lineage parked under
``./data`` is routed to a run-scoped ``tempfile.TemporaryDirectory`` here — the
raw ``cesvinall/`` extract and the CES read both point at ``tmp``. The only
artifact that survives the run is the rebuilt **store** on S3 (the scratch
``NFP_STORE_URI`` prefix). ``write_rebuild_store`` is itself container-safe
(``str(path)`` + ``storage_options_for`` + ``is_remote`` mkdir guard).

The promote step copies rebuild files into the canonical prefix then deletes the
old orphans (filenames encode vintage ranges, so a plain overwrite-mirror would
leave both files and corrupt the store — the exact hazard CLAUDE.md warns about).
``scripts/mirror_store.py`` is overwrite-only and is deliberately NOT used here.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

# --- .env MUST load before any nfp_* import: nfp_lookups.paths reads
#     NFP_STORE_URI at import time. ---
from dotenv import load_dotenv

load_dotenv(".env")

from nfp_download.bls.bulk import download_ces  # noqa: E402
from nfp_ingest.ces_builder import build_ces_panel  # noqa: E402
from nfp_ingest.qcew_acquire import (  # noqa: E402
    acquire_qcew_levels,
    acquire_qcew_size_native,
)
from nfp_ingest.qcew_crosswalk import build_qcew_panel  # noqa: E402
from nfp_ingest.size_class import build_size_class_panel  # noqa: E402
from nfp_lookups.paths import is_canonical_store  # noqa: E402

from nfp_vintages.calendar import advance_release_calendar  # noqa: E402
from nfp_vintages.rebuild_store import (  # noqa: E402
    compose_rebuild_panel,
    write_rebuild_store,
)


def _is_remote(uri: str) -> bool:
    return uri.startswith(("s3://", "s3a://"))


def _store_path(uri: str):
    """A pathlib-compatible handle for a scratch/canonical store location.

    Local prefixes return a plain ``Path``; ``s3://`` prefixes return a
    credentialed ``UPath`` (the same shape ``write_rebuild_store`` accepts).
    """
    if _is_remote(uri):
        from upath import UPath

        endpoint = os.environ.get("AWS_ENDPOINT_URL")
        client_kwargs = {"endpoint_url": endpoint} if endpoint else {}
        return UPath(
            uri,
            key=os.environ.get("AWS_ACCESS_KEY_ID"),
            secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            client_kwargs=client_kwargs,
        )
    return Path(uri)


# ---------------------------------------------------------------------------
# Promote: generalized _t8_promote.py:cutover (copy-then-delete per partition)
# ---------------------------------------------------------------------------


def _s3fs():
    import s3fs

    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    return s3fs.S3FileSystem(
        key=os.environ.get("AWS_ACCESS_KEY_ID"),
        secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        client_kwargs={"endpoint_url": endpoint} if endpoint else {},
    )


def _local_keys(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.glob("**/*.parquet"))


def _s3_keys(fs, prefix: str) -> list[str]:
    """Genuine children of *prefix* only (store vs store-rebuild share a head)."""
    return sorted(k for k in fs.find(prefix) if k.startswith(prefix + "/"))


def _promote_local(scratch: Path, canonical: Path) -> None:
    rel_keys = _local_keys(scratch)
    if not rel_keys:
        sys.exit(f"FATAL: scratch store {scratch} is empty — refusing promote.")
    canonical.mkdir(parents=True, exist_ok=True)
    # 1) copy rebuild files in (under their rebuilt names).
    for rel in rel_keys:
        dst = canonical / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((scratch / rel).read_bytes())
    # 2) delete old-named orphans (anything canonical-side not in the new set).
    new_set = set(rel_keys)
    for p in sorted(canonical.glob("**/*.parquet")):
        if p.relative_to(canonical).as_posix() not in new_set:
            p.unlink()
    # 3) verify: canonical == exactly the rebuild set.
    final = set(_local_keys(canonical))
    if final != new_set:
        sys.exit(f"FATAL: post-promote keyset mismatch under {canonical}.")
    print(f"promote (local): +{len(new_set)} files; canonical == rebuild set, verified")


def _promote_remote(scratch_uri: str, canonical_uri: str) -> None:
    fs = _s3fs()
    src = scratch_uri.removeprefix("s3://").rstrip("/")
    dst = canonical_uri.removeprefix("s3://").rstrip("/")
    src_keys = _s3_keys(fs, src)
    if not src_keys:
        sys.exit(f"FATAL: scratch store {scratch_uri} is empty — refusing promote.")
    new_dst = {k.replace(src, dst, 1): k for k in src_keys}  # dst -> src
    # 1) copy rebuild files in (new names).
    for dst_key, src_key in new_dst.items():
        fs.pipe_file(dst_key, fs.cat_file(src_key))
    # 2) delete old-named orphans.
    for k in _s3_keys(fs, dst):
        if k not in new_dst:
            fs.rm(k)
    # 3) verify keyset.
    final = _s3_keys(fs, dst)
    if final != sorted(new_dst):
        sys.exit(f"FATAL: post-promote keyset mismatch under {canonical_uri}.")
    print(f"promote (s3): +{len(new_dst)} files; canonical == rebuild set, verified")


def _promote_scratch_to_canonical(scratch_uri: str, canonical_uri: str) -> None:
    """Copy-then-delete cutover from *scratch* to *canonical* (no overwrite-mirror)."""
    if _is_remote(canonical_uri):
        _promote_remote(scratch_uri, canonical_uri)
    else:
        _promote_local(Path(scratch_uri), Path(canonical_uri))


# ---------------------------------------------------------------------------
# Rebuild orchestration
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="One-time store rebuild + promote.")
    parser.add_argument(
        "--scratch",
        required=True,
        help="Scratch store URI/path (e.g. s3://alt-nfp/store-rebuild). "
        "Must NOT be the canonical store.",
    )
    parser.add_argument(
        "--canonical",
        required=True,
        help="Canonical store URI/path to promote into (e.g. s3://alt-nfp/store).",
    )
    parser.add_argument(
        "--start-year", type=int, default=2017, help="First QCEW reference year."
    )
    parser.add_argument(
        "--end-year", type=int, default=None, help="Last QCEW reference year (inclusive)."
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Build the scratch store but skip the canonical promote.",
    )
    args = parser.parse_args(argv)

    # Guard FIRST — refuse the canonical store as a scratch target (no I/O before this).
    if is_canonical_store(args.scratch):
        sys.exit(
            f"refusing to bootstrap straight to the canonical store ({args.scratch}); "
            "target a scratch prefix (e.g. s3://alt-nfp/store-rebuild)."
        )

    # Container-safety: every ./data byproduct (raw cesvinall extract, scraped HTML)
    # lands under a run-scoped tempdir — only the rebuilt STORE on S3 survives.
    with tempfile.TemporaryDirectory(prefix="altnfp-bootstrap-") as tmp:
        tmp_root = Path(tmp)
        cesvinall_dir = tmp_root / "downloads" / "ces" / "cesvinall"

        print("=== Bootstrap: download CES triangular CSVs (-> tempdir) ===")
        download_ces(data_dir=tmp_root)

        print("=== Bootstrap: advance release calendar (overlap parity) ===")
        advance_release_calendar()

        print("=== Bootstrap: build CES panel (NSA + SA) ===")
        ces = build_ces_panel(cesvinall_dir=cesvinall_dir)
        print(f"  CES: {ces.height:,} rows")

        print(f"=== Bootstrap: acquire QCEW levels ({args.start_year}-{args.end_year}) ===")
        raw_qcew = acquire_qcew_levels(start_year=args.start_year, end_year=args.end_year)
        qcew_levels = build_qcew_panel(raw_qcew)
        print(f"  QCEW levels: {qcew_levels.height:,} rows")

        print(f"=== Bootstrap: acquire QCEW size native ({args.start_year}-{args.end_year}) ===")
        size_native = acquire_qcew_size_native(
            start_year=args.start_year, end_year=args.end_year
        )
        size = build_size_class_panel(size_native) if size_native.height else None
        if size is not None and size.height:
            print(f"  QCEW size: {size.height:,} rows")
        else:
            size = None
            print("  QCEW size: 0 rows (skipped)")

        print("=== Bootstrap: compose panels ===")
        panel = compose_rebuild_panel(ces, qcew_levels, size)
        print(f"  Combined: {panel.height:,} rows")

        print(f"=== Bootstrap: write scratch store ({args.scratch}) ===")
        # panel is positional-first; store_path second (rebuild_store.py:212-217).
        write_rebuild_store(panel, _store_path(args.scratch), allow_canonical=False)

    if args.no_promote:
        print("Done (scratch only; --no-promote set).")
        return

    print(f"=== Bootstrap: promote {args.scratch} -> {args.canonical} ===")
    _promote_scratch_to_canonical(args.scratch, args.canonical)
    print("Done.")


if __name__ == "__main__":
    main()
```

> **Why the size panel is `None` when empty.** `compose_rebuild_panel(ces, qcew_levels, size)`
> takes `size: pl.DataFrame | None` (`rebuild_store.py:74-78`) and runs the §7 anti-join only when
> `size is not None`. An *empty* size frame would still enter the `if size is not None` branch and
> trip the join machinery; coercing an empty `build_size_class_panel` result back to `None` matches
> the no-size-coverage contract. (The synthetic test returns an empty size frame for exactly this
> path.)

> **Signature note (`write_rebuild_store`).** Real signature:
> `write_rebuild_store(panel, store_path=None, *, allow_canonical=False)` (`rebuild_store.py:212-217`).
> `panel` is positional-first, `store_path` second. The old plan's
> `write_rebuild_store(_store_path(args.scratch), panel=panel)` is wrong (path in the panel slot).
> Above uses the explicit `write_rebuild_store(panel, _store_path(args.scratch),
> allow_canonical=False)`, which hits the same `is_canonical_store(out_path) and not
> allow_canonical` guard (`rebuild_store.py:245`) — a second line of defense even though `main`
> guards `--scratch` earlier.

- [ ] **Step 4: Run, verify pass** —
  `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_bootstrap_store.py -q --no-cov`
  Expected: **PASS** (both tests). Then run the package suite + lint:
  `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests -q -m "not network and not slow" --no-cov`
  and `uv run ruff check scripts/bootstrap_store.py packages/nfp-vintages/`.

- [ ] **Step 5: Commit** —
  `git add scripts/bootstrap_store.py packages/nfp-vintages/src/nfp_vintages/tests/test_bootstrap_store.py`
  then
  `git commit -m "feat(bootstrap): one-time store rebuild+promote script (cli_production_workflow §10)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"`

---

### Task 9.2: Retire the legacy CLI commands + bare-run from `__main__.py`

Delete the retired commands from the everyday surface (§4, §10): `download`,
`download-indicators`, `process`, `current`, `build`, `build-rebuild`, and the
`invoke_without_command=True` bare-run chain. Their reusable bodies have already moved (QCEW
acquire → `nfp_ingest.qcew_acquire` in Phase 1; the calendar scrape →
`nfp_vintages.calendar.advance_release_calendar` in Phase 3; the rebuild compose/write into
`scripts/bootstrap_store.py`, Task 9.1). The new `update`/`status`/`watch` commands (Phases 5–8)
and `snapshot` (with the §4a day-12 fix) remain. The callback keeps only `load_dotenv()` (no
fallthrough run). Retirement MUST keep the suite green — no orphaned imports, no test that invokes a
deleted command.

> **VERIFY THE LIVE FILE FIRST (PINNED).** When this task runs, Phases 5–8 have already added
> `update`/`status`/`watch` and re-shaped `snapshot` into the `_run_snapshot` wrapper (Task 5.1), so
> the file is **larger** than the verification-appendix snapshot below. Re-`grep -n '@app.command'`
> the live file before deleting — delete only the **retired** command spans, leave the production
> commands and the `_run_update`/`_run_snapshot` helpers untouched.

**Files:**
- Modify: `/Users/lowell/Projects/alt-nfp/packages/nfp-vintages/src/nfp_vintages/__main__.py`
- Test: `/Users/lowell/Projects/alt-nfp/packages/nfp-vintages/src/nfp_vintages/tests/test_cli_legacy_retired.py`

- [ ] **Step 1: Write the failing test** — assert the retired command names are gone from the Typer
  app, that the surviving production commands are present, and that the bare invocation (no
  subcommand) no longer triggers a build (it shows usage and exits non-zero, not a pipeline).

```python
"""Legacy CLI retirement: retired commands gone, production surface intact.

Phase 9 of specs/cli_production_workflow.md §10. The bare `alt-nfp` must no
longer chain a store build; the retired stage commands must not be registered.
"""

from __future__ import annotations

from typer.testing import CliRunner

from nfp_vintages.__main__ import app

runner = CliRunner()

_RETIRED = {
    "download",
    "download-indicators",
    "process",
    "current",
    "build",
    "build-rebuild",
}
_KEPT = {"update", "status", "watch", "snapshot"}


def _registered_command_names() -> set[str]:
    names: set[str] = set()
    for cmd in app.registered_commands:
        # Typer derives the CLI name from the function name (underscores -> hyphens)
        # unless an explicit name was passed to @app.command(...).
        names.add(cmd.name or cmd.callback.__name__.replace("_", "-"))
    return names


def test_legacy_commands_are_gone():
    registered = _registered_command_names()
    leaked = _RETIRED & registered
    assert not leaked, f"retired commands still present: {leaked}"


def test_production_commands_present():
    registered = _registered_command_names()
    missing = _KEPT - registered
    assert not missing, f"expected production commands missing: {missing}"


def test_retired_command_invocation_errors():
    result = runner.invoke(app, ["build"])
    assert result.exit_code != 0, "`alt-nfp build` should no longer be a command"


def test_bare_invocation_does_not_run_a_build():
    # No subcommand: the old behavior chained download->...->build. After
    # retirement the bare run must NOT silently rebuild the store; with
    # no_args_is_help it shows usage and exits non-zero.
    result = runner.invoke(app, [])
    assert result.exit_code != 0
    assert "Usage" in result.output
```

- [ ] **Step 2: Run the test, verify it fails** —
  `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_cli_legacy_retired.py -q --no-cov`
  Expected: **FAIL** — `test_legacy_commands_are_gone` fails because `download`/`download-indicators`/
  `process`/`current`/`build`/`build-rebuild` are still registered, and
  `test_bare_invocation_does_not_run_a_build` fails because the `invoke_without_command=True`
  callback still chains a build (the bare run executes the pipeline / does not surface "Usage").
  Confirm the failure is the retirement assertions, not an import error.

- [ ] **Step 3: Implement** — edit `__main__.py` to drop the legacy commands and the bare-run. Only
  the production surface remains. The callback no longer takes `invoke_without_command=True` and only
  loads `.env`; `no_args_is_help=True` on the `typer.Typer(...)` makes a bare `alt-nfp` print usage
  and exit non-zero. **Re-grep the live file first** (Phases 5–8 reshaped it); shown here is the
  header through the callback plus the docstring banner — the `update`/`status`/`watch` commands
  (Phases 5–8) and `snapshot` (Phase 5 day-12 wrapper) and `_run_update`/`_run_snapshot` helpers are
  unchanged and remain below this header (do **not** duplicate or delete them — only the legacy
  command block + bare-run go away).

```python
"""Production CLI for the alt-nfp vintage store.

Usage::

    alt-nfp update --as-of T [--only ces|qcew|indicators]  # capture knowable prints for T
    alt-nfp status [--as-of T] [--store URI]               # store coverage + uncaptured alarm
    alt-nfp watch [--source ces|qcew|all] [--snapshot]     # feed-driven trigger (cron)
    alt-nfp snapshot --as-of T [--grid-end E]              # hash-pinned ModelData (day-12)

One-time historical load is a SCRIPT, not a command::

    uv run python scripts/bootstrap_store.py --scratch s3://alt-nfp/store-rebuild \\
        --canonical s3://alt-nfp/store

The legacy stage pipeline (download / download-indicators / process / current /
build / build-rebuild and the bare-run chain) was retired in the production-workflow
reshape (specs/cli_production_workflow.md §10). The calendar scrape it used now lives in
nfp_vintages.calendar.advance_release_calendar (invoked by `update`); the rebuild compose/
write moved to scripts/bootstrap_store.py.
"""

from __future__ import annotations

from datetime import date  # noqa: F401  (kept iff Phase 5 helpers annotate with it)

import typer
from dotenv import load_dotenv

app = typer.Typer(help="Production vintage-store CLI for alt-nfp.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Load environment config before any command resolves store paths."""
    load_dotenv()


# --- production commands: update / status / watch (Phases 5–8) + snapshot below ---
```

> **Deletion checklist — VERIFY against the live spans first.** In the *pre-Phase-5* file (the
> verification appendix) the retired spans are: `from pathlib import Path` (`__main__.py:16`, only
> used by `build`), the `invoke_without_command=True` callback chain (`:24-33`), `download`
> (`:36-47`), `download-indicators` (`:50-58`), `process` (`:61-78`), `current` (`:80-86`), `build`
> (`:89-106`), and `build-rebuild` (`:109-181` — its only unique imports were `_acquire_qcew_*`
> aliases, which still exist as back-compat re-exports in `rebuild_store.py:36-41` but are dead here;
> delete the command). **Keep** `snapshot` and the `if __name__ == '__main__': app()` tail. After the
> Phase 5–8 edits these line numbers SHIFT — match on the `@app.command(...)` decorator + function
> name, not the number. **Drop the `from pathlib import Path` and any now-unused `typer.Option`-only
> import** so ruff F401 stays clean. The `from datetime import date` import above is only needed if
> the Phase 5 `_run_*` helpers annotate with `date` at module scope; if Phase 5 already added it,
> don't double-import — keep exactly one.

- [ ] **Step 4: Run, verify pass** —
  `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_cli_legacy_retired.py -q --no-cov`
  Expected: **PASS**. Then confirm **no orphaned imports / green suite**:
  `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests -q -m "not network and not slow" --no-cov`
  (in particular `test_cli_update.py` + `test_update_guardrail.py` + `test_bootstrap_store.py` still
  pass — they exercise the kept commands / the script, not the retired ones), and
  `uv run ruff check packages/nfp-vintages/` (catches a now-unused `Path`/`typer.Option`/`date`
  import if one slipped through).

- [ ] **Step 5: Commit** —
  `git add packages/nfp-vintages/src/nfp_vintages/__main__.py packages/nfp-vintages/src/nfp_vintages/tests/test_cli_legacy_retired.py`
  then
  `git commit -m "refactor(cli): retire legacy download/process/build commands + bare-run (§10)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"`

---

### Task 9.3: Update CLAUDE.md command banners + package maps

§14 step 9 and §11 require the docs to reflect the new surface: the production commands
(`update`/`status`/`watch`/`snapshot`), the bootstrap **script** replacing the legacy build chain,
and the new modules (`nfp_ingest/qcew_acquire.py`, `nfp_ingest/capture.py`,
`nfp_vintages/calendar.py`, `nfp_vintages/store_status.py`, `nfp_download/release_dates/feed.py`).
This is a docs-only task (no code, no test).

**Files:**
- Modify: `/Users/lowell/Projects/alt-nfp/CLAUDE.md`
- Modify: `/Users/lowell/Projects/alt-nfp/packages/nfp-vintages/CLAUDE.md`
- Modify: `/Users/lowell/Projects/alt-nfp/packages/nfp-ingest/CLAUDE.md`
- Modify: `/Users/lowell/Projects/alt-nfp/packages/nfp-download/CLAUDE.md`

- [ ] **Step 1: (no failing test — docs only).** Grep to confirm the stale banners that must change
  are present before editing:
  `grep -rn "alt-nfp download\|alt-nfp process\|alt-nfp current\|alt-nfp build\|build-rebuild\|invoke_without_command\|bare alt-nfp" CLAUDE.md packages/*/CLAUDE.md`
  Expected: matches in root `CLAUDE.md` (the `uv run alt-nfp --help` command banner region + the
  "Rebuild to scratch; promote deliberately" hard-rule bullet's "never `alt-nfp build` straight to
  `…/store`" sentence) and `packages/nfp-vintages/CLAUDE.md` (the "Key Commands" fenced block + the
  `__main__.py` map line + the "CLI (`__main__.py`): typer app with subcommands: `download`,
  `download-indicators`, `process`, `current`, `build`" bullet). These are the lines to rewrite.

- [ ] **Step 2: Edit root `CLAUDE.md`.** In the `## Commands` block, after the existing
  `uv run alt-nfp --help` line, document the production surface and the bootstrap script:

```markdown
uv run alt-nfp update --as-of 2026-01-12       # capture knowable month-T prints, append to store
uv run alt-nfp status                          # store coverage + uncaptured/corrected alarm
uv run alt-nfp watch --source all              # BLS-feed-driven trigger (cron)
uv run python scripts/bootstrap_store.py \      # one-time historical rebuild + promote (NOT a command)
    --scratch s3://alt-nfp/store-rebuild --canonical s3://alt-nfp/store
```

  And in the **Hard rules → "Rebuild to scratch; promote deliberately"** bullet, replace the
  "never `alt-nfp build` straight to `…/store`" sentence: the everyday CLI no longer has a `build`
  command; the one-time rebuild path is `scripts/bootstrap_store.py` (scratch prefix → deliberate
  copy-then-delete promote), with the `is_canonical_store` guard still refusing a canonical scratch
  target.

- [ ] **Step 3: Edit `packages/nfp-vintages/CLAUDE.md`.** Replace the entire "Key Commands" fenced
  block so it describes the production surface and the script (drop the bare-`alt-nfp`/`build
  --allow-canonical` note, which referenced the retired bare-run):

````markdown
## Key Commands

```bash
# Production month-T workflow (specs/cli_production_workflow.md)
uv run alt-nfp update --as-of 2026-01-12 [--only ces|qcew|indicators]  # capture + append
uv run alt-nfp status [--as-of 2026-01-12] [--store URI]               # coverage report
uv run alt-nfp watch [--source ces|qcew|all] [--snapshot]              # feed-driven (cron)
uv run alt-nfp snapshot --as-of 2026-01-12 [--grid-end 2026-06-12]     # hash-pinned ModelData

# One-time historical rebuild + promote — a SCRIPT, not a CLI command:
uv run python scripts/bootstrap_store.py --scratch s3://alt-nfp/store-rebuild \
    --canonical s3://alt-nfp/store

# Run vintage tests / lint
pytest src/nfp_vintages/tests/
ruff check src/nfp_vintages/
```
````

  Update the package-structure comment for `__main__.py` to
  `# CLI entry point (update/status/watch/snapshot; legacy build chain retired §10)` and add two map
  lines under `src/nfp_vintages/`:
  `├── calendar.py             # advance_release_calendar() — release-calendar scrape (lifted from __main__)`
  and
  `├── store_status.py         # compute_status()/format_status() — read-only coverage report (status)`.
  Also rewrite the **Key Patterns** "CLI (`__main__.py`): typer app with subcommands: `download`,
  `download-indicators`, `process`, `current`, `build`. Each step is idempotent." bullet to list
  `update`/`status`/`watch`/`snapshot`, note the legacy stage commands were retired (§10), and that
  the one-time bootstrap is now `scripts/bootstrap_store.py`.

- [ ] **Step 4: Edit `packages/nfp-ingest/CLAUDE.md` and `packages/nfp-download/CLAUDE.md`.** In
  `nfp-ingest`'s package-structure map, add (after `vintage_store.py`):
  `├── qcew_acquire.py         # acquire_qcew_levels()/acquire_qcew_size_native() — CEW API slices (was private in nfp-vintages)`
  and
  `├── capture.py              # capture_ces_print()/capture_qcew_quarter() — month-T current-print → store (update)`.
  In `nfp-download`'s `release_dates/` map, add:
  `│   ├── feed.py              # parse_feed()/fetch_feed() — BLS empsit/cewqtr RSS (curl_cffi impersonation, for watch)`.
  Then verify no banner still advertises a retired command:
  `grep -rn "alt-nfp download\|alt-nfp process\|alt-nfp current\|alt-nfp build\|build-rebuild" CLAUDE.md packages/*/CLAUDE.md`
  Expected: **no matches** (every legacy-command reference rewritten to the production surface or the
  script). (The Phase 1/4/3/8 modules may already be documented if their phases edited these maps;
  if a line is present, leave it — do not duplicate.)

- [ ] **Step 5: Commit** —
  `git add CLAUDE.md packages/nfp-vintages/CLAUDE.md packages/nfp-ingest/CLAUDE.md packages/nfp-download/CLAUDE.md`
  then
  `git commit -m "docs(cli): update CLAUDE.md banners/maps for production workflow + bootstrap script (§14.9)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"`

---

**Phase 9 done-when:**
- `scripts/bootstrap_store.py` exists, is importable, and `uv run python scripts/bootstrap_store.py
  --help` prints argparse usage (it is a **script**, not a Typer command); its smoke test
  (`test_bootstrap_store.py`) passes with every heavy/network seam monkeypatched and both store
  prefixes pinned to `tmp_path` — the real bootstrap is never run against MinIO. The bootstrap writes
  **nothing** under `./data` (raw extract → tempdir; store write delegated to the container-safe
  `write_rebuild_store`), targets a **scratch** prefix, and the `is_canonical_store` refusal fires on
  a canonical `--scratch`.
- `alt-nfp` registers only `update`/`status`/`watch`/`snapshot`; `download`/`download-indicators`/
  `process`/`current`/`build`/`build-rebuild` and the bare-run chain are gone (`test_cli_legacy_retired.py`
  passes); a bare `alt-nfp` prints usage + exits non-zero (no build).
- All four CLAUDE.md files describe the production surface + the bootstrap script; `grep` finds no
  retired-command banner.
- `uv run pytest -m "not network and not slow" --no-cov` green (no orphaned imports from the
  retirement); `uv run ruff check .` clean.

---

## Execution order

Phase 9 is the **last** phase; run its tasks strictly in order (each later task assumes the earlier
landed):

1. **Task 9.1** — `scripts/bootstrap_store.py` + smoke test (the rebuild lineage must exist before
   `build`/`build-rebuild` are retired, so the rebuild path is never orphaned).
2. **Task 9.2** — retire the legacy commands + bare-run from `__main__.py` (now safe: every reusable
   body has a new home — Phase 1 `qcew_acquire`, Phase 3 `calendar`, Task 9.1 `bootstrap_store`).
3. **Task 9.3** — docs (banners/maps) reflect the surface only after the code change lands.

Each task is a full TDD cycle: failing test → confirm-fail with the stated reason → complete
implementation (no placeholders) → confirm-pass → package suite `-m "not network and not slow"` →
`ruff check` → scoped `git add <exact paths>` + commit ending
`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. (Task 9.3 is docs-only: grep-confirm the
stale banner instead of a failing test.)

**Plan-level done-when (run before declaring the whole plan complete):**
- `uv run pytest -m "not network and not slow" --no-cov` — full fast suite green (the bootstrap smoke
  + legacy-retired tests pass; no orphaned imports from the legacy deletion).
- `uv run ruff check .` — clean (E,W,F,I,B,C4,UP, line 100).
- Manual: `uv run python scripts/bootstrap_store.py --help` prints argparse usage; `uv run alt-nfp
  --help` lists only `update`/`status`/`watch`/`snapshot`; `grep -rn "alt-nfp build\b\|build-rebuild"
  CLAUDE.md packages/*/CLAUDE.md` returns nothing.

---

## Verification appendix (current-source evidence — line refs are PRE-Phase-5)

These are the spans in the **current** `__main__.py` (before any Phase 5–8 edit lands). They WILL
shift once Phases 5–8 add `update`/`status`/`watch` and re-shape `snapshot`; re-grep before deleting.

- Callback w/ bare-run chain: `__main__.py:24-33` (`invoke_without_command=True` → calls
  `download(); download_indicators(); process(); current(); build(None, allow_canonical=False)`).
- `from pathlib import Path`: `__main__.py:16` (only the retired `build` uses it).
- `download`: `:36-47` · `download-indicators`: `:50-58` · `process`: `:61-78` · `current`: `:80-86`
  · `build`: `:89-106` · `build-rebuild`: `:109-181` · **`snapshot` (KEEP):** `:184-217` · tail
  `if __name__ == '__main__': app()`: `:220-221`.
- `process` already calls `advance_release_calendar` (`:64,69`), no in-file `_build_release_calendar`
  helper (Phase 3 already lifted it) — retiring `process` is just deleting the command.
- `build-rebuild` imports `compose_rebuild_panel`/`write_rebuild_store` (`:151-154`) and the
  acquire helpers from `nfp_ingest.qcew_acquire` (`:144-147`) — all of which the bootstrap now owns.
