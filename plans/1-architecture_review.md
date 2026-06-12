# Implementation Plan: Architecture Review Fixes

Source: deep architecture review of the four-package workspace (2026-06-12).

**Verdict from the review:** the package structure (lookups → download → ingest → vintages)
is sound and worth keeping. This plan fixes the boundary leaks and port residue found,
in dependency order. Tiers 1–3 are safe to do now; Tier 4 items are deliberately
deferred to Phase A2 per the port plan (after golden-master tests exist).

**Global acceptance criteria (apply after every tier):**

- `uv run pytest -m "not network" --no-cov` — all tests pass (baseline: 322 passed, 23 skipped)
- `uv run ruff check packages/` — clean
- No `nfp_lookups` module imports any other `nfp_*` package (enforced by Tier 1)

---

## Tier 1 — Boundary fixes (do first; every week of new code makes these costlier)

> **Status: ✅ COMPLETE (2026-06-12).** All four items done; suite green
> (324 passed, 23 skipped — up 2 from new re-export smoke tests), ruff clean.
> Verified: no executable `alt_nfp` imports remain; `nfp_lookups` imports no
> other `nfp_*` package; `release_dates.config` path imports reduced to
> scraper-internal + `PUBLICATIONS` only. Notes per item below.
> **Bonus fix found during verification:** `nfp_download/bls/qcew.py` had a
> second dead `from alt_nfp.lookups.geography import STATES` lazy import — a
> live `ModuleNotFoundError` on the `state_fips_list=None` path of
> `fetch_qcew_with_geography` (only network-marked tests exercise it, so the
> suite never caught it). Replaced with a top-level `nfp_lookups.geography`
> import.

### 1.1 Fix the layering inversion: move series-ID grammar into nfp-lookups ✅

**Problem:** `packages/nfp-lookups/src/nfp_lookups/industry.py:474` lazy-imports
`build_series_id` from `nfp_download.bls`, inverting the dependency chain
(download already depends on lookups). Its `except ImportError` fallback imports
`alt_nfp.ingest.bls` — the old repo's module path, which can never resolve here.

**Steps:**

1. Move the series-ID construction/parsing logic from
   `packages/nfp-download/src/nfp_download/bls/_programs.py` into a new
   `packages/nfp-lookups/src/nfp_lookups/series_ids.py`
   (it is static BLS series-ID grammar — reference knowledge, not I/O).
2. In `nfp_download/bls/_programs.py`, re-export from `nfp_lookups.series_ids`
   so existing `from nfp_download.bls import build_series_id` call sites keep working
   (`nfp_download/bls/__init__.py` re-exports stay unchanged).
3. In `nfp_lookups/industry.py`, replace the lazy import + dead `alt_nfp` fallback
   with a normal top-level `from nfp_lookups.series_ids import build_series_id`.
4. Move the relevant tests: series-ID grammar cases in
   `packages/nfp-download/tests/bls/test_programs.py` can stay (they test the
   re-export) or move to nfp-lookups tests — prefer moving, keeping a thin
   re-export smoke test in nfp-download.

**Verify:** grep shows no `nfp_download` import anywhere under `packages/nfp-lookups/src/`;
no `alt_nfp` string anywhere in `packages/`.

**Done:** grammar moved to `nfp_lookups/series_ids.py`; `_programs.py` is now a
re-export shim; `industry.py` uses a top-level same-package import (dead
`alt_nfp` fallback deleted). Grammar tests moved to
`packages/nfp-lookups/tests/test_series_ids.py`; thin re-export smoke test kept
at `packages/nfp-download/tests/bls/test_programs.py`. Also exported the
series-ID API (`PROGRAMS`, `build_series_id`, `parse_series_id`, `get_program`,
`list_programs`) from `nfp_lookups/__init__.py`. Verify criterion amended:
"no `alt_nfp` string" holds for *executable imports*; remaining docstring
mentions are queued in Tier 3.5.

### 1.2 Move pipeline artifact paths into nfp_lookups.paths ✅

**Problem:** `RELEASE_DATES_PATH`, `VINTAGE_DATES_PATH`, `RELEASES_DIR` live in
`packages/nfp-download/src/nfp_download/release_dates/config.py`; six modules across
nfp-ingest and nfp-vintages import nfp_download just to learn where an intermediate
parquet sits. `nfp_lookups.paths` is documented as the canonical layout.

**Steps:**

1. Add `RELEASES_DIR`, `RELEASE_DATES_PATH`, `VINTAGE_DATES_PATH` to
   `packages/nfp-lookups/src/nfp_lookups/paths.py` (derived from `DOWNLOADS_DIR` /
   `INTERMEDIATE_DIR` as today).
2. In `nfp_download/release_dates/config.py`, import them from `nfp_lookups.paths`
   and keep re-exports for back-compat (scraper-specific `Publication` config stays put).
3. Update importers to pull from `nfp_lookups.paths` directly:
   - `nfp_ingest/release_dates/vintage_dates.py`
   - `nfp_ingest/releases.py`
   - `nfp_ingest/tagger.py`
   - `nfp_vintages/processing/ces_national.py`
   - `nfp_vintages/processing/qcew.py`
   - `nfp_vintages/processing/sae_states.py`
   - `nfp_vintages/__main__.py` (keeps importing scraper config from nfp_download — fine)

**Verify:** `grep -rn 'release_dates.config import' packages/*/src` shows only
scraper-related imports (`Publication`, `PUBLICATIONS`, `BASE_URL`, `START_YEAR`).

**Done:** all three constants now live in `nfp_lookups.paths`; `config.py`
re-exports them with a `# noqa: F401` back-compat note. All seven importers
updated (six direct + the split import in `nfp_vintages/__main__.py`).

### 1.3 Make `_CES_SECTOR_TO_NAICS` public ✅

**Problem:** `nfp_vintages/processing/sae_states.py` and
`nfp_vintages/processing/ces_national.py` import the underscore-private
`_CES_SECTOR_TO_NAICS` from `nfp_lookups.industry` across a package boundary.

**Steps:**

1. Rename to `CES_SECTOR_TO_NAICS` in `nfp_lookups/industry.py`; keep
   `_CES_SECTOR_TO_NAICS = CES_SECTOR_TO_NAICS` as a deprecated alias if any
   old-repo reference code matters, otherwise just rename.
2. Update the two nfp-vintages importers (and any internal lookups usage).
3. Consider exporting it from `nfp_lookups/__init__.py` alongside the other
   industry maps.

**Done:** renamed everywhere (definition, internal use, both nfp-vintages
processing modules, `test_store_coverage.py`, nfp-lookups CLAUDE.md). No
deprecated alias kept — zero references to the old name remain. Exported from
`nfp_lookups/__init__.py`.

### 1.4 Derive `VINTAGE_STORE_PATH` from lookups ✅

**Problem:** `nfp_vintages/build_store.py` imports `VINTAGE_STORE_PATH` from
`nfp_ingest.vintage_store`. Store *API* ownership by ingest is fine; the *path*
should come from the canonical layout.

**Steps:**

1. Define `VINTAGE_STORE_PATH` in `nfp_lookups/paths.py` (derived from `STORE_DIR`,
   matching the current value in `nfp_ingest/vintage_store.py`).
2. `nfp_ingest/vintage_store.py` imports it from there (keep re-export for callers).
3. `nfp_vintages/build_store.py` imports from `nfp_lookups.paths`.

**Done:** `VINTAGE_STORE_PATH` defined in `nfp_lookups.paths` as a named alias
of `STORE_DIR`; `vintage_store.py` imports it (name still re-exported from that
module for existing callers, e.g. tests); `build_store.py` no longer imports
from nfp_ingest for a path.

---

## Tier 2 — Robustness (CI + path discovery)

> **Status: ✅ COMPLETE (2026-06-12).** Suite at 331 passed / 23 skipped
> (+7 new `test_paths.py` tests), ruff clean. CI workflow created; its exact
> commands verified locally — first real run happens on push.

### 2.1 Add a CI workflow ✅

**Problem:** no `.github/` exists; the port plan says CI is carried over. The suite
is green and fast (~2s) — lock that in before the model phase starts.

**Steps:**

1. Create `.github/workflows/ci.yml`:
   - trigger: push + pull_request on `main`
   - steps: checkout → `astral-sh/setup-uv` → `uv sync` →
     `uv run ruff check .` → `uv run pytest -m "not network" --no-cov`
   - Python 3.12, ubuntu-latest; cache uv.
2. Store-dependent tests already self-skip when `data/` is absent (verified), and
   network tests are marked — no CI-specific test changes needed.

**Done:** `.github/workflows/ci.yml` — checkout → `astral-sh/setup-uv@v5`
(cached) → `uv sync` → `uv run ruff check .` → `uv run pytest -m "not network"
--no-cov`.

### 2.2 Fix `_find_base_dir()` in nfp_lookups.paths ✅

**Problem:** the heuristic looks for a parent containing `data/` + `pyproject.toml`,
but `data/` is gitignored — on a fresh clone the walk always fails and silently hits
the `parents[4]` fallback (correct only for editable installs). Works by coincidence.

**Steps:**

1. Add env override: if `NFP_BASE_DIR` is set, use it (this is also what the
   Phase A2 snapshot workflow will want for pointing at fixture/snapshot dirs).
2. Change the marker to something committed: a parent containing `packages/`
   **and** `uv.lock` (or `pyproject.toml`). Keep `parents[4]` as final fallback.
3. Add a unit test in `packages/nfp-lookups/tests/` covering: env override wins;
   marker discovery from the real tree; fallback depth.

**Done:** precedence is `NFP_BASE_DIR` (expanduser + resolve) → walk-up to
first dir with `packages/` + `pyproject.toml` (committed markers) →
`parents[4]` fallback. 7 tests in
`packages/nfp-lookups/tests/test_paths.py` (discovery precedence + derived
layout invariants).

---

## Tier 3 — Port-residue cleanup (mechanical hygiene)

> **Status: ✅ COMPLETE (2026-06-12).** Notes and deviations per item below.
> Extras beyond the planned items: created a root `CLAUDE.md` (workspace map,
> commands, boundary rules) and documented the paths/series-ids patterns +
> new test files in `packages/nfp-lookups/CLAUDE.md`.

### 3.1 Delete dead files ✅

- `main.py` at repo root (uv init boilerplate). **Done.**

### 3.2 Prune root `pyproject.toml` ✅

- Remove `[project.optional-dependencies]` entirely:
  - `dev` duplicates `[dependency-groups].dev` (the root is a virtual package with
    no build system; the extras are inert).
  - `viz` (matplotlib, arviz) is referenced by nothing in the workspace.
- Remove the `[[tool.mypy.overrides]]` block for `arviz/pymc/pytensor` and reword
  the `[tool.mypy]` comment (nothing here imports those; re-add JAX/numpyro/dynamax
  overrides when `nfp-model-jax` lands).

**Done:** both removed; mypy comment now points forward to JAX-era overrides.

### 3.3 Fix stale docs ✅

- `packages/nfp-lookups/src/nfp_lookups/industry.py:5` — docstring says "for PyMC";
  v2 is JAX. Reword to be model-agnostic ("for the model layer").
- `packages/nfp-vintages/src/nfp_vintages/evaluation.py:4,81` — same ("the PyMC model").
- `packages/nfp-download/CLAUDE.md` Tech Stack — remove "requests (BLS legacy)";
  the code is httpx-only.

**Done:** all three fixed (no `pymc`/`PyMC` mention left in `packages/`).

### 3.4 Root directory organization ✅ (deviated — see note)

- Move `Port and Extend Plan for the JAX NFP Nowcasting Model.md` and
  `Staged Progression to a Bayesian State Space NFP Nowcasting Model.md` into
  `docs/` (kebab-case the filenames while at it).
- Populate `specs/` from the old repo (`~/Projects/alt_nfp`) per the A0 plan, or
  delete the empty dir until needed. Same decision for `docs/` + the mkdocs
  dependency group (no `mkdocs.yml` exists yet — either add one or note it's deferred).

**Done, with deviations:** the two root plan docs were already reorganized by
hand before this step ran — the staged-progression doc now lives in `archive/`
(per the specs→archive convention) and the port-and-extend doc was removed
from the repo; both were left as found, no kebab-case renames applied.
Carried over from the old repo: `specs/*.md` (10 files) and the markdown scar
tissue from `archive/` (20 files; scripts/data/output deliberately left behind
per the port plan). `docs/` remains empty; **mkdocs setup is deferred** (deps
stay in the `docs` dependency group, no `mkdocs.yml` yet).

### 3.5 Fix stale `alt_nfp.*` docstring references (found during Tier 1) ✅

Tier 1 removed all *executable* `alt_nfp` imports, but docstrings still
reference old-repo module paths (`:mod:`~alt_nfp.…``, `python -m alt_nfp.…`).
Update to the `nfp_*` package paths:

- `packages/nfp-lookups/src/nfp_lookups/geography.py:9`
- `packages/nfp-ingest/src/nfp_ingest/{indicators.py:6, aggregate.py:5, tagger.py:3, payroll.py:8,13,160, qcew.py:407}`
- `packages/nfp-download/src/nfp_download/fred.py:5`
- `packages/nfp-vintages/src/nfp_vintages/{build_store.py:3,10, __main__.py:5, processing/qcew.py:7,22}`
- Test module docstrings: `test_lookups.py:1`, `test_ingest.py:1`,
  `test_release_dates.py:1`, `test_vintages.py:1`

**Done:** zero `alt_nfp` strings remain in `packages/`. One reference was also
corrected for accuracy, not just renamed: `indicators.py` now points at
`nfp_lookups.provider_config.CYCLICAL_INDICATORS_DEFAULT` (the constant's
actual name).

### 3.6 Write `README.md` ✅

This is a public repo; the README is empty. Minimum content:

- One-paragraph project description (Bayesian/JAX NFP nowcasting, port-and-extend of v1).
- Package diagram: `nfp-lookups → nfp-download → nfp-ingest → nfp-vintages` with
  one line each.
- Quickstart: `uv sync`, `uv run pytest -m "not network"`, `uv run alt-nfp --help`.
- Note that `data/` is proprietary and not in the repo.

**Done:** README covers all four points plus the `NFP_BASE_DIR` override and a
pointer to per-package CLAUDE.md maps. A root `CLAUDE.md` was also created
(workspace map, commands, hard rules: path centralization, no upward imports,
specs→archive workflow, frozen reference repo).

---

## Tier 4 — Deferred to Phase A2 (do NOT do now)

Sequenced after golden-master tests exist, per the port plan. Recorded here so the
rationale isn't lost:

- **Consolidate the duplicate download layer:** `nfp_vintages/download/{ces,qcew}.py`
  are acquisition (they already use `nfp_download.client` primitives) and belong in
  nfp-download.
- **Resolve naming collisions:** `ces_national.py` / `qcew.py` exist in both
  nfp-ingest (API current-data fetchers) and nfp-vintages/processing (bulk historical
  processors). Rename the vintages ones (e.g. `triangular_ces.py`, `qcew_bulk.py`)
  or pick one home per source during A2 consolidation.
- **CLI home:** the `alt-nfp` CLI lives in nfp-vintages but orchestrates all four
  packages. When `nfp-model-jax` arrives, either two CLIs exist or vintages must
  depend on the model package (wrong direction). Plan a thin top-level app/CLI
  package in A2.
- **ModelData snapshot boundary** (already in the port plan): one function answers
  "what was knowable on date D"; model layer consumes serialized arrays.

## Noted, no action planned

- `nfp_ingest/__init__.py` eagerly imports everything (drags httpx in for
  schema-only consumers). Harmless now; revisit if model-layer import time matters.
- Coverage runs on every local pytest via `addopts`. Fine at ~2s; move coverage to
  CI-only if the suite slows once JAX tests arrive.
- `vintage_store.py` (785 lines) and `qcew.py` (640 lines) are large but cohesive;
  no split needed.

## Suggested commit sequence

1. `Tier 1: fix package layering (series IDs → lookups, paths → lookups, public CES map)`
2. `Tier 2: add CI; harden base-dir discovery with NFP_BASE_DIR override`
3. `Tier 3: prune port residue (dead config, stale docs, README)`

Each commit independently passes the global acceptance criteria.

---

## Final state (all tiers 1–3 complete, 2026-06-12)

- Suite: **331 passed, 23 skipped, 6 network-deselected** (baseline was 322/23/6;
  +2 re-export smoke tests, +7 paths tests). `ruff check .` clean repo-wide.
- Boundary invariants verified: `nfp_lookups` imports no other `nfp_*` package;
  zero `alt_nfp` strings in `packages/`; pipeline paths centralized in
  `nfp_lookups.paths`.
- Tier 4 items remain deferred to Phase A2 by design. Work not yet committed —
  see commit sequence above.
