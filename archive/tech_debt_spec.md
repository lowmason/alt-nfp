# alt_nfp Technical Debt Audit

**Date:** 2026-03-15
**Repository:** `lowmason/alt_nfp`
**Codebase:** ~15K LOC across 5 packages, ~5.8K LOC tests, ~1.8K LOC scripts

---

## Executive Summary

The `alt_nfp` codebase is well-architected as a uv workspace with clean package boundaries (`nfp-lookups` → `nfp-download` → `nfp-ingest` → `nfp-vintages` → `nfp-models`). Configuration has been successfully migrated to Pydantic with TOML support. Type hints are present on all public functions. The primary debt clusters are: **dead code from the pre-package refactor** (2,098 LOC), **swallowed exceptions masking failures**, **print-based observability**, and **missing CI for tests/linting**.

---

## Prioritized Debt Items

### Priority scoring

- **Impact** (1–5): How much does it slow development or risk correctness?
- **Risk** (1–5): What happens if we don't fix it?
- **Effort** (1–5): How hard is the fix? (inverted: lower effort → higher priority score)
- **Score** = (Impact + Risk) × (6 − Effort)

---

### 1. Dead code: `scripts/_parts/` (2,098 LOC)

| Impact | Risk | Effort | Score |
|--------|------|--------|-------|
| 4      | 3    | 1      | **35** |

**What:** The entire `scripts/_parts/` directory contains the pre-refactor monolithic estimation script, split into 8 numbered files (`part01_preamble.py` through `part08_forecast_main.py`). These are **never imported anywhere** — no script, test, or package references them. Every function (`build_model`, `sample_model`, `run_loo_cv`, `print_diagnostics`, `forecast_and_plot`, `plot_residuals`, etc.) has a canonical replacement in the `nfp-models` package.

**Why it matters:** Developers (and Claude Code) can't tell which `build_model` is authoritative. Grep results return two hits for every key function, creating confusion during maintenance. The 2,098 lines inflate the apparent codebase by ~14%.

**Fix:** `rm -rf scripts/_parts/`. Confirm nothing breaks by running the test suite.

---

### 2. Swallowed exceptions (10 instances)

| Impact | Risk | Effort | Score |
|--------|------|--------|-------|
| 4      | 5    | 2      | **36** |

**What:** Ten `except Exception:` blocks across the codebase silently discard errors:

- `panel_adapter.py:495` — file read failure → silently returns `None`
- `diagnostics.py:41` — tree depth extraction → `pass`
- `forecast.py:286` — plotting provider data → `pass`
- `plots.py:207` — plotting provider data → `pass`
- `indicators.py:54` — indicator read failure → returns `None` (at least logs)
- `indicators.py:104` — download failure → prints but continues
- `panel.py:128` — git hash retrieval → `pass`
- `sae_states.py:377` — saves checkpoint then re-raises (actually fine)
- `export_data.py:69` — unknown context

**Why it matters:** Silent failures in `panel_adapter.py` mean a corrupt or missing provider parquet silently produces `None`, which propagates into the model data dict and can cause cryptic PyMC errors far downstream. The plotting `pass` blocks mean visual diagnostics silently omit provider series without warning — you could be missing a provider from every plot and never notice.

**Fix:** Phase 1: Replace `except Exception: pass` in `diagnostics.py`, `forecast.py`, and `plots.py` with `except Exception as e: logger.debug(...)` so failures are traceable. Phase 2: In `panel_adapter.py`, catch `FileNotFoundError` and `pl.exceptions.ComputeError` specifically, and log at `WARNING` level with the file path.

---

### 3. No CI for tests or linting

| Impact | Risk | Effort | Score |
|--------|------|--------|-------|
| 4      | 4    | 2      | **32** |

**What:** The only GitHub Actions workflow is `deploy-docs.yml` for MkDocs. There is no CI job running `pytest`, `ruff`, `mypy`, or any other checks on PRs or pushes.

**Why it matters:** Regressions can land in `main` uncaught. The repo already has `ruff` and `mypy` configured in `pyproject.toml` but they're only useful if someone runs them locally. Given the model-correctness stakes (this feeds published employment estimates), automated validation is essential.

**Fix:** Add a `ci.yml` workflow: `uv sync` → `uv run pytest` → `uv run ruff check` → `uv run mypy packages/`. Effort is low because the tooling is already configured.

---

### 4. Print-based observability (252 `print()` calls in packages)

| Impact | Risk | Effort | Score |
|--------|------|--------|-------|
| 3      | 3    | 3      | **18** |

**What:** The package source code contains 252 `print()` calls. Some modules (`nfp-ingest`) properly use `logging.getLogger(__name__)`, but `nfp-models` modules (especially `diagnostics.py`, `forecast.py`, `benchmark.py`, `plots.py`) use `print()` almost exclusively.

**Why it matters:** Print output can't be filtered, leveled, or redirected. During backtesting (which runs the model dozens of times), print spam is noise. When debugging a specific provider's measurement equation, there's no way to bump one module to DEBUG without getting everything.

**Fix:** Incremental migration — when touching a file for other reasons, replace `print()` with `logger.info()` / `logger.debug()`. Priority targets: `diagnostics.py` (716 LOC, heavily print-based), `benchmark.py`, `forecast.py`.

---

### 5. Tracked binary artifacts in git (30+ PNGs, parquets)

| Impact | Risk | Effort | Score |
|--------|------|--------|-------|
| 3      | 2    | 1      | **25** |

**What:** `output/` and `archive/output/` contain PNGs and parquet files tracked in git. The `archive/` directory alone is 11MB. These are model outputs that change every run — they inflate clone size and create noisy diffs.

**Why it matters:** Every clone downloads ~14MB of binary blobs. The archive exists for reference but should live outside the git history.

**Fix:** Add `output/*.png`, `output/*.parquet`, `archive/output/` to `.gitignore`. Run `git rm --cached` on the tracked files. Consider keeping one `archive/` snapshot as a release artifact on GitHub Releases instead.

---

### 6. Dual HTTP client libraries (`requests` + `httpx`)

| Impact | Risk | Effort | Score |
|--------|------|--------|-------|
| 2      | 2    | 2      | **16** |

**What:** `nfp-download` depends on both `httpx[http2]` and `requests`. The legacy `_http.py` module (ported from `eco-stats`) uses `requests`, while the newer `client.py` and everything in `nfp-vintages` uses `httpx`. This means two HTTP stacks, two retry implementations, two user-agent strings, and two connection pool strategies.

**Why it matters:** Maintenance burden is doubled for HTTP concerns (timeouts, retries, rate limiting, API key injection). The `requests`-based `_http.py` doesn't support HTTP/2.

**Fix:** Port `_http.py` from `requests` to `httpx`, unifying on the `client.py` retry/header infrastructure. Then remove the `requests` dependency from `nfp-download/pyproject.toml`.

---

### 7. Legacy config shim (`nfp_models/config.py`)

| Impact | Risk | Effort | Score |
|--------|------|--------|-------|
| 2      | 2    | 3      | **12** |

**What:** `config.py` exists as a backwards-compatibility shim that instantiates a `NowcastConfig()` default and re-exports ~25 module-level constants (`LOG_SIGMA_QCEW_MID_MU`, `N_HARMONICS`, `ERA_BREAKS`, etc.). Callers `from .config import LOG_TAU_MU` instead of accessing `cfg.model.latent.log_tau_mu`.

**Why it matters:** The shim works but means two ways to access every constant. New code might use the config object; old code uses the flat imports. The shim also hides the fact that these values can be overridden via TOML — callers using the flat constants always get defaults.

**Fix:** Migrate callers in `model.py`, `diagnostics.py`, `residuals.py`, `backtest.py`, `benchmark_backtest.py` to accept a `NowcastConfig` parameter (most already do). Then deprecate or remove the flat constants.

---

### 8. Test coverage gaps (14 untested modules)

| Impact | Risk | Effort | Score |
|--------|------|--------|-------|
| 3      | 4    | 4      | **14** |

**What:** 14 of 54 source modules have no corresponding test file or test imports:

- **nfp-models:** `residuals.py`, `plots.py`, `checks.py`, `benchmark_plots.py`
- **nfp-vintages:** `build_store.py`, `views.py`, `sae_states.py`, `__main__.py`, `evaluation.py`
- **nfp-lookups:** `revision_schedules.py`, `paths.py`, `provider_config.py`
- **nfp-download:** `scraper.py`, `client.py`

**Why it matters:** `sae_states.py` (560 LOC, SAE data processing) and `build_store.py` (vintage store construction) are critical pipeline components. `revision_schedules.py` and `provider_config.py` are configuration modules where a typo could silently change model behavior.

**Fix:** Prioritize by risk: (1) `revision_schedules.py` and `provider_config.py` (static data, easy to unit test), (2) `client.py` (HTTP retry logic), (3) `sae_states.py` and `build_store.py` (integration tests with fixture data). Plot/viz modules are lower priority.

---

### 9. `pyproject.toml` metadata issues

| Impact | Risk | Effort | Score |
|--------|------|--------|-------|
| 1      | 2    | 1      | **15** |

**What:** Three issues in the root `pyproject.toml`:

1. **Placeholder author:** `{name = "Your Name", email = "your.email@example.com"}`
2. **Duplicate entry point:** Both root and `nfp-vintages` define `alt-nfp` CLI entry point, but the root points to `alt_nfp.vintages.__main__:app` (a non-existent module path) while `nfp-vintages` correctly points to `nfp_vintages.__main__:app`
3. **`disallow_untyped_defs = false`** in mypy config — the codebase actually has return types on all public functions, so this could be tightened

**Fix:** Update author, remove the broken root entry point, set `disallow_untyped_defs = true` in mypy.

---

### 10. Missing type annotations on internal helpers (~108 functions)

| Impact | Risk | Effort | Score |
|--------|------|--------|-------|
| 2      | 1    | 4      | **6** |

**What:** ~108 of ~286 function definitions across packages lack return type annotations (mostly private helpers, `__init__` methods, and plotting functions).

**Why it matters:** Low practical impact since public APIs are typed, but blocks enabling `disallow_untyped_defs = true` in mypy.

**Fix:** Low priority. Add annotations incrementally when modifying files. Focus on `model.py` and `panel_adapter.py` first since those are the most-edited modules.

---

## Phased Remediation Plan

### Phase 1 — Quick wins (1 session, alongside feature work)

- [ ] Delete `scripts/_parts/` (item 1)
- [ ] Fix `pyproject.toml` metadata (item 9)
- [ ] Add `output/*.png`, `output/*.parquet`, `archive/output/` to `.gitignore` and `git rm --cached` (item 5)

### Phase 2 — Safety net (1–2 sessions)

- [ ] Add CI workflow for pytest + ruff + mypy (item 3)
- [ ] Replace bare `except Exception` blocks with specific catches + logging (item 2)
- [ ] Write tests for `revision_schedules.py`, `provider_config.py`, `client.py` (item 8, high-value subset)

### Phase 3 — Consolidation (spread across feature work)

- [ ] Port `_http.py` from `requests` to `httpx` (item 6)
- [ ] Migrate config shim callers to `NowcastConfig` (item 7)
- [ ] Convert `print()` to `logging` in touched files (item 4)
- [ ] Add return type annotations incrementally (item 10)