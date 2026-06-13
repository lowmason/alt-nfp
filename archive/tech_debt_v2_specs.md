# Repository Hardening Spec

**Alt-NFP Workspace - Runtime Correctness, Documentation Alignment, and Test Hygiene**

Version: 1.0 | Date: 2026-03-16 | Author: Lowell Mason

---

## 1. Background and Motivation

The 2026-03-16 repository review found a small number of high-impact issues that are now more important than incremental cleanup:

1. Provider signals are being omitted from default panel builds in several application paths.
2. User-facing docs still describe a monolithic `alt_nfp` package that no longer exists.
3. The MkDocs reference pipeline still targets `src/alt_nfp/`, so strict docs builds fail.
4. Default `pytest` execution runs live network tests that depend on external services and secrets.
5. The BLS flat-file downloader uses a weaker HTTP path than the shared retry-aware client already present in the repo.
6. Several stale or half-removed code paths continue to create confusion during maintenance.

These are not cosmetic issues. They affect model inputs, developer trust in docs, reproducibility of local validation, and the reliability of the download pipeline.

This spec complements the broader audit in `archive/tech_debt_spec.md`. That audit identified general debt themes; this spec is the concrete remediation plan for the correctness and maintenance issues found in the latest review.

---

## 2. Goals

1. Restore correct runtime behavior so application entry points include configured provider data.
2. Align all user-facing docs with the actual split-package workspace layout.
3. Make the docs build pass in strict mode.
4. Make default local validation deterministic and secret-free.
5. Harden the BLS download path around shared retry/header behavior.
6. Remove or quarantine stale code that obscures the source of truth.

### 2.1 Non-Goals

- No changes to the statistical model itself.
- No SAE feature re-enablement in this phase.
- No resurrection of a monolithic runtime `alt_nfp` package in this phase.

### 2.2 Canonical Architecture Decision

The canonical Python import surface remains the existing workspace packages:

- `nfp_lookups`
- `nfp_download`
- `nfp_ingest`
- `nfp_vintages`
- `nfp_models`

This spec standardizes docs and tooling around that reality. A future facade package named `alt_nfp` may be added later if desired, but that is explicitly out of scope here.

---

## 3. Problem Summary

| Area | Current problem | User-visible impact |
|---|---|---|
| Panel construction | `build_panel()` defaults to no providers, but app code assumes configured providers are included | Main estimation and backtest paths silently omit provider signals |
| Public docs | README/docs still use `import alt_nfp` and `src/alt_nfp/` | Quickstart examples fail and API docs mislead users |
| Docs tooling | `docs/gen_ref_pages.py` and `mkdocs.yml` target a removed source tree | `mkdocs build --strict` aborts |
| Test defaults | `pytest` includes live network tests by default | Default test runs fail on BLS 403s or missing `FRED_API_KEY` |
| Downloader | BLS flat-file client uses bare `httpx.Client` without shared retry/header behavior | Live downloads are more brittle than they need to be |
| Stale code | Dead path hacks and large commented blocks remain in active modules | Engineers cannot easily tell what is authoritative |

---

## 4. Implementation Plan

The work is organized into five phases. Phases 1-3 should land together or in close sequence because they affect the main validation loop. Phases 4-5 can follow immediately after.

### 4.1 Phase 1 - Runtime Correctness

**Objective:** Ensure configured providers are included wherever the application intends to use them.

#### 4.1.1 Design

`nfp_ingest.build_panel()` is a low-level ingestion function. It should remain package-pure and should not import `nfp_models.settings` or derive providers implicitly from model config.

Instead, application-level code must pass providers explicitly from configuration.

#### 4.1.2 Changes

1. Add a small application-level helper that resolves config into panel inputs.
   - Recommended location: `packages/nfp-model-hmc/src/nfp_models/pipeline.py`
   - Recommended function:

```python
def build_panel_from_config(
    cfg: NowcastConfig,
    *,
    start_year: int = 2012,
    end_year: int | None = None,
    as_of_ref: date | None = None,
) -> pl.DataFrame:
    ...
```

2. Update all application entry points to use this helper or to pass `providers_from_settings(cfg)` directly.
   - `alt_nfp_estimation_v3.py`
   - `packages/nfp-model-hmc/src/nfp_models/backtest.py`
   - `packages/nfp-model-hmc/src/nfp_models/benchmark_backtest.py`
   - `packages/nfp-model-hmc/src/nfp_models/sensitivity.py`
   - `scripts/jan_reforecast.py`
   - any other script currently calling `build_panel()` with no providers

3. Correct `build_panel()` docstrings to state the real behavior:
   - `providers=None` means no provider rows are added
   - configured providers must be passed explicitly by application code

4. Remove the dead repo-root `src` path insertion from `alt_nfp_estimation_v3.py`.

#### 4.1.3 Tests

Add regression coverage that proves provider rows are present when config-derived providers are used:

- unit test for `build_panel_from_config()`
- smoke test that asserts provider source `"g"` appears in the panel when local provider data exists
- regression test for the estimation pipeline input path that ensures the panel is not CES/QCEW-only

#### 4.1.4 Acceptance Criteria

- `build_panel_from_config(NowcastConfig())` returns a panel containing source `"g"` when local provider data exists.
- All application paths that expect provider data pass providers explicitly.
- No active code relies on the nonexistent repo-root `src/` directory.

---

### 4.2 Phase 2 - Documentation and API Alignment

**Objective:** Make docs truthful and make the docs toolchain match the current package layout.

#### 4.2.1 Design

User-facing docs will reference the split workspace packages directly instead of the removed `alt_nfp` package.

Examples:

- `from nfp_ingest import build_panel`
- `from nfp_models.panel_adapter import panel_to_model_data`
- `from nfp_models import run_backtest` only if that symbol actually exists

The API reference will be generated from `packages/*/src`, not from `src/alt_nfp/`.

#### 4.2.2 Changes

1. Rewrite user-facing examples and structure descriptions in:
   - `README.md`
   - `CLAUDE.md`
   - `docs/getting-started/*.md`
   - `docs/user-guide/*.md`
   - `docs/architecture/*.md`

2. Remove or rewrite references to modules that are no longer part of the active package tree.
   - `alt_nfp.data`
   - `alt_nfp.lookups.update_schedule`
   - `src/alt_nfp/...`
   - any stale `use_legacy=True` examples for `build_panel()`

3. Rework `docs/gen_ref_pages.py` to discover workspace packages under `packages/*/src`.
   - Generate pages under `docs/reference/<package_name>/...`
   - Skip private modules as today

4. Update `mkdocs.yml`:
   - point `mkdocstrings` at all package `src` directories
   - remove hard-coded `reference/alt_nfp/...` entries
   - choose one nav strategy for generated API docs and use it consistently

5. Update public docstrings and cross-references in actively documented modules where needed so generated references resolve.
   - first pass should focus on modules surfaced in API docs, not every historical/internal comment

#### 4.2.3 Tests

- add a docs smoke target in CI/local validation:
  - `uv run --group docs mkdocs build --strict`
- add a lightweight import smoke check in docs/examples where appropriate

#### 4.2.4 Acceptance Criteria

- No user-facing docs instruct users to `import alt_nfp`.
- `uv run --group docs mkdocs build --strict` passes with zero warnings.
- Generated API docs cover the actual workspace packages.

---

### 4.3 Phase 3 - Test and CI Hygiene

**Objective:** Make default validation reliable without network access or credentials.

#### 4.3.1 Design

Network tests must be opt-in. Local default test runs and PR CI runs must be green without:

- internet access
- BLS availability
- `FRED_API_KEY`

#### 4.3.2 Changes

1. Change default pytest behavior so non-network tests are the default.
   - preferred approach: add `-m "not network"` to default test command or CI command
   - acceptable alternative: custom `--run-network` option

2. Harden network tests so they self-skip when prerequisites are absent.
   - FRED integration test must skip when `FRED_API_KEY` is unset
   - BLS integration tests should remain marked `network` and should never run in the default suite

3. Split validation into two documented lanes:
   - default local/CI lane: lint + unit/integration tests without network + docs build
   - optional live-data lane: network tests only

4. Add CI guardrails if not already present.
   - `uv run pytest`
   - `uv run ruff check ...`
   - `uv run --group docs mkdocs build --strict`

#### 4.3.3 Tests

- verify `uv run pytest` passes in a clean environment without credentials
- verify `uv run pytest -m network` still collects and runs live tests when explicitly requested

#### 4.3.4 Acceptance Criteria

- Default `pytest` execution succeeds without external secrets.
- Network tests remain available but are never part of the default validation path.
- CI blocks regressions in runtime, docs, and lint hygiene.

---

### 4.4 Phase 4 - Downloader Hardening

**Objective:** Route BLS flat-file downloads through the same retry/header discipline already used elsewhere in the repo.

#### 4.4.1 Design

The shared client in `nfp_download.client` is the source of truth for:

- browser-like headers
- timeout policy
- exponential backoff on 429 and transient 5xx responses
- BLS API key query handling

The BLS flat-file client in `nfp_download.bls._http` should reuse that behavior instead of maintaining a separate weaker implementation.

#### 4.4.2 Changes

1. Refactor `packages/nfp-download/src/nfp_download/bls/_http.py` to use the shared client helpers.
   - use `create_client()` instead of constructing bare `httpx.Client`
   - use `get_with_retry()` for flat-file and CSV fetches where appropriate

2. Preserve the existing on-disk cache behavior.

3. Add a controlled fallback path for CES flat-file failures where practical.
   - for CE national data, fall back to the JSON API helper if flat-file access fails
   - for SM state data, either implement an equivalent API fallback or raise/log a clearer error

4. Normalize logging so retry/fallback behavior is visible but not noisy.

#### 4.4.3 Tests

- unit tests for flat-file retry behavior using mocked 429/5xx responses
- regression test that flat-file failures trigger the configured fallback path
- keep live BLS tests behind the `network` marker

#### 4.4.4 Acceptance Criteria

- BLS downloader code no longer constructs its own bare HTTP client for flat-file fetches.
- Retry/header behavior is centralized.
- CE national download has a tested fallback path when flat-file fetches fail.

---

### 4.5 Phase 5 - Stale Code and Lint Hygiene

**Objective:** Remove active-code ambiguity and make linting useful as a regression gate.

#### 4.5.1 Changes

1. Remove dead or misleading code in active modules.
   - delete the `sys.path` hack in `alt_nfp_estimation_v3.py`
   - remove commented-out SAE branches from active ingest modules where they are not scheduled for near-term reactivation
   - if SAE notes must remain, move them into a short tracked issue note or dedicated spec section rather than inline commented code

2. Narrow lint scope to maintained code and generated exclusions.
   - exclude `archive/`, `output/`, `site/`, and any generated docs artifacts from Ruff
   - migrate deprecated Ruff top-level settings to `[tool.ruff.lint]`

3. Fix active-code Ruff issues in maintained files.
   - prioritize `packages/*/src`
   - then tests
   - leave historical archive scripts out of scope for the enforcement pass

#### 4.5.2 Tests

- `uv run ruff check .` passes on the chosen maintained scope
- no maintained files contain large commented-out alternative implementations

#### 4.5.3 Acceptance Criteria

- Lint output is actionable rather than dominated by archived code.
- The active codebase has a clear source of truth.

---

## 5. File Targets

### 5.1 Runtime Correctness

- `alt_nfp_estimation_v3.py`
- `packages/nfp-ingest/src/nfp_ingest/panel.py`
- `packages/nfp-model-hmc/src/nfp_models/backtest.py`
- `packages/nfp-model-hmc/src/nfp_models/benchmark_backtest.py`
- `packages/nfp-model-hmc/src/nfp_models/sensitivity.py`
- `scripts/jan_reforecast.py`

### 5.2 Documentation

- `README.md`
- `CLAUDE.md`
- `docs/gen_ref_pages.py`
- `mkdocs.yml`
- `docs/getting-started/*.md`
- `docs/user-guide/*.md`
- `docs/architecture/*.md`
- `docs/reference/index.md`

### 5.3 Tests and CI

- `pyproject.toml`
- `tests/ingest/bls/test_downloads.py`
- `tests/test_fred.py`
- `.github/workflows/ci.yml` or equivalent workflow file

### 5.4 Downloader

- `packages/nfp-download/src/nfp_download/bls/_http.py`
- `packages/nfp-download/src/nfp_download/client.py`
- `packages/nfp-download/src/nfp_download/bls/ces_national.py`
- `packages/nfp-download/src/nfp_download/bls/ces_state.py`
- downloader unit tests

### 5.5 Cleanup

- `packages/nfp-ingest/src/nfp_ingest/vintage_store.py`
- `packages/nfp-ingest/src/nfp_ingest/release_dates/vintage_dates.py`
- `tests/test_release_dates.py`
- `pyproject.toml`

---

## 6. Validation Matrix

After all phases are complete, the following commands should pass:

```bash
# Default validation lane
uv run pytest
uv run ruff check .
uv run --group docs mkdocs build --strict

# Optional live-data lane
uv run pytest -m network
```

If the network lane requires credentials, its tests must self-skip with clear messages when the required environment variables are not present.

---

## 7. Rollout Order

Recommended merge order:

1. Phase 1 - runtime correctness
2. Phase 3 - test defaults and CI guardrails
3. Phase 2 - docs and API alignment
4. Phase 4 - downloader hardening
5. Phase 5 - stale code and lint cleanup

Phase 1 should land first because it changes the actual model inputs. Phase 3 should land immediately after so the corrected behavior is protected by default validation. Phase 2 can then rewrite docs against the corrected runtime behavior.

---

## 8. Definition of Done

This spec is complete when all of the following are true:

1. Main application paths include configured provider rows by default through config-aware call sites.
2. No public docs instruct users to import a nonexistent `alt_nfp` package.
3. The docs site builds in strict mode with zero warnings.
4. Default `pytest` execution succeeds without network access or secrets.
5. BLS flat-file downloads use the shared retry/header infrastructure.
6. Active-code linting is enforceable and no longer dominated by stale paths or archived files.

