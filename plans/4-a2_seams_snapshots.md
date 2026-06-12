# Implementation Plan: A2 — Seam Fixes + ModelData Snapshot

> **Status: ✅ COMPLETE (2026-06-12).** All four workstreams done; suite at
> **389 passed / 1 intentional skip** (+9 A2 golden parity, +8 snapshot
> tests), ruff clean, A1 masters still green throughout.
>
> - **4.1** done: `download_ces`/`download_qcew`/`download_qcew_bulk` moved
>   to `nfp_download/bls/bulk.py` (single module — no name collisions);
>   network test moved to `nfp-download/tests/bls/test_bulk_network.py`;
>   `nfp_vintages/download/` deleted; vintages dropped its now-unused
>   `curl-cffi` dependency.
> - **4.2** done: `processing/ces_national.py` → `ces_triangular.py`,
>   `processing/qcew.py` → `qcew_bulk.py`.
> - **4.3** done: `nfp_ingest/model_data.py` with `ModelDataConfig`
>   (defaults verified live against the old settings), faithful
>   `panel_to_model_data` port (PP_COLORS dropped), `build_model_data`
>   entry point; `PROVIDERS_DEFAULT` added to nfp-lookups; indicator
>   parquets copied. **A2 golden parity: 9/9 as-of dates — every array,
>   frame, and map identical** to the old repo's two-layer outputs.
> - **4.4** done: `nfp_ingest/snapshots.py` (collect/save/load/content_hash,
>   `snapshot_model_data`), `alt-nfp snapshot --as-of [--grid-end]` CLI;
>   hash-stability proven (build-twice → identical hash); CLI smoke wrote
>   `s3://alt-nfp/snapshots/asof=2026-01-12/model_data_7791422c4a8b.npz`.
> - Gates: A1+A2 masters pass; `model_data.py`/`snapshots.py` import no
>   acquisition code (grep-verified); store-gated tests skip cleanly in CI
>   mode (verified with `.env` removed).
>
> **Major finding — latent regression in the frozen reference:** since its
> settings refactor, `panel_adapter` resolves `indicators_dir` relative to
> the model *package* directory (`resolve_paths(parents[2])`), which has no
> `data/` — so **every default-config run of the old repo silently dropped
> the cyclical indicators** (claims_c/jolts_c = None; `model.py` then builds
> no φ₃ block). The A2 masters pin the *intended* behavior by routing the
> old code's own relative-path mechanism back to the repo root
> (`PathsConfig(data_dir="../../data")`). **Consequence for A3:** the parity
> baseline must be re-run with this corrected config, or the reference
> posterior will lack φ₃ — and any historical old-repo backtest results that
> used default config understate the cyclical contribution.

Phase A2 of `plans/0-port_and_staged_plan.md`: consolidate the download
layers, move all knowability logic into the data side behind a single
`build_model_data(as_of=D)` entry point, and introduce the serialized,
hash-pinned snapshot artifact. The A1 golden masters are the regression net
for every step.

**Gates (from the plan of record):** A1 golden masters still pass; the model
layer consumes finished arrays and imports nothing from acquisition;
snapshots are hash-stable across regeneration.

## Workstreams

### 4.1 Download consolidation (deferred Tier-4 seam)

`nfp_vintages/download/{ces,qcew}.py` are acquisition and belong in
nfp-download:

1. Move both modules into a single `nfp_download/bls/bulk.py`
   (`download_ces`, `download_qcew`, `download_qcew_bulk` + URL constants
   and helpers — one module, so no name collision with anything).
2. Move `packages/nfp-vintages/tests/test_download_network.py` →
   `packages/nfp-download/tests/bls/test_bulk_network.py` (it tests
   transport + page structure, both download concerns).
3. Update the only consumer (`nfp_vintages/__main__.py`), fix the two
   docstring references in `processing/qcew.py`, delete
   `nfp_vintages/download/`, update both CLAUDE.mds.

**Gate:** suite + A1 masters green; `grep` shows no `nfp_vintages.download`.

### 4.2 Processing renames (deferred Tier-4 naming collisions)

- `nfp_vintages/processing/ces_national.py` → `ces_triangular.py`
  (no longer collides with `nfp_ingest/ces_national.py`)
- `nfp_vintages/processing/qcew.py` → `qcew_bulk.py`
  (no longer collides with `nfp_ingest/qcew.py`)
- Update `__main__.py` imports + vintages CLAUDE.md map.

### 4.3 Knowability → `nfp_ingest.model_data` (the heart)

Port the old repo's `nfp_models/panel_adapter.py` (600 lines, read in full)
into a new `packages/nfp-ingest/src/nfp_ingest/model_data.py`:

1. **`ModelDataConfig` dataclass** (data-layer home for the knowability +
   measurement-metadata knobs, frozen defaults extracted from the old
   settings system): `era_breaks=[2020-01-01]`, `bd_qcew_lag=6`,
   `provider_pub_lag_weeks=3`,
   `qcew_post_covid_boundary_mult={0: 5.0, 1: 3.5, 2: 2.0}` (default 1.0),
   `indicators=CYCLICAL_INDICATORS_DEFAULT` (claims lag-1, jolts lag-2 —
   already in nfp-lookups). The old pydantic/toml settings system stays a
   model-package concern for A3; the data layer gets plain defaults.
2. **`panel_to_model_data(panel, providers, as_of=…, config=…)`** — faithful
   port: vintage_date cutoff, national scope restriction, model calendar,
   QCEW series + revision-keyed noise multipliers + post-COVID boundary
   inflation, CES best-available selection with contiguous vintage-index
   remapping, provider growth/births arrays with the 3-week publication-lag
   censoring, BD covariates (birth_rate, bd_proxy, bd_qcew_lagged), cyclical
   indicator loading/centering with per-indicator publication-lag masking,
   and the reconstructed levels frame. **Dropped:** `PP_COLORS` (plotting
   concern — does not belong in the data layer) and the settings dependency.
3. **`build_model_data(as_of, …)`** — the one function that answers "what
   was knowable on D": layer-1 `build_panel(as_of_ref=as_of)` + layer-2
   `panel_to_model_data(as_of=as_of)`.
4. **`PROVIDERS_DEFAULT`** (the G config) added to
   `nfp_lookups.provider_config` — its fields are already public via the
   committed A1 manifest.
5. Copy the frozen FRED indicator parquets from the old repo into local
   `data/indicators/` (public FRED data, gitignored anyway).

**Gate: A2 golden fixtures.** Generated from the old repo (same read-only
old-venv pattern as A1) at the same 9 as-of dates: old `build_panel` → old
`panel_to_model_data(PROVIDERS, as_of=D, cfg defaults)`, arrays + metadata
serialized to `s3://alt-nfp/golden/a2/`, manifest committed. New
`build_model_data(as_of=D)` must reproduce **every array** (NaN-aware exact
equality), every index set, the vintage maps, and the calendar. The
2026-02-12 negative master stays layer-1 (unbuildable horizon — also
unbuildable through `build_model_data`, asserted).

### 4.4 The snapshot artifact

1. `save_model_data(data, path)` / `load_model_data(path)`: single `.npz`
   holding all arrays plus one JSON metadata entry (dates, maps, provider
   configs, config knobs, schema version). Panel/levels frames are **not**
   in the snapshot (the panel is already golden-mastered and reproducible
   from the store; the model consumes arrays).
2. `content_hash(data)`: sha256 over a canonical serialization — sorted
   (name, dtype, shape, raw bytes) of every array + the canonical metadata
   JSON. **Not** the file bytes: npz is a zip and zips embed timestamps.
3. CLI: `alt-nfp snapshot --as-of YYYY-MM-DD [--grid START END]` writing
   `snapshots/asof=<D>/model_data_<hash12>.npz` under `NFP_SNAPSHOTS_URI`
   (default: `s3://alt-nfp/snapshots` when the store is remote, else
   `data/snapshots/`).
4. Tests: save/load round-trip preserves every array and the hash;
   **hash-stability** (two independent `build_model_data` runs → identical
   hash); CLI smoke against MinIO.

## Out of scope (A3+)

- The JAX model itself; porting the pydantic settings system; the CLI-home
  question (bites when a model CLI exists); precomputing the full backtest
  snapshot grid (one `--grid` run is demonstrated, the full grid is an A4
  concern when the backtest loop lands).

## Sequencing

4.1 → 4.2 (mechanical, masters as net) → 4.3 port + A2 fixtures (the real
work) → 4.4 snapshots → gates annotated in plans/0.
