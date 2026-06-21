# alt-nfp — Architecture Sketch

## Purpose

Bayesian state-space **nowcasting of US nonfarm payrolls (NFP)** from real-time data vintages. It fuses CES survey prints, QCEW administrative anchors, private payroll-provider microdata, and cyclical indicators (jobless claims, JOLTS openings) under strict **as-of censoring**, so every backtest sees only what was knowable on a given day. This is the **v2 repo**: the data layer is ported from a frozen reference (`~/Projects/alt_nfp`), and the model layer is rewritten in JAX, gated against that reference for **port-fidelity**. The reference is a work-in-progress, not validated truth — **parity is a fidelity floor, not a correctness certificate**; correctness is validated against external ground truth (published BLS / ALFRED real-time vintages), see `specs/plans/0`. Phases A0–A4 passed; A5 (real competitors: consensus + naive floors; ADP dropped) is in progress.

## Languages & frameworks

- **Python 3.12** throughout (`requires-python >=3.12`).
- **Data layer:** Polars (DataFrames/lazy I/O), NumPy, `universal-pathlib` + `s3fs` (local/S3 transparency), `curl-cffi` + `httpx[http2]` + BeautifulSoup/lxml (BLS/FRED fetching & scraping), Typer (CLI).
- **Model layer:** JAX + NumPyro (NUTS; `vmap` batching). Importing `nfp_model` globally enables float64 (parity is defined in double precision). (An early plan floated dynamax/Kalman marginalization; it was never pursued and isn't in the path — the AR(1) latent is a hand-rolled `jax.lax.scan`.)
- **Tooling:** `uv` (workspace + lockfile), pytest (+cov), ruff, black, mypy (soft), mkdocs-material (docs group).

## Repository = a `uv` workspace of 5 packages

A **virtual workspace root** (`alt-nfp`, not itself installable) with `members = ["packages/*"]`. The four data packages form a **strict linear dependency chain**; the model package sits apart and imports **no `nfp_*` package**.

```
nfp-lookups  →  nfp-download  →  nfp-ingest  →  nfp-vintages
                                     ⇣ (arrays/snapshots only — NO import)
                                 nfp-model   (jax / numpyro / numpy only)
```

| Package | Role | Key modules |
|---|---|---|
| **nfp-lookups** | Foundation; imports no other `nfp_*`. Schemas, industry/geography hierarchies, revision schedules, series-ID grammar, **canonical paths** (`NFP_BASE_DIR`, `NFP_STORE_URI`). | `paths`, `schemas`, `revision_schedules`, `provider_config`, `benchmark_revisions`, `series_ids` |
| **nfp-download** | HTTP fetching/scraping only — no transformation. | `bls/` (`bulk`, `ces_national`, `ces_state`, `qcew`, `_http`), `fred`, `release_dates/` (`scraper`, `parser`, `config`) |
| **nfp-ingest** | Vintage store, as-of censoring, panel/growth construction, provider compositing, indicators, and the **knowability + snapshot boundary**. | `vintage_store`, `panel`, `model_data` (`build_model_data(as_of=D)`), `snapshots`, `compositing`, `indicators`, `releases`, `tagger`, `ces_national`/`qcew`/`aggregate` |
| **nfp-vintages** | Historical vintage reconstruction pipeline + the `alt-nfp` CLI (top of the chain). | `__main__` (Typer app), `build_store`, `processing/` (`ces_triangular`, `qcew_bulk`, `combine`, `sae_states`), `views`, `evaluation` |
| **nfp-model** | JAX/NumPyro inference: ModelData arrays in → posterior out. Never sees a `vintage_date`. | `model` (the NumPyro model), `sampling` (`fit_model`), `batch` (`fit_model_batch`, vmap), `nowcast`, `parity`, `data` (snapshot/dict intake), `config` (`PRESETS`, `ModelPriors`) |

## Entry points

- **CLI: `alt-nfp`** → `nfp_vintages.__main__:app` (Typer). Idempotent pipeline subcommands, run in dependency order:
  `download` → `download-indicators` → `process` → `current` → `build` → `snapshot`.
  (Bare `alt-nfp` runs download → download-indicators → process → current.)
- **Library API: `nfp_model`** — `fit_model(data)`, `fit_model_batch(batched)`, `from_snapshot`, `model_inputs`, `nowcast_summary`, plus `PRESETS`/`ModelPriors`/`SamplerSettings`. This is how downstream code runs inference.
- **Harness scripts** (`scripts/`, not packaged): `generate_golden_masters.py` / `generate_a2_golden.py` / `generate_a3_reference.py` (build frozen fixtures from the reference repo), `run_a3_parity.py`, `run_a4_backtest.py` (the `snapshot`/`serial`/`batched`/`compare` backtest), `mirror_store.py` (push a local store into the bucket).

## Build / test / deploy

- **Install:** `uv sync` (workspace + `dev` group).
- **Test:** `uv run pytest` — `testpaths = packages`; markers `network` (excluded in CI) and `slow` (MCMC smoke). Coverage across all five `src/` trees by default; suites use `--no-cov` for speed. Store-dependent tests **self-skip** when the vintage store is unavailable.
- **Lint/format:** `uv run ruff check .` (line 100; `E,W,F,I,B,C4,UP`; excludes `docs`), black (100), mypy (soft — research code leans on Polars expression dynamism).
- **CI** (`.github/workflows/ci.yml`, push/PR to `main`): `uv sync` → `ruff check .` → `pytest -m "not network" --no-cov` on `ubuntu-latest`.
- **"Deploy":** none in the production sense — this is a research/inference repo. Operational surface is the `alt-nfp` CLI (pipeline maintenance) and the `nfp_model` library (fits/backtests). GPU is the intended A4 speed lever; the same batched code runs unmodified there.
- **Config & data:** `.env` (gitignored) loaded by the root `conftest.py` and the CLI; `NFP_STORE_URI` (e.g. `s3://alt-nfp/store`) + `AWS_*` select MinIO/S3, unset ⇒ local `data/store/` fallback (CI mode). **All filesystem layout comes from `nfp_lookups.paths`** (override root with `NFP_BASE_DIR`). The canonical store holds the **rebuilt** schema (reconstructable public CES/QCEW, 2017+; promoted from `…/store-rebuild` on 2026-06-18) — it is **replaceable**, not append-only/irreplaceable. Still: never `alt-nfp build` straight to `…/store` — rebuild to a scratch prefix and promote deliberately (see root `CLAUDE.md`).

## How the system fits together (data & control flow)

The design enforces **three physically separated concerns**, with a serialized artifact — not a function call — as the boundary between knowability and inference:

```
        ┌─────────────── ACQUISITION (network, credentials; run rarely) ───────────────┐
        │  BLS flat files / JSON API, FRED, release-calendar scrape, provider files     │
        │                              nfp-download                                     │
        └───────────────────────────────────┬──────────────────────────────────────────┘
                                             │ raw files → data/downloads/  (append-only archive)
        ┌────────────────────────────────────▼─────────────────────────────────────────┐
        │ KNOWABILITY  (pure, deterministic, no network, no model — ruthlessly tested)   │
        │  nfp-vintages.processing ──► revisions.parquet  +  releases.parquet            │
        │            (ces_triangular, qcew_bulk, combine)        (releases.build_releases)│
        │                                   │                                            │
        │            build_store  ─────────►│  VINTAGE STORE  (Hive parquet, S3/MinIO)   │
        │                                   │  partitioned by (source, seasonally_adj);  │
        │                                   │  holds LEVELS only + vintage/revision tags │
        │                                   ▼                                            │
        │  nfp-ingest.vintage_store.transform_to_panel(as_of_ref=D)                      │
        │   • two-layer as-of censoring  (vintage_date≤D  +  ref_date<D, then rank-based │
        │     CES/QCEW selection)        • log-growth computed per revision-cohort        │
        │                                   ▼                                            │
        │  build_panel  +  providers (compositing)  +  indicators                        │
        │                                   ▼                                            │
        │  build_model_data(as_of=D)  ──►  snapshot_model_data  ──►  *.npz + JSON meta    │
        │     (one dict answering "what was knowable on D")        (content-hash pinned)  │
        └────────────────────────────────────┬─────────────────────────────────────────┘
                                              │  ModelData arrays  (NO vintage_date crosses here)
        ┌─────────────────────────────────────▼────────────────────────────────────────┐
        │ INFERENCE  (JAX-land; imports only jax/numpyro/numpy — test-enforced)          │
        │  nfp_model.data.from_snapshot/model_inputs  →  strips to pure arrays           │
        │  nfp_model.model.nfp_model  (NumPyro state-space):                             │
        │     latent AR(1) continuing-units (era-specific means) + Fourier seasonal      │
        │     + structural birth/death + QCEW Student-t anchor + CES vintage-indexed     │
        │     + per-provider iid/AR(1) likelihoods                                       │
        │  fit_model (NUTS)   │   fit_model_batch (vmap over as-of grid, padded+masked)  │
        │                                   ▼                                            │
        │  posterior  →  nowcast_summary  /  parity.compare_reduced  (backtest reports)  │
        └────────────────────────────────────────────────────────────────────────────────┘
```

**Control flow.** The `alt-nfp` CLI orchestrates the left/middle column as idempotent, rarely-run steps (acquisition + store maintenance + snapshot baking). The model layer is a pure library invoked by the `scripts/` harnesses (`run_a4_backtest.py`: `snapshot → serial → batched → compare`) or any caller holding a snapshot. Because the boundary is a hash-pinned `.npz`, the GPU/backtest loop never touches the network, every run pins to a snapshot hash, and failures localize cleanly to one side of the seam.

## Design invariants that make sense of the code

- **The knowability/inference boundary is an artifact, not an import.** `nfp-model` consuming finished arrays (its only deps are jax/numpyro/numpy; intake goes through `nfp_model.data`) is what lets the model be developed offline against fixtures and run identically on CPU/GPU.
- **Two-layer as-of censoring is mandatory** (settled empirically): combined `vintage_date≤D` + `ref_date<D` filtering, then rank-based revision selection (`_select_ces_at_horizon`, `_select_qcew_at_horizon`). Vintage-date-only or ref-date-only filtering each fail in known ways.
- **The store holds levels; growth is derived at read time** per `(source, geo, industry, revision, benchmark_revision)` cohort — a convention with evaluation consequences documented in `specs/ces_growth_convention.md` (the open A5 scoring question).
- **Parity is the Phase-A *port-fidelity* floor — not correctness.** Each A-gate is a statistical match against the frozen reference (`~/Projects/alt_nfp`), enforced by golden-master fixtures (`s3://alt-nfp/golden/a{1,2,3}/`) and `nfp_model.parity`. It proved the JAX rewrite reproduced the reference; it does **not** certify the reference is right (it's a buggy WIP). Correctness is validated against external ground truth (published BLS / ALFRED), see `specs/plans/0`. The reference stays frozen as the port target + fixture generator — not as an assumed fallback.
- **Design record is in-repo:** `specs/plans/` (roadmap + gate logs), `specs/` (active design), `specs/completed/` (implemented/superseded specs + pre-port todos; reference articles in `references/`), and a per-package `CLAUDE.md` map.

## Two load-bearing choices

- The repo's spine is a **deliberately one-directional import graph** enforced by the `uv` workspace `[tool.uv.sources]` and an in-test assertion that `nfp-model` imports no `nfp_*`. That single rule is what makes "the model never sees a `vintage_date`" a structural guarantee rather than a convention.
- The **content-hashed `.npz` snapshot** is the load-bearing architectural choice: it turns a fuzzy "what was knowable on D" question into a frozen, addressable artifact, which is simultaneously the censoring contract, the GPU-batching enabler, and the parity-gate anchor.
