# nfp-vintages

Vintage data pipeline for CES, QCEW, and SAE employment data.

## Overview

Pipeline: process → build for managing real-time data vintages (acquisition
lives in `nfp_download.bls.bulk` since the A2 seam fix). Provides:
- **Processing** (`processing/`): transform raw downloads into revision-tagged parquet
- **Store builder** (`build_store.py`): merge revisions + current estimates into `data/store/`
- **Views** (`views.py`): `real_time_view()`, `final_view()`, `specific_vintage_view()`
- **Evaluation** (`evaluation.py`): `vintage_diff()`, noise multiplier construction
- **A5 / Tier 0–1 eval** (`a5.py`, `scoreboard.py`, `diagnostics.py`, `competitors/`):
  the **private** first-print scoreboard — month-type/calibration/CRPS metrics
  (`scoreboard.py`), Aruoba revision + Mincer–Zarnowitz diagnostics +
  `pooled_first_print_bias` (the §5A post-hoc offset δ) (`diagnostics.py`), and the
  competitor adapters (`competitors/naive.py` random-walk + trailing-mean floors;
  `competitors/consensus.py` `load_consensus`/`Consensus`). The §5A offset surfaces
  in `run_a5_backtest.py:cmd_score` as a `model_5a` competitor row (point + draws
  shifted by δ); `A5_NO_PROVIDERS=1` builds the public-only skeleton. Evaluation-side
  only — no `nfp-model` import.
- **Track B — Total assembly** (`assembly.py`, `wedge_diagnostics.py`):
  `assemble_total()` convolves the private nowcast posterior with the government
  **wedge** posterior into a Total-NFP posterior; `score_total()` scores it vs the
  Total `00` first print + consensus; `wedge_diagnostics.py` holds the wedge
  decomposition + RIF intervention-sd calibration. Spec: `specs/completed/government_wedge.md`.
- **CLI**: `alt-nfp` (or `python -m nfp_vintages`)

## Tech Stack

- **Language**: Python 3.12 (requires >= 3.12)
- **Dependencies**: httpx (SAE fetches), polars, typer (CLI)
- **Build**: hatchling
- **Internal deps**: `nfp-lookups` (industry, geography, revision schedules), `nfp-download` (HTTP client), `nfp-ingest` (vintage store read/write)

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

## Package Structure

```
src/nfp_vintages/
├── __init__.py             # Exports: real_time_view, final_view, vintage_diff
├── __main__.py             # CLI entry point (update/status/watch/snapshot; legacy build chain retired §10)
├── calendar.py             # advance_release_calendar() — release-calendar scrape (lifted from __main__)
├── store_status.py         # compute_status()/format_status() — read-only coverage report (status)
├── views.py                # real_time_view(), final_view(), specific_vintage_view()
├── evaluation.py           # vintage_diff(), build_noise_multiplier_vector()
├── build_store.py          # Merge revisions + current → Hive-partitioned store
│   # ── A5 / Tier 0–1 evaluation layer (private first-print scoreboard) ──
├── a5.py                   # A5 first-print index / target extraction (private '05')
├── scoreboard.py           # month-type classify, interval coverage, CRPS, change_draws_k, venue
├── diagnostics.py          # OLS, first→third revision table, Aruoba + Mincer–Zarnowitz
├── competitors/
│   ├── naive.py            # RandomWalk + TrailingMean floors
│   └── consensus.py        # load_consensus() + Consensus (Track B; None-tolerant → '—')
│   # ── Track B — Total assembly (private nowcast ⊕ government wedge) ──
├── assembly.py             # assemble_total() + score_total() (vs Total first print + consensus)
├── wedge_diagnostics.py    # wedge decomposition residual + RIF intervention-sd calibration
│   # (store-rebuild infra rebuild_store.py / rebuild_gates.py — see specs/plans/completed/10–12)
└── processing/
    ├── __init__.py
    ├── ces_triangular.py   # CES triangular-revision CSV processing
    ├── qcew_bulk.py        # QCEW vintage processing (4 input streams)
    ├── sae_states.py       # State and Area Employment processing (disabled)
    └── combine.py          # Combine vintage files
```

Downloads (`download_ces`, `download_qcew`, `download_qcew_bulk`) moved to
`nfp_download.bls.bulk` in A2 — see nfp-download's CLAUDE.md.

## Code Style

- **Formatter**: black (line length 100)
- **Linter**: ruff (line length 100, rules: E, W, F, I, B, C4, UP)
- Line length limit: 100 characters

## Key Patterns

- **Vintage store format**: Hive-partitioned parquet at `data/store/`, partitioned by `(source, seasonally_adjusted)`. Sources: `ces`, `qcew`, `sae`.
- **QCEW processing** (`processing/qcew_bulk.py`): four input streams: (1) total all-ownership, (2) private 2-digit NAICS, (3) government by ownership (→ sectors 91/92/93), (4) manufacturing 3-digit NAICS (→ durable 31 / nondurable 32). Employment units converted from persons to thousands.
- **CES processing** (`processing/ces_triangular.py`): parses triangular revision CSV structure from `cesvinall/`, assigns vintage dates from release schedule.
- **SAE processing** (`processing/sae_states.py`): State and Area Employment, fetched via httpx.
- **Views** (`views.py`): pure Polars operations on vintage DataFrames. `real_time_view()` returns what was known at a given date. `final_view()` returns latest available revision.
- **Evaluation** (`evaluation.py`): `vintage_diff()` computes revision magnitudes. `build_noise_multiplier_vector()` constructs empirical noise multipliers by source and revision. Uses `nfp_lookups.revision_schedules` for CES/QCEW revision specs.
- **CLI** (`__main__.py`): typer app with the production subcommands `update` (capture month-T prints → append to store), `status` (read-only coverage + uncaptured alarm), `watch` (BLS-feed-driven trigger for cron), and `snapshot` (hash-pinned ModelData, day-12). The legacy stage commands (`download`/`download-indicators`/`process`/`current`/`build`/`build-rebuild`) and the bare-run chain were retired in the production-workflow reshape (§10); the one-time historical rebuild is now the `scripts/bootstrap_store.py` script, not a CLI command.
- **Total assembly seam** (`assembly.py`, Track B): the private nowcast draws are in **growth/index space**, the wedge draws are native **change-k** — `assemble_total` converts the private leg via `scoreboard.change_draws_k` (using a **first-finite** `(base_index, idx_to_level)` anchor from `nfp_ingest.model_data.levels_provenance`, to avoid the `base_index` NaN class) and resamples it to the **wedge** draw count before the element-wise add. The two MCMC fits are independent (no shared seed); pairing is positional after resample. A residual-coupling knob exists but is **default off** (point-invariant). Consumed by `scripts/run_a5_backtest.py:cmd_total`.

## Data Layout

> **Container contract (plans/15):** on Bloomberg nothing writes under `./data`. The
> **store** is S3 (`NFP_STORE_URI`); `release_dates.parquet`/`vintage_dates.parquet` are
> persistent inputs on S3 (`NFP_DATA_URI`). Everything else below is **rebuild scratch** →
> `tempfile`, not `data/`: raw `downloads/` (re-fetched each rebuild), the
> `{ces,qcew,sae}_revisions`/`revisions` intermediates, scraped release HTML, and the SAE
> checkpoint. The local `data/` tree below is the dev/CI fallback layout only.

```
data/
├── store/                  # Vintage store → NFP_STORE_URI (S3) on Bloomberg
│   ├── source=ces/
│   ├── source=qcew/
│   └── source=sae/
├── downloads/              # Rebuild scratch (→ tempfile): raw fetched files
│   ├── ces/cesvinall/      # CES triangular revision CSVs
│   ├── qcew/               # QCEW bulk + revisions
│   └── releases/           # Scraped BLS schedule HTML (tempfile)
└── intermediate/           # Pipeline byproducts
    ├── ces_revisions.parquet   # rebuild scratch → tempfile
    ├── qcew_revisions.parquet  # rebuild scratch → tempfile
    ├── revisions.parquet       # rebuild scratch → tempfile
    ├── release_dates.parquet   # persistent input → NFP_DATA_URI (S3)
    └── vintage_dates.parquet   # persistent input → NFP_DATA_URI (S3)
```

## Test Mapping

Tests live in `src/nfp_vintages/tests/` within this package:
- `test_vintages.py` — vintage view & evaluation tests
- (download transport tests moved to `packages/nfp-download/src/nfp_download/tests/bls/test_bulk_network.py`)
