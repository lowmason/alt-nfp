# alt-nfp

Bayesian state-space NFP nowcasting from real-time data vintages. This is the
v2 repo: data packages ported from `~/Projects/alt_nfp` (underscore); the model
layer is rewritten in JAX/NumPyro as `nfp-model`. The port was gated against the
old repo for **fidelity** (did the rewrite reproduce it?) — but that repo is a
work-in-progress with bugs, **not validated truth**. Parity is a port-fidelity
floor, not a correctness certificate; correctness is validated against **external
ground truth** (published BLS / ALFRED real-time vintages) — see `specs/plans/0`.
Roadmap: `specs/plans/`; design record: `specs/` (active) and `specs/completed/`
(implemented/superseded).

## Workspace

uv workspace. The four data packages form a **linear dependency chain** —
enforce it; `nfp-model` sits apart and imports **no** `nfp_*` package:

```
nfp-lookups → nfp-download → nfp-ingest → nfp-vintages
                                  ⇣ (arrays/snapshots only, no import)
                              nfp-model
```

| Package | Role |
|---|---|
| `nfp-lookups` | Foundation: schemas, hierarchies, revision schedules, series-ID grammar, canonical paths. Imports no other `nfp_*` package — ever. |
| `nfp-download` | HTTP clients/scrapers for BLS + FRED. Fetching only, no transformation. |
| `nfp-ingest` | Vintage store API, as-of censoring, panel construction, provider ingestion, compositing, ModelData + snapshots. |
| `nfp-vintages` | Historical vintage reconstruction pipeline + `alt-nfp` CLI (top of the chain). |
| `nfp-model` | JAX/NumPyro inference: ModelData arrays in, posterior out. Imports only jax/numpyro/numpy; importing it enables global float64. |

Each package has its own `CLAUDE.md` with the internal map.

## Commands

```bash
uv sync                                   # install workspace + dev group
uv run pytest -m "not network" --no-cov   # full local suite (~3 min; MCMC smoke included)
uv run pytest -m "not network and not slow" --no-cov  # fast suite (~30s), skips MCMC smoke
uv run ruff check .                       # lint (line 100; E,W,F,I,B,C4,UP)
uv run alt-nfp --help                     # production vintage-store CLI
uv run alt-nfp update --as-of 2026-01-12  # capture knowable month-T prints, append to store
uv run alt-nfp status                     # store coverage + uncaptured/corrected alarm
uv run alt-nfp watch --source all         # BLS-feed-driven trigger (cron)
uv run python scripts/bootstrap_store.py \  # one-time historical rebuild + promote (NOT a command)
    --scratch s3://alt-nfp/store-rebuild --canonical s3://alt-nfp/store
```

CI (`.github/workflows/ci.yml`) runs ruff + the non-network suite on push/PR
to `main`.

## Hard rules

- **Paths**: all filesystem layout comes from `nfp_lookups.paths` (override
  root with `NFP_BASE_DIR`). Never construct data paths in other packages.
- **The vintage store lives in MinIO/S3**, not on disk: `NFP_STORE_URI`
  (e.g. `s3://alt-nfp/store`) + `AWS_*` env vars in the gitignored `.env`
  (loaded by the root `conftest.py` for tests and by the CLI). Unset ⇒ local
  `data/store/` fallback (what CI uses). Store code accepts both `Path` and
  `UPath`; Polars I/O gets options via `nfp_lookups.paths.storage_options_for`.
  Re-sync from a local store with `uv run python scripts/mirror_store.py`
  (targets `NFP_STORE_URI`; it refuses the canonical `…/store` unless you pass
  `--allow-canonical`).
- **Container storage contract (Bloomberg) — no code writes under `./data`.** The
  compute container has a tiny disk footprint, so every persistent artifact goes to S3
  via an env URI (each unset ⇒ local `data/` fallback for dev/CI). Besides
  `NFP_STORE_URI`: `NFP_SNAPSHOTS_URI` (ModelData snapshots), `NFP_DATA_URI` (indicators,
  `competitors/consensus`, the vintage/release-date schedules), and `NFP_PROVIDERS_URI`
  (the **separate** provider store — not under `NFP_DATA_URI`, not seeded by this repo).
  Store/data/providers locations resolve in `nfp_lookups.paths` (`_store_location`,
  `data_location`, `providers_location`); `NFP_SNAPSHOTS_URI` resolves in
  `nfp_ingest.snapshots.snapshots_location`. All thread `storage_options_for` + an
  `is_remote` mkdir guard. Rebuild scratch (raw downloads, HTTP cache, SAE checkpoint)
  goes to `tempfile` automatically; dev scripts must take their output root as an
  arg/env (a `/tmp` path or `s3://` URI), never `data/`. See
  `specs/plans/completed/15-container_safe_storage.md`.
- **Boundaries**: no upward imports (e.g. lookups must not import download);
  no cross-package imports of underscore-private names.
- **`data/` is proprietary and gitignored** — this repo is public. Tests that
  need the vintage store self-skip when it's unavailable; network tests are
  marked `@pytest.mark.network`.
- **Rebuild to scratch; promote deliberately.** `s3://alt-nfp/store` now holds
  the **rebuilt** schema (reconstructable public CES/QCEW, 2017+; promoted from
  `…/store-rebuild` on 2026-06-18 via `plans/10` T8, prior canonical preserved at
  `…/store-prev-20260618`). It is **not** append-only/irreplaceable — the old
  "live-captured, exists in no raw input" framing is retired (see memory
  `store-replaceable-and-rebuild-backlog`). Still: the everyday CLI has **no**
  `build` command — the one-time rebuild is `scripts/bootstrap_store.py` (never
  write straight to `…/store`). Rebuilds target a scratch prefix
  (`NFP_STORE_URI=s3://alt-nfp/store-rebuild …`); promotion to canonical is the
  deliberate copy-then-delete cutover the bootstrap generalizes from `plans/10`
  T8 — snapshot the prior canonical first, then copy-then-delete per partition
  (filenames encode vintage ranges, so a plain overwrite would leave both files
  and corrupt the store). `is_canonical_store` still guards
  `write_rebuild_store`/`build_store`/`mirror_store` (and `bootstrap_store.py`
  refuses a canonical `--scratch`) against accidental clobber.
- **Specs workflow**: implemented specs move from `specs/` to `specs/completed/`;
  implemented plans move from `specs/plans/` to `specs/plans/completed/`. Superseded
  pre-port material (old todos) lives in `specs/completed/todos/`.
- The old repo at `~/Projects/alt_nfp` is the frozen **port reference** — read it
  to see what the JAX rewrite was ported from, never modify it. But it is a buggy
  WIP, **not an oracle**: a reference value is not automatically correct. When code
  must be *correct* (not merely port-faithful), validate against external ground
  truth (published BLS / ALFRED), and re-baseline the golden if a correctness fix
  diverges from the reference. Parity ≠ correctness.
