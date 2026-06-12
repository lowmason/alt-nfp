# alt-nfp

Bayesian state-space NFP nowcasting from real-time data vintages. This is the
v2 repo: data packages ported from the frozen reference implementation at
`~/Projects/alt_nfp` (underscore); the model layer will be rewritten in JAX as
`nfp-model-jax` behind parity gates. Roadmap: `plans/`; design record:
`specs/` (active) and `archive/` (implemented/superseded).

## Workspace

uv workspace, **linear dependency chain** — enforce it:

```
nfp-lookups → nfp-download → nfp-ingest → nfp-vintages
```

| Package | Role |
|---|---|
| `nfp-lookups` | Foundation: schemas, hierarchies, revision schedules, series-ID grammar, canonical paths. Imports no other `nfp_*` package — ever. |
| `nfp-download` | HTTP clients/scrapers for BLS + FRED. Fetching only, no transformation. |
| `nfp-ingest` | Vintage store API, as-of censoring, panel construction, provider ingestion, compositing. |
| `nfp-vintages` | Historical vintage reconstruction pipeline + `alt-nfp` CLI (top of the chain). |

Each package has its own `CLAUDE.md` with the internal map.

## Commands

```bash
uv sync                                   # install workspace + dev group
uv run pytest -m "not network" --no-cov   # fast suite (~2s); coverage runs by default otherwise
uv run ruff check .                       # lint (line 100; E,W,F,I,B,C4,UP)
uv run alt-nfp --help                     # vintage pipeline CLI
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
  Re-sync from a local store with `uv run python scripts/mirror_store.py`.
- **Boundaries**: no upward imports (e.g. lookups must not import download);
  no cross-package imports of underscore-private names.
- **`data/` is proprietary and gitignored** — this repo is public. Tests that
  need the vintage store self-skip when it's unavailable; network tests are
  marked `@pytest.mark.network`.
- **Never rebuild the canonical store in place.** `s3://alt-nfp/store`
  contains live-captured release-day vintage rows (national CES, Mar 2025–
  Jan 2026 and ongoing) that exist in no raw input — a from-scratch
  `alt-nfp build` to that URI would silently destroy them. Rebuilds target a
  scratch prefix (`NFP_STORE_URI=s3://alt-nfp/store-rebuild …`); the
  canonical store only ever takes appends.
- **Specs workflow**: implemented specs move from `specs/` to `archive/`.
- The old repo at `~/Projects/alt_nfp` is the frozen reference — read it for
  parity questions, never modify it.
