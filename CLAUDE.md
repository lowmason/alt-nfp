# alt-nfp

Bayesian state-space NFP nowcasting from real-time data vintages. This is the
v2 repo: data packages ported from the frozen reference implementation at
`~/Projects/alt_nfp` (underscore); the model layer is rewritten in
JAX/NumPyro as `nfp-model` behind parity gates. Roadmap: `plans/`; design
record: `specs/` (active) and `archive/` (implemented/superseded).

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
  Re-sync from a local store with `uv run python scripts/mirror_store.py`
  (targets `NFP_STORE_URI`; it refuses the canonical `…/store` unless you pass
  `--allow-canonical`).
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
  `store-replaceable-and-rebuild-backlog`). Still: never `alt-nfp build` straight
  to `…/store`. Rebuilds target a scratch prefix
  (`NFP_STORE_URI=s3://alt-nfp/store-rebuild …`); promotion to canonical is the
  deliberate T8 cutover — snapshot the prior canonical first, then copy-then-delete
  per partition (filenames encode vintage ranges, so a plain overwrite would leave
  both files and corrupt the store). `is_canonical_store` still guards
  `build_store`/`mirror_store` against accidental clobber.
- **Specs workflow**: implemented specs move from `specs/` to `archive/`.
- The old repo at `~/Projects/alt_nfp` is the frozen reference — read it for
  parity questions, never modify it.
