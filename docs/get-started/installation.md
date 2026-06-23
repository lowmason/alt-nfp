# Installation

## Prerequisites

- [uv](https://docs.astral.sh/uv/) 0.4 or later
- Python 3.11+
- Git

## Install the workspace

Clone the repo and sync all packages and the dev dependency group:

```bash
git clone https://github.com/lowmason/alt-nfp.git
cd alt-nfp
uv sync
```

To also install the documentation dependencies (MkDocs + plugins):

```bash
uv sync --group docs
```

## The five-package workspace

alt-nfp is a [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/) containing five packages arranged in a **linear dependency chain**. The chain is enforced — no package may import from a package above it:

```
nfp-lookups → nfp-download → nfp-ingest → nfp-vintages
                                  ↓ (arrays/snapshots only, no import)
                              nfp-model
```

| Package | Role |
|---|---|
| `nfp-lookups` | Foundation: schemas, hierarchies, revision schedules, series-ID grammar, canonical paths. Imports no other `nfp_*` package — ever. |
| `nfp-download` | HTTP clients/scrapers for BLS + FRED. Fetching only, no transformation. |
| `nfp-ingest` | Vintage store API, as-of censoring, panel construction, provider ingestion, compositing, ModelData + snapshots. |
| `nfp-vintages` | Historical vintage reconstruction pipeline + the `alt-nfp` CLI (top of the chain). |
| `nfp-model` | JAX/NumPyro inference: ModelData arrays in, posterior out. Imports only `jax`/`numpyro`/`numpy`; importing it enables global float64. |

## Environment and the vintage store

The vintage store lives in **MinIO/S3, not on disk.** Four environment variables select where each artifact class lives; when any variable is unset the code falls back to a local `data/` subdirectory (what CI uses — never set on Bloomberg):

| Variable | What it points to | Local fallback |
|---|---|---|
| `NFP_STORE_URI` | Hive-partitioned CES/QCEW vintage store (e.g. `s3://alt-nfp/store`) | `data/store/` |
| `NFP_SNAPSHOTS_URI` | Hash-pinned ModelData `.npz` snapshots (e.g. `s3://alt-nfp/snapshots`) | `data/snapshots/` |
| `NFP_DATA_URI` | Indicators, `competitors/consensus.parquet`, vintage/release-date schedules (e.g. `s3://alt-nfp`) | `data/` |
| `NFP_PROVIDERS_URI` | Provider parquets — a **separate** store not seeded by this repo | `data/providers/` |

Object-storage credentials go in the same file:

| Variable | Purpose |
|---|---|
| `AWS_ACCESS_KEY_ID` | S3 / MinIO access key |
| `AWS_SECRET_ACCESS_KEY` | S3 / MinIO secret key |
| `AWS_ENDPOINT_URL` | Override endpoint (e.g. `http://127.0.0.1:9000` for local MinIO; leave unset for AWS S3) |

Create a `.env` file (gitignored) at the repo root with the variables above and fill in your values:

```bash
# .env (repo root, gitignored) — the *_URI roots + AWS_* credentials from the tables above, e.g.:
NFP_STORE_URI=s3://alt-nfp/store
NFP_DATA_URI=s3://alt-nfp
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
# AWS_ENDPOINT_URL=http://127.0.0.1:9000   # local MinIO only; leave unset for AWS S3
```

The root `conftest.py` loads `.env` automatically for `pytest`. The `alt-nfp` CLI loads it on startup.

## The `.env` gotcha

Any **ad-hoc Python process** that doesn't call `load_dotenv()` before importing
`nfp_lookups.paths` will read `NFP_STORE_URI` as unset and silently operate on the
**empty local `data/store/`** — not the real S3 store. The `pytest` suite and the `alt-nfp`
CLI both load `.env` on startup, but a bare `python -c "..."` or a script invoked directly
does not. Always confirm which store a script is pointed at with `uv run alt-nfp status`
before running anything consequential.

## Verify the install

```bash
uv run pytest -m "not network and not slow" --no-cov   # fast suite (~30 s)
uv run alt-nfp --help
```

The fast suite skips MCMC smoke tests and any test that requires live network access or the
vintage store. Tests that need the store self-skip when it is unavailable.
