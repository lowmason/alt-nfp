# Boundaries & Paths

## Import boundaries

Two rules govern every import in the workspace:

**No upward imports.** A package may only import from packages below it in the
dependency chain. `nfp-lookups` imports nothing internal. `nfp-download` imports
only `nfp-lookups`. `nfp-ingest` imports only `nfp-lookups` and `nfp-download`.
`nfp-vintages` imports the three below it. `nfp-model` imports **no** `nfp_*`
package at all (only `jax`, `numpyro`, `numpy`).

Concretely, this means:

- `nfp-lookups` must not import `nfp-download` (or any package above it)
- `nfp-download` must not import `nfp-ingest` or `nfp-vintages`
- `nfp-ingest` must not import `nfp-vintages`
- `nfp-model` must not import any `nfp_*` package

The `nfp-model` boundary is test-enforced in
`packages/nfp-model/src/nfp_model/tests/test_model_unit.py::TestBoundary`.

**No cross-package imports of underscore-private names.** Symbols prefixed with
`_` are package-internal. Other packages must not import them, even if Python
allows it at runtime.

## Filesystem paths

**All filesystem layout comes from `nfp_lookups.paths`.** Every path constant
used anywhere in the workspace is defined there and must be imported from there.
No other package may construct data directory paths on its own.

The root of the layout is controlled by the `NFP_BASE_DIR` environment variable.
Discovery order at import time:

1. `NFP_BASE_DIR` env var (set before first import)
2. Walk up from the calling file to find a directory containing `packages/` + `pyproject.toml`
3. Fixed-depth fallback for editable installs

Setting `NFP_BASE_DIR` is the correct way to relocate the entire data tree (for
example, to a scratch volume on a build machine).

### Key path constants

| Symbol | Default location |
|---|---|
| `BASE_DIR` | Repo root (or `NFP_BASE_DIR`) |
| `DATA_DIR` | `BASE_DIR/data/` |
| `STORE_DIR` | `DATA_DIR/store/` |
| `DOWNLOADS_DIR` | `DATA_DIR/downloads/` (rebuild scratch) |
| `INTERMEDIATE_DIR` | `DATA_DIR/intermediate/` (rebuild scratch) |
| `RELEASE_DATES_PATH` | `DATA_DIR/intermediate/release_dates.parquet` |
| `VINTAGE_DATES_PATH` | `DATA_DIR/intermediate/vintage_dates.parquet` |

The actual resolved values depend on env vars — see [Storage contract](storage-contract.md)
for how `NFP_STORE_URI`, `NFP_DATA_URI`, and `NFP_PROVIDERS_URI` redirect the
persistent artifacts to S3/MinIO.

### Helper functions

`nfp_lookups.paths` also exposes:

- `storage_options_for(path)` — builds the Polars/object_store credential dict
  for any S3 path; pass this wherever Polars reads or writes a remote parquet
- `is_remote(path)` — returns `True` for `UPath` objects backed by S3; used to
  guard `mkdir()` calls that would fail on an S3 prefix
- `upath_for(uri)` — the shared credentialed-`UPath` builder behind every
  `*_location()` helper
- `data_location()`, `providers_location()` — resolve `NFP_DATA_URI` /
  `NFP_PROVIDERS_URI` to a `UPath` (S3 or local fallback)
