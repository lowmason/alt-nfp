# Storage Contract

## The container constraint

**No code writes under `./data` in production.** The compute environment (Bloomberg)
has a minimal local disk footprint. Every persistent artifact goes to S3 via an
environment URI. The `./data/` tree is a local **dev/CI fallback only** — when the
relevant env var is unset, the code falls back to a local path automatically.

This rule applies to all persistent artifacts. Temporary/rebuild scratch (raw
downloads, HTTP cache, intermediate parquet) goes to `tempfile` and is never
routed to S3 or `data/`.

## Environment URIs

Four env vars route persistent artifacts to S3 buckets. Each has a local fallback
for development and CI.

| Env var | What it roots | Local fallback |
|---|---|---|
| `NFP_STORE_URI` | The vintage store (Hive-partitioned parquet) | `data/store/` |
| `NFP_SNAPSHOTS_URI` | ModelData snapshots (hash-pinned `.npz` + JSON meta) | `data/snapshots/` |
| `NFP_DATA_URI` | Indicators, competitors/consensus, release/vintage-date schedules | `data/` |
| `NFP_PROVIDERS_URI` | The provider store (separate from `NFP_DATA_URI`) | `data/providers/` |

Three of the four (`NFP_STORE_URI`, `NFP_DATA_URI`, `NFP_PROVIDERS_URI`) are
resolved in `nfp_lookups.paths` via `upath_for(uri)` and the `*_location()`
helpers (`data_location()`, `providers_location()`, etc.). The exception is
`NFP_SNAPSHOTS_URI`, which is resolved by `nfp_ingest.snapshots.snapshots_location()`
— same `upath_for` / `storage_options_for` pattern, just in a different package.

### S3 credentials

When an S3 URI is set, the code also reads `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, and `AWS_ENDPOINT_URL` (MinIO endpoint; region defaults
to `us-east-1`). These are loaded from `.env` (gitignored) by the root
`conftest.py` for tests and by the CLI at startup.

## Polars I/O helpers

Wherever Polars reads or writes a remote parquet, the call must pass credential
options via `nfp_lookups.paths.storage_options_for(path)`. Example pattern used
throughout the codebase:

```python
from nfp_lookups.paths import storage_options_for, is_remote

opts = storage_options_for(target_path)
df.write_parquet(str(target_path), storage_options=opts)
```

The `is_remote(path)` guard prevents `mkdir()` calls on S3 prefixes (which would
fail, since S3 has no directory objects):

```python
if not is_remote(target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)
```

Pass `str(path)` to Polars functions when the path is a `UPath` — Polars does not
accept `UPath` objects directly in all versions.

## Rebuild-to-scratch, promote deliberately

The vintage store is **replaceable** — it is reconstructed from publicly available
CES/QCEW bulk downloads. It is **not** append-only or irreplaceable. This has two
operational consequences:

**Never write a rebuild directly to the canonical store.** The one-time historical
rebuild is `scripts/bootstrap_store.py`, a script (not a CLI command):

```bash
uv run python scripts/bootstrap_store.py \
    --scratch s3://alt-nfp/store-rebuild \
    --canonical s3://alt-nfp/store
```

The script writes to a scratch prefix (`NFP_STORE_URI=s3://alt-nfp/store-rebuild`)
and the promotion to canonical is a deliberate copy-then-delete cutover — not a
mirror overwrite. The reason: store filenames encode vintage ranges, so a plain
overwrite would leave orphaned files from the prior schema and corrupt the store.
Snapshot the prior canonical first, then copy-then-delete per partition.

**The `is_canonical_store` guard.** The functions `write_rebuild_store`,
`build_store`, and `mirror_store` check whether the target is the canonical store
prefix and refuse to write unless the caller passes an explicit override. This
prevents accidental clobber. `bootstrap_store.py` also refuses if `--scratch`
resolves to the canonical path.

**The `alt-nfp` CLI has no `build` command.** The everyday production CLI
(`alt-nfp update` / `alt-nfp status` / `alt-nfp watch` / `alt-nfp snapshot`)
handles real-time capture and monitoring. There is no `alt-nfp build` — the
historical rebuild is always a script, never a CLI subcommand.
