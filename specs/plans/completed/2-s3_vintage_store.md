# Implementation Plan: S3-Backed Vintage Store (MinIO)

> **Status: ✅ COMPLETE (2026-06-12).** All steps done and verified live
> against MinIO:
>
> - Local mode (no `.env`): 339 passed / 23 skipped — identical behavior to
>   before, store tests skip. This is what CI sees.
> - Remote mode (`.env` with `NFP_STORE_URI=s3://alt-nfp/store`): **356
>   passed / 6 skipped** — 17 store-dependent tests un-skipped and pass
>   against MinIO. Remaining 6 skips: 5 need `data/intermediate/`
>   release/vintage-dates parquets (pipeline artifacts, not yet generated in
>   this repo — A0 gate work), 1 is the intentional local-mode-only
>   `VINTAGE_STORE_PATH == STORE_DIR` invariant.
> - Migration: bucket `alt-nfp` created; old repo's store (3 parquet files)
>   mirrored. Parity check: **770,568 rows, value-identical** between
>   `s3://alt-nfp/store` and the old repo's local store.
> - Write-path smoke (scratch prefix, then cleaned): append → dedup-append
>   (0 rows, correctly) → read-back → `compact_partition` (fragments →
>   single `compacted.parquet`) all work against MinIO.
> - `ruff check .` clean.
>
> Deviation from plan: none material. Bonus: the `python-dotenv` bug fix
> (2s.1) means `uv run alt-nfp` no longer crashes at import.

Goal: the vintage store lives in an S3 bucket served by the local MinIO
instance (`http://127.0.0.1:9000`, LaunchAgent `com.lowell.minio`, data root
`~/S3`) instead of `data/store/` on disk. Local-path operation must keep
working — tests build throwaway stores in `tmp_path`, and CI has no MinIO.

## Design decisions

- **`UPath` (universal-pathlib), not a signature rewrite.** Store functions
  do pathlib things (`/` joins, `exists`, `glob`, `mkdir`, `unlink`).
  `upath.UPath("s3://…")` keeps that API over s3fs, so `store_path` stays
  one object threaded through existing signatures. Polars itself talks to S3
  natively (rust object_store) via `storage_options` — s3fs is only used for
  the pathlib-style operations and file deletes.
- **One env contract, read by `nfp_lookups.paths` at import** (same pattern
  as `NFP_BASE_DIR`):
  - `NFP_STORE_URI` — e.g. `s3://alt-nfp/store`. Unset ⇒ local `STORE_DIR`
    exactly as today.
  - `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` — MinIO root credentials.
  - `AWS_ENDPOINT_URL` — `http://127.0.0.1:9000`.
  - `AWS_REGION` — optional, defaults `us-east-1`.
  - `aws_allow_http` is derived from the endpoint scheme, not configured.
- **`VINTAGE_STORE_PATH` becomes the switch point.** It is already the
  default `store_path` everywhere that matters; when `NFP_STORE_URI` is set
  it becomes a `UPath`, otherwise it stays `STORE_DIR` (a `Path`). Consumers
  that default to `STORE_DIR` directly (`panel.py`, `payroll.py`) are
  repointed to `VINTAGE_STORE_PATH`.
- **Scope: the store only.** `data/downloads/`, `data/intermediate/`,
  `data/indicators/`, provider files, and `output/` stay local — they are
  pipeline scratch or inputs, not the shared artifact. Phase A2's ModelData
  snapshots should target the same bucket later (`s3://alt-nfp/snapshots/`).
- **Bucket layout:** bucket `alt-nfp`, prefix `store/` →
  `s3://alt-nfp/store/source=ces/seasonally_adjusted=true/….parquet`
  (hive layout unchanged).
- **CI is unaffected:** no `NFP_STORE_URI` in CI ⇒ local mode ⇒ store tests
  skip, as today.

## Steps

### 2s.1 Dependencies + the dotenv bug

1. `nfp-lookups`: add `universal-pathlib` and `s3fs` (the paths module owns
   storage location, so it owns the deps; imported lazily — local-only users
   never touch s3fs at runtime, but it's installed workspace-wide anyway).
2. **Bug fix:** `nfp_vintages/__main__.py` does `from dotenv import
   load_dotenv` but `python-dotenv` is declared nowhere and not installed —
   `uv run alt-nfp` crashes today (`ModuleNotFoundError`). Add
   `python-dotenv` to nfp-vintages dependencies.
3. Add `python-dotenv` to the root `dev` dependency group (for the test
   conftest, next step).

### 2s.2 Env loading for tests

Root `conftest.py`: `load_dotenv()` at module top so pytest sessions see
`NFP_STORE_URI` + `AWS_*` before any `nfp_lookups` import. The CLI already
calls `load_dotenv()` itself. `.env` is gitignored (already).

### 2s.3 `nfp_lookups.paths`: URI support

1. `_store_location() -> Path | UPath`: returns `UPath(NFP_STORE_URI,
   key=…, secret=…, client_kwargs={"endpoint_url": …})` when the env var is
   set (lazy `upath` import), else `STORE_DIR`.
2. `VINTAGE_STORE_PATH = _store_location()`.
3. `storage_options_for(path) -> dict[str, str] | None`: polars/object_store
   options for a remote path (`aws_access_key_id`, `aws_secret_access_key`,
   `aws_endpoint_url`, `aws_region`, `aws_allow_http` when the endpoint is
   plain http); `None` for local paths. Takes the path (not global state) so
   explicitly-passed UPaths work too.
4. `is_remote(path) -> bool` helper (protocol check).

### 2s.4 `nfp_ingest.vintage_store`: remote-aware I/O

- `read_vintage_store`: scan `str(store_path / "**/*.parquet")` (works for
  both `Path` and `UPath`) and pass
  `storage_options=storage_options_for(store_path)`.
- `append_to_vintage_store`: existing-partition read gets `str(...)` +
  storage options; `mkdir` guarded to local paths (S3 has no directories);
  writes go through polars `write_parquet(str(target),
  storage_options=...)`.
- `compact_partition`: same treatment; `UPath.glob`/`unlink` already work
  via s3fs.
- Type hints widen to `Path | UPath` where store paths flow.

### 2s.5 Downstream defaults

- `nfp_ingest/panel.py` and `nfp_ingest/payroll.py`: default store location
  `STORE_DIR` → `VINTAGE_STORE_PATH`.
- `nfp_vintages/build_store.py`: same `mkdir` guard + remote-aware write for
  its partition writes (revisions/releases inputs stay local reads).

### 2s.6 Tests

1. `test_paths.py`: `VINTAGE_STORE_PATH == STORE_DIR` invariant becomes
   conditional on `NFP_STORE_URI` being unset; add unit tests for
   `storage_options_for` (env-driven, no network): remote path with http
   endpoint ⇒ `aws_allow_http`, local path ⇒ `None`.
2. `test_store_coverage.py` skipif + `test_ingest.py:174` skip: probe
   `VINTAGE_STORE_PATH` instead of `STORE_DIR`, wrapped so an unreachable
   endpoint ⇒ skip, not error.
3. Existing `tmp_path` store tests keep passing untouched (functions still
   accept plain `Path`).

### 2s.7 Migration + local env

1. Write `.env` (gitignored): `NFP_STORE_URI`, `AWS_*` from
   `~/.config/minio/env`, plus API keys carried from the old repo's `.env`
   (FRED/BLS/BEA/CENSUS).
2. Create bucket `alt-nfp` and mirror the old repo's `data/store/` (5.5 MB)
   into `s3://alt-nfp/store/` via a small `scripts/mirror_store.py` (s3fs;
   kept for future re-syncs since no `mc`/`aws` CLI is installed).
3. **Verification gate:** with `.env` present, the ~23 store-dependent tests
   un-skip and pass against MinIO; without it (CI mode), they skip; full
   suite + ruff green both ways; a manual `read_vintage_store().collect()`
   sanity-checks row counts against the old repo's store.

### 2s.8 Docs

- Root `CLAUDE.md` + `README.md`: store lives in MinIO/S3; env contract;
  local fallback.
- `packages/nfp-lookups/CLAUDE.md`: paths pattern gains the store-URI rules.
- `packages/nfp-ingest/CLAUDE.md`: store path accepts local or `s3://`.

## Acceptance criteria

- `NFP_STORE_URI` unset: suite identical to today (store tests skip; no new
  deps imported at runtime).
- `NFP_STORE_URI` set to the MinIO bucket: store-dependent tests pass
  against S3; `read_vintage_store` row counts match the old repo's local
  store exactly.
- `ruff check .` clean; no package other than nfp-lookups reads storage env
  vars.

## Deferred

- Moving `downloads/`/`intermediate/`/`indicators/`/provider files to S3 —
  revisit when the full pipeline (`alt-nfp`) is exercised for the A0 gate.
- ModelData snapshots to `s3://alt-nfp/snapshots/` — Phase A2.
- MinIO bucket versioning/replication policy — operational, not code.
