# CLI Reference

`alt-nfp` is the production CLI for the vintage store. It covers the monthly
real-time capture path and the BLS-feed-driven automation; one-time historical
reconstruction is a separate script (see [Bootstrap script](#bootstrap-script)
below).

```
alt-nfp --help
alt-nfp <command> --help
```

---

## Production workflow overview

```
                  daily cron
                      │
                 alt-nfp watch          ← polls BLS RSS feed
                      │ new release?
                      ▼
             alt-nfp update --as-of T   ← advance calendar + capture prints
                      │
                      ▼
             alt-nfp status [--as-of T] ← verify nothing is uncaptured
                      │
                      ▼
           alt-nfp snapshot --as-of T   ← bake hash-pinned ModelData (day 12)
```

Each month's vintage is the print BLS publishes, captured in real time and
appended to the store. Historical reconstruction is a one-time bootstrap
operation, not part of the monthly path.

---

## Commands

### `update`

Advance the release calendar, capture knowable prints for the given date, and
append them to the store.

```bash
alt-nfp update --as-of 2026-01-12
alt-nfp update --as-of 2026-01-12 --only ces
alt-nfp update --as-of 2026-01-12 --only qcew
alt-nfp update --as-of 2026-01-12 --only indicators
alt-nfp update --as-of 2026-01-12 --no-refresh-calendar
```

| Flag | Required | Description |
|---|---|---|
| `--as-of DATE` | yes | Knowledge cutoff (`YYYY-MM-DD`). Rows with `vintage_date > DATE` are not appended. When called by `watch`, this is the feed `pubDate` (the actual release day). |
| `--only SOURCE` | no | Restrict to one source: `ces`, `qcew`, or `indicators`. Omit to run all three. |
| `--no-refresh-calendar` | no | Skip the release-calendar scrape. Use only when the calendar already covers the target date. |

**What it does**

1. **Advances the release calendar** (`vintage_dates.parquet`) to `--as-of`
   before any capture — mandatory, because the CES tagging step and the QCEW
   knowability guard both read this calendar. An un-advanced calendar produces a
   silent empty capture.
2. **CES** — fetches the current print via the BLS JSON API, tags
   vintage/revision metadata, and appends to the store. Requires `BLS_API_KEY`.
3. **QCEW** — conditional; a no-op most months (QCEW is quarterly). Captures
   the quarter whose rev-0 `vintage_date` is ≤ `--as-of`, if one is newly
   knowable.
4. **Indicators** — calls `download_indicators()` (a full FRED refresh/overwrite,
   not a vintage append). Requires `FRED_API_KEY`.

After each source, `update` compacts any partition that has more than one
fragment — the self-healing path for a crash between append and compact. This
compaction runs on the **local store only**; it is skipped on a remote `s3://`
store.

Output lines signal what happened:

```
  CES: appended 3, skipped 0
  QCEW: appended 12, skipped 0
  CORRECTED-LEVEL ces 2025-11-01 05 rev0/bmr0: 159234.0 -> 159190.0
  Indicators: 2880 rows across 4 series
```

A `CORRECTED-LEVEL` line means BLS published a corrected level for an
already-stored revision. The store keeps the original; the correction requires
manual review (auto-replacement is not implemented).

---

### `status`

Read-only store coverage and "what's uncaptured" health report.

```bash
alt-nfp status
alt-nfp status --as-of 2026-01-12
alt-nfp status --store s3://alt-nfp/store
alt-nfp status --as-of 2026-01-12 --store s3://alt-nfp/store
```

| Flag | Required | Description |
|---|---|---|
| `--as-of DATE` | no | Sets the UNCAPTURED alarm cutoff (`YYYY-MM-DD`). The alarm fires if the store lags behind what BLS should have published by this date. Does **not** censor vintage reads. |
| `--store URI` | no | Override the store URI or path. Accepts `s3://` and local paths. Default: `VINTAGE_STORE_PATH` (from `NFP_STORE_URI` env var, or the local `data/store/` fallback). |

**Report contents**

- **Header** — resolved store URI with `REMOTE` / `LOCAL` / `CANONICAL` flags.
  When reading a local store, the header includes a prominent `LOCAL FALLBACK`
  warning. This is always printed when `NFP_STORE_URI` is unset or points to a
  local path, whether intentional or because the `.env` file was not loaded.
- **Per-source coverage** — for each `(source, seasonally_adjusted)` partition:
  earliest/latest `ref_date`, row count, last capture (`max(vintage_date)`),
  distinct vintage count.
- **UNCAPTURED alarm** — per source, flags any ref-month or ref-quarter that BLS
  should have published by `--as-of` but is absent from the store. A missed
  monthly CES capture cannot be recovered after the fact (the BLS API has no
  historical memory of first prints).
- **Missing-month list** — gaps in the headline series, with known-shutdown months
  annotated rather than flagged as errors.
- **CORRECTED-LEVEL rows** — corrections detected at capture time that were not
  auto-applied.

---

### `watch`

Poll the BLS release feed and trigger `update` when a new release is detected.
Designed to run as a daily cron job.

```bash
alt-nfp watch
alt-nfp watch --source ces
alt-nfp watch --source qcew
alt-nfp watch --source all --snapshot
alt-nfp watch --store s3://alt-nfp/store
```

| Flag | Required | Description |
|---|---|---|
| `--source SOURCE` | no | Which feed(s) to poll: `all` (default), `ces`, or `qcew`. |
| `--snapshot` | no | After capturing a new release, also write a hash-pinned ModelData snapshot at the day-12 anchor for that ref-month. |
| `--store URI` | no | Override the store URI or path (same semantics as `status --store`). |

**Behavior**

`watch` fetches the BLS RSS feed(s), picks the most-recent item by `pubDate`,
then checks the store to see whether that release's ref-period is already
captured. On days with nothing new it is a clean no-op. A same-day CES + QCEW
co-release triggers both source updates.

When `--snapshot` is set, the snapshot date is the **day-12 anchor** of the
captured ref-month (e.g., `2026-01-12`), not the raw `pubDate` — because
`snapshot --as-of` enforces the day-12 convention and would reject a
release-day date that falls on a different day.

A typical cron line (daily at 08:30):

```cron
30 8 * * * cd /path/to/project && uv run alt-nfp watch --source all --snapshot
```

---

### `snapshot`

Write one or more hash-pinned ModelData snapshots — the handoff from the
vintage store to the model layer.

```bash
alt-nfp snapshot --as-of 2026-01-12
alt-nfp snapshot --as-of 2026-01-12 --grid-end 2026-06-12
```

| Flag | Required | Description |
|---|---|---|
| `--as-of DATE` | yes | Knowledge cutoff (`YYYY-MM-DD`). **Must fall on the 12th** — enforced for both single-date and grid runs. |
| `--grid-end DATE` | no | If set, write a snapshot for every month's 12th from `--as-of` through this date (inclusive). |

The day-12 convention is enforced unconditionally: passing any other day raises
a `BadParameter` error. Each snapshot is written to `NFP_SNAPSHOTS_URI` (or the
local `data/snapshots/` fallback) and identified by a content hash.

Output per date:

```
  2026-01-12: s3://alt-nfp/snapshots/2026-01-12.parquet (hash a3f1c9e2b4d8)
```

---

## Environment variables

| Variable | Role | Fallback |
|---|---|---|
| `NFP_STORE_URI` | Vintage store location (`s3://` or local path) | `data/store/` (local) |
| `NFP_SNAPSHOTS_URI` | Snapshot output location | `data/snapshots/` (local) |
| `NFP_DATA_URI` | Indicator and calendar files | `data/` (local) |
| `BLS_API_KEY` | Required by `update` for CES JSON API fetch | — (hard failure) |
| `FRED_API_KEY` | Required by `update --only indicators` | — (hard failure) |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | S3/MinIO credentials | — |
| `AWS_ENDPOINT_URL` | MinIO endpoint (omit for real AWS S3) | — |

Load these from a `.env` file at the project root. The CLI calls `load_dotenv()`
automatically before any command resolves store paths.

!!! warning "LOCAL FALLBACK"
    If `NFP_STORE_URI` is unset or the `.env` file is not loaded, all commands
    silently read and write the local `data/store/` directory. The `status`
    command prints a `LOCAL FALLBACK` warning in this case. In a container
    environment (e.g., Bloomberg), this means writes go to ephemeral disk and
    are lost on restart.

---

## Bootstrap script

`scripts/bootstrap_store.py` is a **one-time script**, not a CLI command. It
reconstructs the full historical store from raw BLS sources and promotes it to
the canonical location. Run it when setting up a new environment or after a
schema change that requires a full rebuild.

```bash
NFP_STORE_URI=s3://alt-nfp/store-rebuild \
  uv run python scripts/bootstrap_store.py \
  --scratch s3://alt-nfp/store-rebuild \
  --canonical s3://alt-nfp/store
```

| Flag | Required | Description |
|---|---|---|
| `--scratch URI` | yes | Scratch store URI/path for the rebuild output. **Must not be the canonical store.** |
| `--canonical URI` | yes | Canonical store URI/path to promote into after the rebuild. |
| `--start-year YEAR` | no | First QCEW reference year (default: `2017`). |
| `--end-year YEAR` | no | Last QCEW reference year, inclusive (default: current year). |
| `--no-promote` | no | Build the scratch store but skip the promote step. |

**Rebuild pipeline** (in order):

1. Download CES triangular CSVs to a run-scoped temp directory (`cesvinall/`).
2. Advance the release calendar (`vintage_dates.parquet`) for overlap parity.
3. Build the CES panel (NSA + SA store-schema rows).
4. Acquire QCEW levels and size-class data from the CEW API; build panels.
5. Compose the combined panel.
6. Write to the scratch store (`write_rebuild_store`, `allow_canonical=False`).
7. Promote scratch → canonical (copy-then-delete per partition).

**Scratch-then-promote safety model**

The promote step copies rebuilt files into the canonical prefix and then deletes
any old files that are no longer in the rebuilt set. This copy-then-delete
pattern is required because store filenames encode vintage ranges — a plain
overwrite would leave both old and new files in the partition, corrupting the
store. `scripts/mirror_store.py` is deliberately not used in the bootstrap
promote for this reason.

The script refuses to write directly to the canonical store — if `--scratch`
resolves to the canonical path, it exits immediately:

```
refusing to bootstrap straight to the canonical store (s3://alt-nfp/store);
target a scratch prefix (e.g. s3://alt-nfp/store-rebuild).
```

Use `--no-promote` to inspect the scratch store before committing to the
canonical location. Scratch and canonical must share the same backend (both
`s3://` or both local).

**Scope:** national-only, 2017 onward. QCEW is fetched live from the CEW API
(not the bulk ZIPs). No byproduct lands under `./data` — raw downloads go to a
temp directory that is cleaned up at the end of the run.
