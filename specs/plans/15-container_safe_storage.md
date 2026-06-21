# Container-Safe Storage Implementation Plan (plans/15)

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.
> **Status:** DRAFT — awaiting maintainer go-ahead before any code edits (high blast radius
> near the store path; see Global Constraints).

**Goal:** Make the code stop writing under `./data/` so it runs on Bloomberg's small-footprint
container: persistent artifacts the production nowcast reads go to **S3**; rebuild-only scratch
goes to **`tempfile`** under `/tmp` and is deleted when the run ends.

**Architecture:** Extend the *existing* env-URI pattern — `_store_location()`/`NFP_STORE_URI`
and `snapshots_location()`/`NFP_SNAPSHOTS_URI` already return a `pathlib.Path` **or** a
`upath.UPath` and guard writes with `storage_options_for()` + `is_remote()`. Add one sibling,
`data_location()`/`NFP_DATA_URI`, route the production-read artifact roots through it, and
convert the bulky rebuild byproducts to `tempfile`. No new dependency, no new mechanism.

**Tech stack:** Python 3.12, Polars, NumPy, `universal_pathlib`/`s3fs`, Typer CLI, pytest,
uv workspace.

---

## Global Constraints

- **Hard rule (the whole point):** no code path may write under `./data/` on Bloomberg.
  Permanent → S3; temporary → `tempfile` (under `/tmp`), removed when no longer needed.
- **Reuse the established idiom**, do not invent a parallel one: a `*_location()` function
  reads an env URI and returns `Path | UPath`; every Polars write/read passes
  `storage_options=storage_options_for(path)`; `mkdir` is called only when `not is_remote(path)`;
  `np.savez`/byte writes go through `path.open("wb")` (UPath-compatible).
- **Tier-C decision (maintainer, 2026-06-20):** ALL rebuild inputs/intermediates → `tempfile`
  (accept re-downloading raw CES/QCEW from BLS each rebuild). Do **not** persist `downloads/`
  or the bulky `intermediate/` revisions to S3.
- **Store-write safety (carry-over, hard — see memory `store-write-test-safety`):** never run a
  store-writing function against the real/canonical store in a test (`tmp_path`/monkeypatched
  env only). This plan does not touch `NFP_STORE_URI`, `build_store`, `rebuild_store`, or
  `is_canonical_store`; leave the store path exactly as-is.
- **Line length 100; ruff E,W,F,I,B,C4,UP. Tests must be `-m "not network"`-clean**
  (use `monkeypatch` + `tmp_path`; MinIO round-trips are manual verification only).
- Python floor 3.12; no new third-party deps (`upath`/`s3fs` already present via the store).

---

## Classification recap (from the 2026-06-20 audit)

| Tier | Artifacts | Written by | Verdict |
|---|---|---|---|
| **already-safe** | `store/`, `snapshots/` | `vintage_store`/`build`/`rebuild`; `snapshots.py` | leave as-is; set `NFP_STORE_URI`/`NFP_SNAPSHOTS_URI` on Bloomberg |
| **already-safe** | model posteriors | `nfp-model` | writes nothing to disk |
| **A → S3** | `indicators/`, `competitors/consensus.parquet`, `intermediate/vintage_dates.parquet`, `intermediate/release_dates.parquet` | `indicators.py`, `competitors/consensus.py`, `vintage_dates.py`/`__main__.py`, `__main__.py` | **Phase 1** |
| **A′ → separate store** | `providers/g/` | external drop-in; read by `payroll.py` | **Phase 1** — point at the **separate Bloomberg provider store** via `NFP_PROVIDERS_URI`; **not** under `NFP_DATA_URI`, **not** seeded by us (maintainer, 2026-06-20) |
| **C → tempfile** | `downloads/` (raw CES/QCEW/HTM), `intermediate/{ces,qcew,sae}_revisions.parquet` + `revisions.parquet` + `sae_checkpoint.parquet`, HTTP cache | `bls/bulk.py`, `processing/*`, `release_dates/scraper.py`, `bls/_http.py` | **Phase 2** |
| **D → caller-supplied** | `backtests/`, `golden_*`, `a3/` | dev scripts (`sys.argv[1]`) | **Phase 3** (point at `/tmp` or `s3://`; fix 2 hardcoders) |

`vintage_dates.parquet`/`release_dates.parquet` live under `intermediate/` but are **production
inputs** (`releases.py:74-85` reads `VINTAGE_DATES_PATH`), so they are Tier-A (S3), carved out
of the Tier-C "intermediate → tempfile" rule.

---

# Phase 1 — Tier A: persistent artifacts → S3

### Task 1: `data_location()` + `NFP_DATA_URI` in `nfp_lookups.paths`

**Files:**
- Modify: `packages/nfp-lookups/src/nfp_lookups/paths.py`
- Modify: `packages/nfp-lookups/src/nfp_lookups/__init__.py` (export new names)
- Test: `packages/nfp-lookups/src/nfp_lookups/tests/test_paths.py`

**Interfaces:**
- Produces: `data_location() -> Path | UPath` and `providers_location() -> Path | UPath` (the
  separate provider store, `NFP_PROVIDERS_URI`); constant `COMPETITORS_DIR` (new) and re-rooted
  `INDICATORS_DIR`, `VINTAGE_DATES_PATH`, `RELEASE_DATES_PATH`; `PROVIDERS_DIR` (new) rooted via
  `providers_location()`, **not** `data_location()`. All resolve to local `DATA_DIR/...` when the
  relevant env var is unset (preserves current behaviour + CI).

- [ ] **Step 1: Write the failing test**

```python
# test_paths.py
def test_data_location_local_when_unset(monkeypatch):
    monkeypatch.delenv("NFP_DATA_URI", raising=False)
    from importlib import reload
    from nfp_lookups import paths
    reload(paths)
    assert paths.data_location() == paths.DATA_DIR
    assert paths.INDICATORS_DIR == paths.DATA_DIR / "indicators"

def test_data_location_remote_when_set(monkeypatch):
    monkeypatch.setenv("NFP_DATA_URI", "s3://alt-nfp")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://127.0.0.1:9000")
    from importlib import reload
    from nfp_lookups import paths
    reload(paths)
    loc = paths.data_location()
    assert paths.is_remote(loc)
    assert str(loc / "indicators").startswith("s3://alt-nfp/indicators")
    reload(paths)  # restore module state for other tests
```

- [ ] **Step 2: Run it, verify it fails** — `uv run pytest packages/nfp-lookups/src/nfp_lookups/tests/test_paths.py -k data_location -v` → FAIL (`data_location` undefined).

- [ ] **Step 3: Implement.** In `paths.py`, after `_store_location()` add a sibling that reuses
  the same credential plumbing, then re-root the Tier-A constants:

```python
def _upath(uri: str) -> Any:
    """Build a credentialed UPath from env (shared by all *_location helpers)."""
    from upath import UPath  # deferred: s3fs only needed in remote mode

    client_kwargs = {}
    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    if endpoint:
        client_kwargs["endpoint_url"] = endpoint
    return UPath(
        uri,
        key=os.environ.get("AWS_ACCESS_KEY_ID"),
        secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        client_kwargs=client_kwargs,
    )


def data_location() -> Any:
    """Root for PERSISTENT non-store data artifacts (indicators, competitors,
    derived release/vintage schedules).

    ``NFP_DATA_URI`` (e.g. ``s3://alt-nfp``) selects object storage; unset selects
    the local ``DATA_DIR``. Returns ``Path`` or ``UPath`` — same env/credential
    contract as :func:`_store_location`. Bulky rebuild byproducts (downloads, the
    revisions intermediates) are NOT routed here; they use tempfile (plans/15 Tier C).
    Provider data lives on a SEPARATE store — see :func:`providers_location`.
    """
    uri = os.environ.get("NFP_DATA_URI")
    return _upath(uri) if uri else DATA_DIR


def providers_location() -> Any:
    """Root for provider parquets — a SEPARATE store from the alt-nfp data bucket.

    On Bloomberg the provider data lives on its own compute store (maintainer,
    2026-06-20), so it gets its own env var ``NFP_PROVIDERS_URI`` and is NOT seeded
    by this repo. Unset → local ``DATA_DIR`` (current dev behaviour). The relative
    ``ProviderConfig.file`` (e.g. ``providers/g/g_provider.parquet``) joins to this root.
    """
    uri = os.environ.get("NFP_PROVIDERS_URI")
    return _upath(uri) if uri else DATA_DIR


_DATA_ROOT = data_location()
INDICATORS_DIR = _DATA_ROOT / "indicators"
COMPETITORS_DIR = _DATA_ROOT / "competitors"
PROVIDERS_DIR = providers_location()  # the provider store root; cfg.file joins to it
RELEASE_DATES_PATH = _DATA_ROOT / "intermediate" / "release_dates.parquet"
VINTAGE_DATES_PATH = _DATA_ROOT / "intermediate" / "vintage_dates.parquet"
```

  Delete the old local-only assignments of `INDICATORS_DIR`, `RELEASE_DATES_PATH`,
  `VINTAGE_DATES_PATH` (lines ~55, ~61-62). **Keep `DOWNLOADS_DIR`, `INTERMEDIATE_DIR`,
  `RELEASES_DIR`, `STORE_DIR`, `OUTPUT_DIR` local — they are Tier C / out of scope.**
  Export `data_location`, `providers_location`, `COMPETITORS_DIR`, `PROVIDERS_DIR` from
  `__init__.py`. (`_store_location` should later be refactored to reuse `_upath`, but that is a
  store-path edit — out of scope here per the store-safety constraint.)

- [ ] **Step 4: Run tests, verify pass** — `uv run pytest packages/nfp-lookups/src/nfp_lookups/tests/test_paths.py -v` → PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(paths): add data_location()/NFP_DATA_URI for persistent non-store artifacts"`

---

### Task 2: indicators read/write → S3-capable

**Files:**
- Modify: `packages/nfp-ingest/src/nfp_ingest/indicators.py`
- Test: `packages/nfp-ingest/src/nfp_ingest/tests/test_cyclical_indicators.py` (or `test_new_ingest.py`)

**Interfaces:**
- Consumes: `INDICATORS_DIR` (now `Path | UPath`), `storage_options_for`, `is_remote`.

- [ ] **Step 1: Failing test** — round-trip through a fake-remote path object. Simplest
  no-network proof is the storage-options/guard wiring; assert local still round-trips and that
  a remote-typed path skips `mkdir`:

```python
def test_download_indicators_skips_mkdir_on_remote(monkeypatch, tmp_path):
    from nfp_ingest import indicators
    # local path still works end-to-end
    df = pl.DataFrame({"ref_date": [date(2020,1,1)], "value": [1.0]})
    monkeypatch.setattr(indicators, "fetch_fred_series", lambda *a, **k: df)
    out = indicators.download_indicators(
        indicators=[{"name": "claims", "fred_id": "X", "freq": "weekly"}],
        store_dir=tmp_path, api_key="fake",
    )
    assert out["claims"] == 1
    assert (tmp_path / "claims.parquet").exists()
    assert indicators.read_indicator("claims", store_dir=tmp_path).height == 1
```

- [ ] **Step 2: Verify current behaviour** — run it; it should PASS today (local). The remote
  guard is exercised by inspection + the manual MinIO step (Task 12). Add the guard now.

- [ ] **Step 3: Implement.** Pass storage options on read/write; guard `mkdir`:

```python
from nfp_lookups.paths import INDICATORS_DIR, is_remote, storage_options_for

# read_indicator():
    return pl.read_parquet(fpath, storage_options=storage_options_for(fpath))

# download_indicators():
    if not is_remote(store_dir):
        store_dir.mkdir(parents=True, exist_ok=True)
    ...
    df.write_parquet(out_path, storage_options=storage_options_for(out_path))
```

  Widen the `store_dir: Path` hints to `store_dir: Path | Any` (UPath is `os.PathLike`).

- [ ] **Step 4: Run** `uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/ -k indicator -v` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(indicators): route reads/writes through storage_options + is_remote guard"`

---

### Task 3: consensus path → S3-correct (+ promote `_upath` to public `upath_for`)

**Why the rename:** consensus lives in `nfp_vintages` and needs the credentialed-UPath builder,
but the repo's hard boundary rule forbids cross-package imports of underscore-private names. Task 1
created it as `_upath` (private); promote it to a public `upath_for` so `nfp_vintages` may import it.

**Files:**
- Modify: `packages/nfp-lookups/src/nfp_lookups/paths.py` (rename `_upath` → `upath_for`; update its two callers `data_location`/`providers_location`)
- Modify: `packages/nfp-lookups/src/nfp_lookups/__init__.py` (export `upath_for`)
- Modify: `packages/nfp-vintages/src/nfp_vintages/competitors/consensus.py`
- Test: `packages/nfp-vintages/src/nfp_vintages/tests/test_competitors.py` (consensus tests already live here)

**Interfaces:**
- Consumes: `upath_for` (public, from `nfp_lookups.paths`), `COMPETITORS_DIR`, `storage_options_for`.

- [ ] **Step 1: Promote the helper.** In `paths.py` rename `def _upath(uri)` → `def upath_for(uri)`
  and update the two call sites (`data_location`, `providers_location`). Export `upath_for` from
  `__init__.py`. Confirm no `_upath` references remain: `grep -rn "_upath" packages/` returns nothing.

- [ ] **Step 2: Failing test** — an `s3://` env value must survive resolution (today
  `Path("s3://…")` collapses the `//`):

```python
def test_consensus_path_preserves_s3_uri(monkeypatch):
    monkeypatch.setenv("NFP_CONSENSUS_PATH", "s3://alt-nfp/competitors/consensus.parquet")
    from nfp_vintages.competitors.consensus import consensus_path
    assert str(consensus_path()).startswith("s3://alt-nfp/competitors/consensus.parquet")
```

- [ ] **Step 3: Run, verify fail** (`Path` mangles `s3://` → `s3:/`).

- [ ] **Step 4: Implement.** Resolve env URIs as UPath via the public `upath_for`, default through
  `COMPETITORS_DIR`, and pass storage options on read:

```python
def consensus_path(path=None):
    """Resolve path -> arg -> NFP_CONSENSUS_PATH -> COMPETITORS_DIR/consensus.parquet."""
    if path is not None:
        return _as_path(path)
    env = os.environ.get("NFP_CONSENSUS_PATH")
    if env:
        return _as_path(env)
    from nfp_lookups.paths import COMPETITORS_DIR
    return COMPETITORS_DIR / "consensus.parquet"

def _as_path(p):
    s = str(p)
    if s.startswith(("s3://", "s3a://")):
        from nfp_lookups.paths import upath_for  # public credentialed-UPath builder
        return upath_for(s)
    return Path(p)

# load_consensus():
    from nfp_lookups.paths import storage_options_for
    df = pl.read_parquet(p, storage_options=storage_options_for(p))
```

- [ ] **Step 5: Run** `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_competitors.py -q --no-cov` → PASS.
- [ ] **Step 6: Commit** — `git commit -m "feat(consensus): s3:// URIs + COMPETITORS_DIR; promote paths.upath_for to public"`

> **DRY note:** the shared `paths.upath_for(uri)` helper is the single credentialed-`UPath`
> constructor; `data_location`, `providers_location`, and `consensus._as_path` all call it.

---

### Task 4: provider parquets → read from the separate provider store

> Providers are **not** in `s3://alt-nfp` and are **not** seeded by us — on Bloomberg they live
> on a separate compute store (maintainer, 2026-06-20). This task makes the read path resolve via
> `providers_location()`/`NFP_PROVIDERS_URI` and S3-capable; it does **no** uploading.

**Files:**
- Modify: `packages/nfp-ingest/src/nfp_ingest/payroll.py` (provider file resolution + reads)
- Test: `packages/nfp-ingest/src/nfp_ingest/tests/test_new_ingest.py` (provider load path)

**Interfaces:**
- Consumes: `providers_location()` / `PROVIDERS_DIR` (the separate provider store root),
  `storage_options_for`. `ProviderConfig.file` stays a relative posix string
  (`"providers/g/g_provider.parquet"`); only the ROOT it joins to changes.

- [ ] **Step 1: Failing/guard test** — provider file resolves under `data_location()` not raw
  `DATA_DIR`, and reads pass storage options. Keep it local (tmp_path) end-to-end:

```python
def test_provider_resolves_under_data_root(monkeypatch, tmp_path):
    # write a fake provider parquet under tmp_path/providers/g/
    ...
    monkeypatch.setattr(payroll, "DATA_DIR", tmp_path)  # current default arg source
    series = payroll.load_provider_series(ProviderConfig(name="G", file="providers/g/g_provider.parquet", ...))
    assert series is not None
```

- [ ] **Step 2: Run** → confirm current behaviour, then harden.

- [ ] **Step 3: Implement.** Where `payroll.py` resolves `_data_dir = data_dir or DATA_DIR` and
  reads `pl.read_parquet(str(fpath))` (lines ~45, ~61, ~92, ~176): default `_data_dir` to
  `providers_location()` instead of `DATA_DIR`, and add
  `storage_options=storage_options_for(fpath)` to every provider `pl.read_parquet`. Provider files
  are read-only inputs (no `mkdir`/write to guard).

- [ ] **Step 4: Run** `uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/ -k provider -v` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(payroll): resolve+read provider parquets via data_location() (S3-capable)"`

---

### Task 5: vintage_dates / release_dates writers + readers → S3-capable

**Files:**
- Modify: `packages/nfp-ingest/src/nfp_ingest/release_dates/vintage_dates.py:419-420`
- Modify: `packages/nfp-ingest/src/nfp_ingest/releases.py:74-85` (reader)
- Modify: `packages/nfp-vintages/src/nfp_vintages/__main__.py:160-167` (CLI writer)
- Test: `packages/nfp-ingest/src/nfp_ingest/tests/test_release_dates.py`

**Interfaces:**
- Consumes: re-rooted `VINTAGE_DATES_PATH`, `RELEASE_DATES_PATH`, `is_remote`, `storage_options_for`.

- [ ] **Step 1: Failing/guard test** — round-trip vintage_dates through a passed path, asserting
  `mkdir` is skipped for a remote-typed path and storage options are threaded. Local tmp_path
  round-trip must still pass.

- [ ] **Step 2: Run** → baseline.

- [ ] **Step 3: Implement** the standard treatment at each site:

```python
# vintage_dates.py
if not is_remote(VINTAGE_DATES_PATH):
    VINTAGE_DATES_PATH.parent.mkdir(parents=True, exist_ok=True)
df.write_parquet(VINTAGE_DATES_PATH, storage_options=storage_options_for(VINTAGE_DATES_PATH))

# releases.py reader
if not VINTAGE_DATES_PATH.exists():   # UPath.exists() works for s3
    ...
pl.read_parquet(VINTAGE_DATES_PATH, storage_options=storage_options_for(VINTAGE_DATES_PATH))

# __main__.py — same pattern for RELEASE_DATES_PATH and VINTAGE_DATES_PATH
```

  Note `tagger.py` also reads `VINTAGE_DATES_PATH` but is legacy/no-callers (memory
  `ces-live-capture-path`); update it for consistency but it is not on the production path.

- [ ] **Step 4: Run** `uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/ -k "release or vintage_dates" -v` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(release-dates): route vintage/release-dates I/O through storage_options + is_remote"`

---

### Task 6: seed the Tier-A S3 prefixes (one-time data move) — **DONE: `--apply` run 2026-06-20 (6 files → `s3://alt-nfp/{indicators,intermediate}`, md5-verified)**

**Files:**
- Create: `scripts/seed_data_s3.py` (a small, idempotent uploader; mirrors `mirror_store.py`)

This moves the *current local* Tier-A artifacts into S3 so the production nowcast can read them.
It is data movement, not app code — keep it a script. **Providers are excluded** — they live on a
separate Bloomberg store and are not ours to move (maintainer, 2026-06-20).

- [ ] **Step 1:** Write `scripts/seed_data_s3.py`: load `.env`; for each of
  `indicators/`, `competitors/`, `intermediate/vintage_dates.parquet`,
  `intermediate/release_dates.parquet` that exists under local `DATA_DIR`, `fs.put_file` to
  `s3://<bucket>/<relpath>` using the same `s3fs.S3FileSystem` setup as `mirror_store.py`.
  Do **not** upload `providers/`. Refuse to overwrite the **store** prefixes; print every upload.
- [ ] **Step 2:** Dry-run print (no `--apply`) → review the planned key list.
- [ ] **Step 3:** `uv run python scripts/seed_data_s3.py --apply` against MinIO; verify with a
  content-hash compare (reuse the audit's md5-index approach) that every uploaded object matches
  its local source.
- [ ] **Step 4: Commit** — `git commit -m "chore(scripts): seed Tier-A persistent artifacts into S3 (indicators/providers/competitors/schedules)"`

---

# Phase 2 — Tier C: rebuild scratch → tempfile

> **STATUS (2026-06-20):** Tasks **8 + 10 DONE** (commits `6e1b633`, `70acb48`).
> Tasks **7 + 9 SUPERSEDED → folded into plan 16** (`16-cli_production_workflow.md` Task 9.1,
> `scripts/bootstrap_store.py`). Reason: Task 7 threads one in-process `TemporaryDirectory`
> through `__main__.py`'s `download`/`process`/`build` — but those are separate `@app.command()`
> subcommands that run as separate processes for a real rebuild, so a single tempdir cannot span
> producer→consumer. Plan 16 Phase 9 **deletes** that legacy lineage and moves the bulk rebuild
> to a single-process script, which is the only place the run-scoped tempdir actually works.
> Task 9 (scraped-HTML temp) feeds `process`'s calendar build, which plan 16 also restructures
> and deletes. The Tier-C tempfile requirement is recorded in plan 16 Task 9.1's container-safety
> callout so it is not lost. (Maintainer decision, 2026-06-20.)

The rebuild pipeline writes raw downloads + revision intermediates to local `data/` and reads
them back in a later stage. To make it container-safe **without** persisting them (maintainer
decision), thread a single run-scoped `tempfile.TemporaryDirectory` through the rebuild stages
so producer and consumer share it, then let it auto-delete.

### Task 7: run-scoped temp root for the rebuild — **SUPERSEDED → plan 16 Task 9.1**

**Files:**
- Modify: `packages/nfp-vintages/src/nfp_vintages/__main__.py` (the `download`/`process`/`build`
  orchestration) — wrap a rebuild invocation in `with tempfile.TemporaryDirectory(prefix="altnfp-rebuild-") as tmp:`
  and pass `Path(tmp)` as the `data_dir` to download (`bls/bulk.py` already accepts `data_dir`)
  and as the intermediate root to processing/combine.
- Test: `packages/nfp-vintages/src/nfp_vintages/tests/test_vintages.py` (a smoke test that the
  rebuild orchestration accepts an injected temp root and writes nothing under `DATA_DIR`).

- [ ] **Step 1:** Make `processing/{ces_triangular,qcew_bulk,combine,sae_states}.py` take their
  input/output roots as **parameters** (default to the current `DOWNLOADS_DIR`/`INTERMEDIATE_DIR`
  constants for back-compat) instead of module-level `OUTPUT_PATH`/`CES_DIR`/`BULK_PATH` only.
  Keep the constants as defaults; add `out_dir`/`in_dir` params.
- [ ] **Step 2:** Failing test: call the rebuild orchestration with `NFP_BASE_DIR` pointed at a
  `tmp_path` and assert no files land under `tmp_path/data` after the run (everything in the
  injected TemporaryDirectory, which is gone on exit). Mock the network download.
- [ ] **Step 3:** Implement the `TemporaryDirectory` threading in `__main__.py`.
- [ ] **Step 4:** Run the vintages suite `-m "not network"` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "refactor(rebuild): thread a run-scoped tempdir through download→process→build"`

### Task 8: HTTP response cache → tempfile default — **DONE (commit `6e1b633`)**

> Implemented: `BLSHttpClient.cache_dir` default was `'.cache/bls'` (CWD-relative, not `data/` as
> drafted below) → now `None` ⇒ per-process `tempfile.mkdtemp(prefix='altnfp-httpcache-')`,
> `atexit`-cleaned; explicit `cache_dir=` still persists. Live path (`releases.py` `current` /
> `fetch_ces_national`). `TestCacheDirDefault` added.

**Files:** Modify `packages/nfp-download/src/nfp_download/bls/_http.py` (`cache_dir`, lines ~144, ~360, ~410).

- [ ] **Step 1:** Default `cache_dir` to `None` → a per-process `tempfile.mkdtemp(prefix="altnfp-httpcache-")`
  (created lazily, registered with `atexit`/`weakref.finalize` for cleanup), instead of a `data/`
  path. Keep an explicit `cache_dir=` override for callers that want persistence.
- [ ] **Step 2:** Test: constructing the client without `cache_dir` writes its cache under a
  temp dir, not under `DATA_DIR`.
- [ ] **Step 3:** Implement + cleanup hook.
- [ ] **Step 4:** Run download unit tests `-m "not network"` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "refactor(download): default HTTP cache to tempfile, not data/"`

### Task 9: scraped release HTML → temp (inside Task 7's run root) — **SUPERSEDED → plan 16 Task 9.1**

**Files:** Modify `packages/nfp-download/src/nfp_download/release_dates/scraper.py:233-242`.

- [ ] **Step 1:** Make the scraper's `out_dir` a required/injected param fed by Task 7's temp
  root (default keeps `RELEASES_DIR` for ad-hoc local use). The parsed `release_dates.parquet`
  it feeds is already persisted to S3 (Task 5); the raw HTML pages are scratch.
- [ ] **Step 2:** Test: scraping with an injected `out_dir=tmp_path` writes HTML there and nothing
  under `DATA_DIR`.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5: Commit** — `git commit -m "refactor(scraper): write raw release HTML to an injected temp dir"`

### Task 10: SAE checkpoint → tempfile — **DONE (commit `70acb48`)**

> Implemented: SAE is disabled (`combine.py` no longer reads `sae_revisions.parquet`), so both
> `CHECKPOINT_PATH` and `OUTPUT_PATH` (were under `INTERMEDIATE_DIR`) now resolve under a stable
> `/tmp/altnfp-sae` subdir; added the missing parent `mkdir` before the checkpoint write. A stable
> temp path (not per-process `mkdtemp`) keeps within-run checkpoint resume and would survive across
> same-machine processes if SAE were re-enabled. `test_sae_states.py` added.

**Files:** Modify `packages/nfp-vintages/src/nfp_vintages/processing/sae_states.py` (`CHECKPOINT_PATH`).
SAE is currently disabled (per CLAUDE.md), so this is low-risk hygiene.

- [ ] **Step 1:** Route `CHECKPOINT_PATH` through the injected temp root (Task 7) or a
  `tempfile` default; guard `mkdir`.
- [ ] **Step 2:** Test (or note SAE-disabled and cover by inspection).
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5: Commit** — `git commit -m "refactor(sae): checkpoint to tempfile (SAE disabled; hygiene)"`

---

# Phase 3 — Dev scripts + docs + verification

> **STATUS (2026-06-20):** Mostly DONE. `.env.example` + root/package CLAUDE.md container-contract
> docs + `_t8_promote.py` banner committed (`55a20b5`, `9c74e6e`). **`_05_convergence_fit.py` left
> AS-IS** (out of scope: local do-not-commit scratch, never imported/shipped; its `BASELINE_DIR`
> output is a PERSISTENT reference baseline, so the planned tempfile default would delete it — Bloomberg
> escape hatch is a one-line `s3://` at point of use). **Task 6 seed + Task 12 MinIO verify DONE**
> (2026-06-20): 6 Tier-A files seeded to `s3://alt-nfp/{indicators,intermediate}` (md5-verified);
> `build_model_data(as_of=2024-12-12)` read indicators+schedules+store from S3 (T=95) with zero `./data`
> writes. The verify caught a real bug — polars can't consume a UPath **object** for `s3://` ("Object does
> not have a .read() method"); fixed by `str()`-ing the path at 12 Tier-A I/O sites (commit `85fcc7b`,
> matches the store's `str(...)` pattern). `NFP_PROVIDERS_URI` skipped per maintainer (skip-3).

### Task 11: fix the two `data/`-hardcoders; document the script contract — **`_t8` DONE (`55a20b5`); `_05` left as-is (see Phase 3 status)**

**Files:**
- Modify: `scripts/_05_convergence_fit.py:61` (`BASELINE_DIR`)
- Modify: `scripts/_t8_promote.py:35` (`LOCAL_BACKUP`) — one-time tool already used; make its
  local backup dir a `tempfile`/argv path or leave with a clear "local-only, not for Bloomberg"
  banner.

- [ ] **Step 1:** `_05_convergence_fit.py`: take the baseline output root from `sys.argv`/env
  (default `tempfile.mkdtemp` or an `s3://` URI) instead of the hardcoded
  `"data/05_convergence_baseline"`; if an `s3://` target, write the `.npz`/json via UPath.
- [ ] **Step 2:** The argv-driven scripts (`run_a4/a5_backtest`, `run_a3_parity`,
  `run_tier1_diagnostics`, `generate_*_golden`, `regen/stage_golden`) need **no code change** —
  document in each module docstring (and `CLAUDE.md`) that the output root must be a `/tmp`
  path or an `s3://` URI on Bloomberg, never `data/`.
- [ ] **Step 3: Commit** — `git commit -m "refactor(scripts): de-hardcode data/ in _05_convergence_fit; document /tmp|s3 output contract"`

### Task 12: docs, `.env.example`, and MinIO end-to-end verification — **DONE (`55a20b5`,`9c74e6e`; seed + MinIO verify 2026-06-20; UPath→str fix `85fcc7b`)**

**Files:**
- Modify: root `CLAUDE.md` "Hard rules" (add `NFP_DATA_URI`), `packages/*/CLAUDE.md` data-layout
  sections.
- Create: `.env.example` (no such file today — only the gitignored `.env`). Document
  `NFP_STORE_URI`, `NFP_SNAPSHOTS_URI`, `NFP_DATA_URI=s3://alt-nfp`, `NFP_PROVIDERS_URI`
  (separate provider store), and the `AWS_*` keys with placeholder values.

- [ ] **Step 1:** Document the four URIs (`NFP_STORE_URI`, `NFP_SNAPSHOTS_URI`, `NFP_DATA_URI`,
  `NFP_PROVIDERS_URI`) as the container contract; note Tier-C is tempfile and dev scripts take a
  caller-supplied root.
- [ ] **Step 2: Manual MinIO verification (network):** with `.env` pointing at MinIO and
  `NFP_DATA_URI=s3://alt-nfp` set, run a full `build_model_data(as_of=…)` and confirm it reads
  indicators/providers/consensus/vintage_dates **from S3** and the run writes **nothing** under
  `./data/` (e.g. `find data -newermt '-5 min'` is empty). Capture the output.
- [ ] **Step 3:** Full local suite `uv run pytest -m "not network" --no-cov` + `uv run ruff check .` → green.
- [ ] **Step 4: Commit** — `git commit -m "docs: NFP_DATA_URI container contract; verify build_model_data reads Tier-A from S3"`

---

## Self-review (run before declaring done)

1. **Coverage:** every Tier-A artifact (indicators, providers, competitors, vintage_dates,
   release_dates) has a write/read task routing it through `data_location()` + `storage_options_for`
   + `is_remote` (Tasks 2–5) and a seed task (6). Every Tier-C artifact (downloads, revisions,
   HTTP cache, scraped HTML, SAE checkpoint) has a tempfile task (7–10). Both `data/`-hardcoding
   scripts are fixed (11). Store + snapshots untouched (constraint).
2. **No store regressions:** grep the diff — `NFP_STORE_URI`, `is_canonical_store`, `build_store`,
   `rebuild_store`, `VINTAGE_STORE_PATH` must be unchanged.
3. **Tests are network-free:** all new tests use `monkeypatch`/`tmp_path`; the only network step
   is the manual MinIO verification (Task 12).
4. **DRY:** the credentialed-`UPath` constructor exists once (`paths._upath`) and is reused.

## Open items for maintainer sign-off

- **`NFP_DATA_URI` bucket/prefix:** plan assumes `s3://alt-nfp` (objects at
  `s3://alt-nfp/{indicators,competitors,intermediate}/...`), matching the audit's observed
  layout. Confirm, or specify a different prefix.
- **`NFP_PROVIDERS_URI` value/layout:** providers are on a separate Bloomberg store (resolved
  2026-06-20). Need its URI and whether `ProviderConfig.file` (`providers/g/g_provider.parquet`)
  matches that store's layout or needs a path remap. Not blocking Phase 1 code (default stays
  local); needed before the Bloomberg run.
