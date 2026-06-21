# Audit Remediation Implementation Plan

> **Status: ✅ SUBSTANTIALLY IMPLEMENTED — moved to `plans/completed/` 2026-06-21.**
> Phases 1–2 landed in code (the `- [ ]` checkboxes below were never ticked, but the
> commits exist): `H-2a` credential-free CI fixtures (`780b5a4`), `H-3` dead-BD-array
> drop + snapshot `SCHEMA_VERSION→3` (`428b84c`), `H-4a` all-zero-indicator warning
> (`baef57c`), `H-4b` calendar under-cardinality fail-loud (`3521800`), `T5`/`L-13`
> retry docstring (`7d439c0`), `T10–T14`. **Only the Phase-3 model-input items remain** —
> the `H-4a` NaN sentinel and the `H-2b` full-model parity-in-CI smoke (recorded under
> `T15`, `2a05c7c`) — **explicitly deferred to the next golden-master / MCMC run**, a
> port-time activity (same build-here / validate-on-port posture as A5). Not "stale":
> implemented with a port-deferred verification tail.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the hazards documented in `specs/audit_alt_nfp.md` and
`specs/audit_nfp_ingest.md` from prose into code — guards, assertions, tests,
and version contracts — without breaking the Phase-A parity baseline.

**Architecture:** Three phases ordered by *parity risk and verifiability*, not
severity. Phase 1 touches no model inputs and is fully CI-verifiable now.
Phase 2 changes selection/precondition logic that is statically verifiable
against the frozen reference at `~/Projects/alt_nfp` (read-only). Phase 3
changes what reaches the model; its parity impact lives in S3-gated golden
masters / posteriors, so implementation lands now but parity *verification* is
deferred to the next golden-master/MCMC run (same posture as A5).

**Tech Stack:** Python 3.12, Polars, NumPyro/JAX (model only), pytest, ruff.
uv workspace (`nfp-lookups → nfp-download → nfp-ingest → nfp-vintages`;
`nfp-model` apart).

---

## PARITY POLICY (binding on every Phase 2 / Phase 3 task)

When a fix would change selection logic or model inputs, the implementer MUST
first compare against the **frozen reference** at `~/Projects/alt_nfp`
(underscore). **`~/Projects/alt_nfp` is READ-ONLY — never modify it.**
Classify the finding:

- **(a) parity-neutral** — reference does the same thing AND the change does not
  move the posterior (e.g. a code path that never fires on real data, or an
  array the model never reads). → Fix freely.
- **(b) port-bug** — v2 diverged from the reference. → Fix to match the
  reference; this *restores* parity. Note the divergence in the commit.
- **(c) correct-but-divergent** — reference exhibits the same behavior AND
  fixing it would move the posterior / change a golden master. → **STOP. Do not
  change behavior.** Report the reference comparison + a recommendation to the
  controller, who surfaces it to the user for a per-item ruling. Fix only the
  docstring/invariant to tell the truth.

"Parity" means **posterior/array parity**, not `content_hash` parity. Dropping
an array the model never reads (H-3) changes the hash but is posterior-neutral —
that is case (a), with an operational (snapshot re-bake) cost, not a baseline
divergence.

A green fast-suite is **not** evidence of parity — the fast suite cannot see the
golden masters (that gap is finding H-2). Never mark a Phase 3 task "parity-
verified" on a fast-suite pass alone.

---

## STORE-WRITE SAFETY (binding on every store-touching task)

*Added after an incident: the C-1 guard's own red-phase test ran `build_store`
against the real canonical MinIO (live `.env` creds via conftest) and wiped a
year of irreplaceable CES live-captures. Recovered from the reference store.*

- **No test may call a store-WRITING function** (`build_store`,
  `append_to_vintage_store`, `compact_partition`, `mirror_store`) against
  anything but a `tmp_path`. Never the canonical URI, never a real remote URI.
- **Strip store creds when running store-touching tests.** Dispatch instructions
  and/or an autouse fixture must blank `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL`, `NFP_STORE_URI` so accidental S3
  I/O cannot reach the live MinIO. Read-only golden-master tests that *need*
  creds are explicitly opted-in, never the default.
- **A guard test must fail closed.** Testing that a destructive op is *refused*
  must not, in any phase (including TDD red), be able to execute that op against
  real infrastructure. Construct the URI / blank creds so the worst case is a
  harmless local failure.
- Every subagent dispatched onto store code is told `~/Projects/alt_nfp` is the
  read-only recovery source and the canonical MinIO at `127.0.0.1:9000` is
  append-only / never a test target.

---

# PHASE 1 — Safety rails + hygiene (no model inputs; verify fully now)

### Task 1: Guard `build_store` against the canonical store (C-1)

**Files:**
- Modify: `packages/nfp-vintages/src/nfp_vintages/build_store.py`
- Modify: `packages/nfp-vintages/src/nfp_vintages/__main__.py` (`build` command)
- Test: `packages/nfp-vintages/tests/test_build_store_guard.py` (create)
- Docs: `packages/nfp-vintages/CLAUDE.md` (bare-`alt-nfp` note, L-9)

**Context.** `build_store` defaults `out_path = store_path or VINTAGE_STORE_PATH`
(the canonical store when `NFP_STORE_URI=s3://alt-nfp/store`) and wipes each
partition (`f.unlink()` loop) before rewriting. The CLI `build` passes no
`store_path`, and bare `alt-nfp` runs `build(None)`. So one bare run against the
documented default config silently destroys live-captured rows that exist in no
raw input. The CLAUDE.md hard rule "Never rebuild the canonical store in place"
is currently enforced only by a comment. The local fallback (`data/store/`,
`is_remote` False) must stay rebuildable (CI relies on it).

- [ ] **Step 1: Write the failing test**

```python
# packages/nfp-vintages/tests/test_build_store_guard.py
import pytest
from upath import UPath
from nfp_vintages.build_store import build_store


def test_refuses_canonical_remote_store_without_optin():
    # s3://alt-nfp/store is the canonical URI; guard must fire BEFORE any I/O.
    with pytest.raises(RuntimeError, match="canonical"):
        build_store(store_path=UPath("s3://alt-nfp/store"))


def test_allows_scratch_rebuild_prefix(monkeypatch, tmp_path):
    # A scratch prefix (…/store-rebuild) is not the canonical store: guard passes.
    # Point revisions at a missing file so we fail AFTER the guard, proving the
    # guard let us through (FileNotFoundError, not RuntimeError 'canonical').
    with pytest.raises((FileNotFoundError, Exception)) as exc:
        build_store(
            revisions_path=tmp_path / "nope.parquet",
            store_path=UPath("s3://alt-nfp/store-rebuild"),
        )
    assert "canonical" not in str(exc.value).lower()


def test_allows_canonical_with_explicit_optin(tmp_path):
    with pytest.raises((FileNotFoundError, Exception)) as exc:
        build_store(
            revisions_path=tmp_path / "nope.parquet",
            store_path=UPath("s3://alt-nfp/store"),
            allow_canonical=True,
        )
    assert "canonical" not in str(exc.value).lower()


def test_local_store_never_guarded(tmp_path):
    # Local path: is_remote False, guard never fires even for a '/store' suffix.
    with pytest.raises((FileNotFoundError, Exception)) as exc:
        build_store(revisions_path=tmp_path / "nope.parquet", store_path=tmp_path / "store")
    assert "canonical" not in str(exc.value).lower()
```

- [ ] **Step 2: Run the test, verify it fails** — `uv run pytest packages/nfp-vintages/tests/test_build_store_guard.py -v` → FAIL (no `allow_canonical` kwarg / no guard).

- [ ] **Step 3: Add the guard.** In `build_store`, add `allow_canonical: bool = False` to the signature; immediately after `out_path = store_path or VINTAGE_STORE_PATH` (before any read):

```python
from nfp_lookups.paths import is_remote  # already imported in this module

if is_remote(out_path) and str(out_path).rstrip("/").endswith("/store") and not allow_canonical:
    raise RuntimeError(
        "refusing to rebuild the canonical store in place "
        f"({out_path}); write to a scratch prefix (e.g. .../store-rebuild) "
        "or pass allow_canonical=True. See CLAUDE.md 'Never rebuild the "
        "canonical store in place'."
    )
```

- [ ] **Step 4: Thread an explicit CLI opt-in.** In `__main__.py::build`, add
  `--allow-canonical` (default `False`, help text quoting the hazard) and pass
  it through to `build_store`. The bare `alt-nfp` callback keeps calling
  `build(None)` and does **not** opt in, so bare `alt-nfp` against the canonical
  URI now fails loud instead of destroying data.

- [ ] **Step 5: Run the tests, verify they pass** — all four green.

- [ ] **Step 6: Fix doc drift (L-9).** In `packages/nfp-vintages/CLAUDE.md`,
  correct the bare-`alt-nfp` description to include `build`, and note the new
  canonical-store guard + the `--allow-canonical` opt-in.

- [ ] **Step 7: Commit** — `git add -A && git commit` (message: `fix(vintages): guard build_store against canonical-store in-place rebuild (C-1)`).

---

### Task 2: Collision-proof `append_to_vintage_store` filename (I-1)

**Files:**
- Modify: `packages/nfp-ingest/src/nfp_ingest/vintage_store.py` (`append_to_vintage_store`, ~line 712)
- Test: `packages/nfp-ingest/tests/test_vintage_store.py` (append/compact write-path class)

**Context.** The append filename is `f"v_{vmin}_{vmax}.parquet"` — derived only
from the batch's vintage-date range. `write_parquet` replaces the target. The
anti-join dedup guarantees the *rows* are new but NOT that the *filename* is
unique. Two multi-vintage appends spanning the same `[min, max]` but otherwise
disjoint produce the same filename; the second silently overwrites the first.
The monthly single-vintage live-capture path is safe (`vmin == vmax`); the
dangerous pattern is bulk/backfill. This is the one write path the architecture
promises is append-safe (it's the mitigation behind C-1), so it must not lose
data.

- [ ] **Step 1: Write the failing test** (`tmp_path`, no S3):

```python
def test_append_two_disjoint_multivintage_batches_conserves_rows(tmp_path):
    import polars as pl
    from datetime import date
    from nfp_ingest.vintage_store import append_to_vintage_store

    def _batch(emps, refs, vints):
        return pl.DataFrame({
            "geographic_type": ["national"] * len(emps),
            "geographic_code": ["00"] * len(emps),
            "industry_type": ["national"] * len(emps),
            "industry_code": ["00"] * len(emps),
            "ref_date": refs,
            "vintage_date": vints,
            "revision": pl.Series([0] * len(emps), dtype=pl.UInt8),
            "benchmark_revision": pl.Series([0] * len(emps), dtype=pl.UInt8),
            "employment": emps,
            "source": ["ces"] * len(emps),
            "seasonally_adjusted": [True] * len(emps),
        })

    # Batch A and B both span vintages [Jan, Mar] but are otherwise disjoint
    # (different ref_dates → distinct uniqueness keys → both survive anti-join).
    a = _batch([100.0, 101.0], [date(2024, 1, 1), date(2024, 2, 1)], [date(2024, 2, 1), date(2024, 4, 1)])
    b = _batch([200.0, 201.0], [date(2024, 3, 1), date(2024, 4, 1)], [date(2024, 2, 1), date(2024, 4, 1)])

    assert append_to_vintage_store(a, store_path=tmp_path) == 2
    assert append_to_vintage_store(b, store_path=tmp_path) == 2

    part = tmp_path / "source=ces" / "seasonally_adjusted=true"
    got = pl.read_parquet(str(part / "*.parquet"))
    assert got.height == 4  # FAILS on old code: B clobbers A → 2
```

- [ ] **Step 2: Run it, verify it fails** — height == 2 (clobbered).

- [ ] **Step 3: Make the filename content-addressed.** Add a module-level helper
  and use it for the stem suffix:

```python
import hashlib

def _content_suffix(df: pl.DataFrame) -> str:
    """Short, order-insensitive content hash of a partition write batch."""
    h = df.hash_rows().sort()  # Series[UInt64], order-insensitive after sort
    return hashlib.sha1(h.to_numpy().tobytes()).hexdigest()[:12]
```

  Replace the filename construction:

```python
vmin = partition_df["vintage_date"].min()
vmax = partition_df["vintage_date"].max()
fname = f"v_{vmin}_{vmax}_{_content_suffix(partition_df)}.parquet"
```

  Idempotency note: re-appending an identical surviving batch yields the same
  suffix → same filename → harmless overwrite with byte-identical content.
  `compact_partition` already merges fragments, so extra files are harmless.

- [ ] **Step 4: Run it, verify it passes** — height == 4. Run the full
  `test_vintage_store.py` to confirm no regression.

- [ ] **Step 5: Commit** — `fix(ingest): collision-proof append_to_vintage_store filename (I-1)`.

---

### Task 3: Align append/compact vintage_date tie-break (M-6)

**Files:**
- Modify: `packages/nfp-ingest/src/nfp_ingest/vintage_store.py` (`compact_partition`, ~line 775; uniqueness-key docstrings)
- Test: `packages/nfp-ingest/tests/test_vintage_store.py`

**Context.** Both `append` and `compact` dedup on a key that *excludes*
`vintage_date`. `append`'s anti-join keeps the **existing** (first-written) row;
`compact` sorts `vintage_date` **descending** and keeps first → keeps
`max(vintage_date)` (last). They resolve a collision *oppositely*. A vintage
store records *first observation*, so the rule should be **earliest
`vintage_date` wins**, matching append. In practice append prevents the second
write so collisions are rare, but the rule must be explicit and identical.

- [ ] **Step 1: Write the failing test** (`tmp_path`):

```python
def test_compact_keeps_earliest_vintage_on_key_collision(tmp_path):
    import polars as pl
    from datetime import date
    from nfp_ingest.vintage_store import compact_partition
    part = tmp_path / "source=ces" / "seasonally_adjusted=true"
    part.mkdir(parents=True)

    def _row(emp, vint):
        return pl.DataFrame({
            "geographic_type": ["national"], "geographic_code": ["00"],
            "industry_type": ["national"], "industry_code": ["00"],
            "ref_date": [date(2024, 1, 1)], "vintage_date": [vint],
            "revision": pl.Series([0], dtype=pl.UInt8),
            "benchmark_revision": pl.Series([0], dtype=pl.UInt8),
            "employment": [emp],
        })
    # Same uniqueness key, two vintage_dates, in two fragments.
    _row(100.0, date(2024, 2, 1)).write_parquet(str(part / "v_a.parquet"))
    _row(999.0, date(2024, 5, 1)).write_parquet(str(part / "v_b.parquet"))

    compact_partition(tmp_path, "ces", True)
    got = pl.read_parquet(str(part / "*.parquet"))
    assert got.height == 1
    assert got["employment"][0] == 100.0          # earliest vintage wins
    assert got["vintage_date"][0] == date(2024, 2, 1)
```

- [ ] **Step 2: Run it, verify it fails** — old code keeps 999.0 (max vintage).

- [ ] **Step 3: Flip the sort to ascending** in `compact_partition`:

```python
.sort("vintage_date", descending=False)   # earliest vintage wins (matches append)
.unique(subset=ukey, keep="first")
```

  Add a one-line comment on **both** `ukey` definitions (append + compact):
  `# vintage_date excluded from key; earliest-observed vintage wins on collision`.

- [ ] **Step 4: Run it, verify it passes.** Full `test_vintage_store.py` green.

- [ ] **Step 5: Commit** — `fix(ingest): align compact_partition tie-break with append (earliest vintage wins) (M-6)`.

---

### Task 4: Fail-loud on calendar parse under-cardinality (H-4b)

**Files:**
- Modify: `packages/nfp-download/src/nfp_download/release_dates/scraper.py` (`parse_index_page`)
- Test: `packages/nfp-download/tests/` (release-dates test module)

**Context.** `parse_index_page` parses BLS HTML heuristically. A page-structure
drift makes it return an empty/partial list with **no error**, feeding wrong
vintage dates into the censoring layer. The *fetch* path already degrades
gracefully (catches `FetchError`, falls back to cached pages); the *parse* path
has no sanity check. Read the function first to match its signature/return type.

- [ ] **Step 1: Write the failing test.** Feed `parse_index_page` HTML that is
  well-formed but contains zero matching release anchors; assert it raises a
  distinct error (e.g. `ParseError` / `ValueError`) mentioning the publication
  name, rather than returning `[]`. Add a positive test that valid HTML with
  ≥ expected entries still parses cleanly.

- [ ] **Step 2: Run it, verify it fails** (currently returns `[]`).

- [ ] **Step 3: Implement.** After parsing, if `len(entries)` is below a small
  floor (e.g. `< 1` for any page, or a per-publication minimum if one is
  clearly justified by the HTML structure), raise with the publication name and
  the URL/context. Keep the raised type catchable by the existing
  `__main__.py` handling if appropriate, OR a new type the caller logs as a
  hard warning — match the existing degradation contract (the caller already
  prints WARNING and continues on `FetchError`). Do **not** silently continue
  with a partial list.

- [ ] **Step 4: Run it, verify it passes.** Confirm `test_release_dates.py` green.

- [ ] **Step 5: Commit** — `fix(download): fail loud on calendar parse under-cardinality (H-4b)`.

---

### Task 5: Retry transport exceptions in `get_with_retry` (M-5)

**Files:**
- Modify: `packages/nfp-download/src/nfp_download/client.py` (`get_with_retry`)
- Test: `packages/nfp-download/tests/` (client test; mock transport)

**Context.** `get_with_retry` retries only on HTTP status (429 / ≥500). A
`ConnectTimeout` / `RequestException` / reset from `client.get` propagates
immediately — despite this being the retry helper. Read the function and the
imports first (both httpx and curl_cffi transport exception types are in play).

- [ ] **Step 1: Write the failing test.** Patch the underlying client so the
  first call raises a transport exception (e.g. `httpx.ConnectTimeout`) and the
  second returns a 200; assert `get_with_retry` returns the 200 after a retry.
  Also assert it still gives up after `max_retries` transport failures (raises).

- [ ] **Step 2: Run it, verify it fails** (exception propagates on first call).

- [ ] **Step 3: Implement.** Wrap `client.get(...)` in `try/except` over the
  httpx + curl_cffi transport exception types; on those, back off
  (`time.sleep(min(2**attempt, 120))`) and `continue`, capped by the same
  `max_retries`; re-raise the last exception if attempts are exhausted. Keep the
  existing status-code branch unchanged. Use `time.sleep` patchable in the test
  so it runs fast.

- [ ] **Step 4: Run it, verify it passes.**

- [ ] **Step 5: Commit** — `fix(download): retry transport exceptions in get_with_retry (M-5)`.

---

### Task 6: AST-based import-boundary test (M-7)

**Files:**
- Modify: `packages/nfp-model/tests/test_model_unit.py` (`TestBoundary`)

**Context.** `TestBoundary` scans stripped source lines for `import nfp_` /
`from nfp_` (exempting `nfp_model`). It misses dynamic imports
(`importlib.import_module('nfp_ingest')`) and `import numpy, nfp_ingest` lines.
This test is the *only* runtime guardrail for the "nfp-model imports no
`nfp_*`" invariant. Read the current test first.

- [ ] **Step 1: Write the failing test** (a self-test of the checker): construct
  a small source string containing `import_module("nfp_ingest")` and assert the
  boundary checker flags it. (If reworking in place, add a temp fixture file or
  parametrize the checker over source strings.)

- [ ] **Step 2: Run it, verify it fails** (line scan misses the dynamic import).

- [ ] **Step 3: Implement.** Replace the line scan with `ast.parse` + `ast.walk`
  over every `.py` under `nfp_model`: flag `ast.Import` / `ast.ImportFrom`
  whose module starts with `nfp_` and is not `nfp_model`; additionally flag any
  `ast.Call` to `import_module`/`__import__` with a constant string arg starting
  with `nfp_`. Keep the test's existing pass criteria (the real package must
  still pass).

- [ ] **Step 4: Run it, verify it passes** against the real `nfp_model` source.

- [ ] **Step 5: Commit** — `test(model): AST-based import-boundary check (M-7)`.

---

### Task 7: Reconcile the version + dependency contract (M-8)

**Files:**
- Modify: `packages/nfp-lookups/pyproject.toml`, `packages/nfp-download/pyproject.toml`,
  `packages/nfp-ingest/pyproject.toml`, `packages/nfp-vintages/pyproject.toml`
  (`requires-python`, numpy bound), and `packages/nfp-model/pyproject.toml` (numpy bound)
- Modify: any package `CLAUDE.md` that says "requires >= 3.10"

**Context.** Root + `nfp-model` declare `>=3.12`; the four data packages declare
`>=3.10`; the lockfile and prose say 3.12 throughout. The 3.10 floors are
untested fiction. `numpy>=1.24.0,<2.4` in lookups/ingest vs unbounded in
nfp-model — papered over by the single lockfile.

- [ ] **Step 1: Set `requires-python = ">=3.12"`** in all four data packages.
- [ ] **Step 2: Align the numpy bound** — apply the same `<2.4` ceiling to
  `nfp-model` (or remove it everywhere consistently; prefer matching the
  existing `>=1.24.0,<2.4`).
- [ ] **Step 3: Update CLAUDE.md** "requires >= 3.10" → ">= 3.12" wherever it appears.
- [ ] **Step 4: `uv lock --check`** (or `uv sync`) to confirm the lock is still
  consistent; run `uv run pytest -m "not network and not slow" --no-cov` to
  confirm nothing broke.
- [ ] **Step 5: Commit** — `chore: pin all packages to py>=3.12, align numpy bound (M-8)`.

---

### Task 8: Mechanical hygiene cluster (I-3, L-10, L-11, L-12, L-13 + ingest low-cluster)

**Files (one commit per coherent sub-item; group in this task):**
- `packages/nfp-ingest/src/nfp_ingest/vintage_store.py` — **I-3**: confirm
  `_QCEW_MAX_REVISION` is unreferenced (grep the package + tests); if dead,
  delete it and the local redefinition in the test, OR wire the QCEW rank rules
  and the test to import it. Prefer deleting if truly unused. Also lift the
  `2017-01-12` literal in `_select_qcew_at_horizon` to a named module constant
  (e.g. `_QCEW_MONTHLY_START`).
- `packages/nfp-model/src/nfp_model/parity.py` — **L-11**: remove the dead
  `"lam_ces"` exclusion name. (Leave the `lam_<name>` vs `lambda_ces` naming
  as-is unless trivial — cosmetic only; note it in the commit.)
- root `pyproject.toml` — **L-12**: drop `dynamax` from `keywords`.
- root `pyproject.toml` `[tool.pytest.ini_options].addopts` — **L-10**: every
  documented invocation passes `--no-cov`, so the five `--cov` trees never
  produce coverage. Either remove the dead `--cov` addopts or document why they
  stay. Prefer removing (YAGNI); confirm CI/local commands unaffected.
- `packages/nfp-download/src/nfp_download/client.py` — **L-13**: give
  `BLSHttpClient` a `close()` and `__enter__`/`__exit__` (or document the
  intended lifetime). Read `_http.py` first to confirm the pool surface.
- `packages/nfp-ingest/src/nfp_ingest/model_data.py` — **industry-scope comment**:
  the inline comment "no fallback 05→00 so we stay private-only" contradicts the
  documented default `industry_code='00'` (total nonfarm). Reconcile the comment
  to reality (likely the comment is stale). **dead `999` branch**: the
  `when(revision_number == -1).then(999)` remap in `_qcew_series_with_meta` is
  filtered to `source == 'qcew'` where `revision_number == -1` never occurs —
  delete the dead branch. *(Both are comment/dead-code only — NO value change;
  if either turns out to affect output, STOP and treat under the Parity Policy.)*
- `packages/nfp-ingest/src/nfp_ingest/snapshots.py` — `snapshot_model_data`
  computes `content_hash` twice (filename + inside `save_snapshot`); pass the
  digest in to skip the recompute. **collect_snapshot `__` guard**: assert
  provider names contain no `__` (mirror the assertion `batch.py` already has),
  since the `f'{name}__g_pp'` keying depends on it.

- [ ] **Step 1:** For each sub-item, make the change; where a test exists, keep
  it green; where the change is a deletion, run the package's test module to
  confirm nothing referenced it.
- [ ] **Step 2:** `uv run ruff check .` and `uv run pytest -m "not network and not slow" --no-cov`.
- [ ] **Step 3:** Commit per coherent sub-item (small, reviewable diffs).

---

### Task 9: Credential-free data-transform fixtures so CI can see the censoring (H-2a / TestGap)

**Files:**
- Create: `packages/nfp-ingest/tests/fixtures/` (tiny synthetic vintage store + expected censored panel)
- Create: `packages/nfp-ingest/tests/test_transform_censoring_ci.py`

**Context.** Both `model_data.py` and the store write path lean on golden-master
/ hash-stability tests that are S3-gated, so the fast CI suite verifies the
selection *helpers* and the snapshot *codec* but not the full censored
transform. This task commits a small, credential-free fixture and a store-free
test exercising the real branches that golden masters otherwise guard — the
package-level instance of H-2. (NOTE: this is NOT the parity-vs-frozen-reference
gate; the full model parity-in-CI smoke is H-2b, deferred to Phase 3 notes.)

- [ ] **Step 1: Build a tiny synthetic store fixture** (hand-authored Polars
  frame matching `VINTAGE_STORE_SCHEMA`): a few national CES ref-months with
  rev-0/1/2 at `benchmark_revision=0`, plus one QCEW series, written to a
  `tmp_path`-style Hive layout under the fixtures dir.
- [ ] **Step 2: Write tests** over `transform_to_panel(lf, as_of_ref=D)` that
  assert the censored CES diagonal (1 rev-0, 1 rev-1, rest rev-2), consecutive
  ref_dates, and that `vintage_date > D` rows are excluded. Add a case that
  feeds `panel_to_model_data` a horizon-censored panel and checks the CES vintage
  remap sizing + a post-COVID QCEW boundary multiplier on a synthetic frame.
- [ ] **Step 3: Run** `uv run pytest packages/nfp-ingest/tests/test_transform_censoring_ci.py -v` → green, **with no store env set** (these must run in CI).
- [ ] **Step 4: Commit** — `test(ingest): credential-free censoring/transform fixtures for CI (H-2a)`.

---

# PHASE 2 — Statically parity-verifiable correctness (read frozen reference; no MCMC)

> Every task here follows the **PARITY POLICY** above. The implementer reads the
> corresponding logic in `~/Projects/alt_nfp` (READ-ONLY) and classifies (a)/(b)/(c).

### Task 10: CES fallback benchmark-row leak (I-2)

**Files:**
- Modify: `packages/nfp-ingest/src/nfp_ingest/vintage_store.py` (`_select_ces_at_horizon` fallback, ~lines 110-115)
- Test: `packages/nfp-ingest/tests/test_vintage_store.py` (`TestSelectCesAtHorizon`)

**Context.** The docstring says benchmark-revised rows `(revision=2,
benchmark_revision>0)` are *never* selected. But the fallback sorts
`["revision","benchmark_revision"]` descending and keeps first — so it *will*
pick a benchmark row when one is the most-revised option for an unmatched
ref_date. Not a temporal look-ahead (already `vintage_date ≤ D`), but a
model-specification leak: benchmark-quality CES reaches the model directly for
those months, which the design routes through QCEW only. The fallback rarely
fires on a normal triangular store, and the existing test uses a benchmark-free
triangle, so the benchmark-selection path is **untested**.

- [ ] **Step 1: Verify against the reference.** Find the CES horizon-selection
  fallback in `~/Projects/alt_nfp`. Does the reference also include benchmark
  rows in the fallback? Record the answer + the file:line.
- [ ] **Step 2: Classify & act.**
  - If **(b)** the reference excludes benchmark rows (v2 drifted): add
    `& (pl.col("benchmark_revision") == 0)` to the fallback candidate set, and
    add a test with a benchmark row present that asserts it is NOT selected.
    Restores parity.
  - If **(a)** the fallback provably never fires on real triangular data AND the
    reference matches: fix the docstring to state the actual fallback behavior;
    optionally add the `benchmark_revision == 0` filter only if it changes no
    golden master (confirm via the S3 golden run later — if uncertain, treat as (c)).
  - If **(c)** the reference includes benchmark rows AND a real month would be
    affected: **STOP**, report to controller for a user ruling; fix only the
    docstring.
- [ ] **Step 3: Test** the chosen behavior; run `TestSelectCesAtHorizon` green.
- [ ] **Step 4: Commit** with the classification in the message.

---

### Task 11: `panel_to_model_data` layer-1 precondition (I-4)

**Files:**
- Modify: `packages/nfp-ingest/src/nfp_ingest/model_data.py` (`panel_to_model_data` docstring + a guard/marker)
- Test: `packages/nfp-ingest/tests/` (new store-free unit, or extend Task 9's file)

**Context.** `build_model_data` is safe (runs `build_panel(as_of_ref=D)` then
`panel_to_model_data(as_of=D)`). But `panel_to_model_data` is also a public
entry point and on its own applies only `vintage_date ≤ as_of` +
`_ces_best_available`'s "highest available revision per month." On a raw
all-vintages panel, "highest available" for old months resolves to the
**benchmark-revised** value — the look-ahead the rank-based selection exists to
prevent. The docstring doesn't warn the input panel must already be
horizon-selected. The two entry points have different censoring semantics and
the weaker one looks like the convenience.

- [ ] **Step 1: Verify against the reference.** Read `_ces_best_available` (its
  body, not just the docstring) and the reference's equivalent. Does
  `_ces_best_available` already exclude `benchmark_revision > 0`? Does the
  reference route external callers exclusively through the full path? Record findings.
- [ ] **Step 2: Classify & act.**
  - If `_ces_best_available` already excludes benchmark rows → the risk is
    "different-but-both-safe"; **document the precondition loudly** in the
    `panel_to_model_data` docstring (must receive a `build_panel(as_of_ref=D)`
    output) and add a cheap, optional consistency check (e.g. warn if the panel
    has >1 row per `(series, ref_date)` pre-selection, indicating an
    un-horizon-selected panel).
  - If it does **not** exclude benchmark rows → this is a real look-ahead on the
    direct path: add a precondition guard or marker-column check, OR exclude
    benchmark rows in `_ces_best_available`. Treat any posterior-moving change
    under the Parity Policy (likely (a) on the `build_model_data` path since
    layer-1 already selected; confirm).
- [ ] **Step 3: Test** with a store-free synthetic raw panel: assert the
  documented precondition is enforced/warned. `build_model_data` path unaffected.
- [ ] **Step 4: Commit** with the classification in the message.

---

# PHASE 3 — Model-input changes (implement now; parity VERIFICATION deferred to golden-master/MCMC run)

> These move what reaches the model. Implement + unit-test the mechanics now,
> but **do not claim parity** on a fast-suite pass. Parity is confirmed only by
> the S3 golden masters / a posterior comparison — defer that run (like A5).

### Task 12: Pre-flight — snapshot-hash ripple check for H-3 (do BEFORE Task 13)

**Files:** read-only investigation + a short note appended to this plan.

**Context.** H-3 drops `birth_rate`/`bd_proxy`/`bd_qcew_lagged` from the snapshot
and bumps `SCHEMA_VERSION`. `content_hash` is over array bytes + meta, so the
bump changes every snapshot's hash. Any harness that *pins* a hash (A4 vmap
backtest, the A5 harness on its branch, committed snapshot manifests) would
fail to resolve the old hash.

- [ ] **Step 1:** Grep for hard-coded snapshot hashes / pinned digests across
  `scripts/`, `plans/`, tests, and the A4/A5 harnesses (note: A5 lives on branch
  `a5-real-competitors`). List every consumer that pins a hash.
- [ ] **Step 2:** Record in this plan (append under "Phase 3 notes") the blast
  radius and the re-bake step needed (`alt-nfp snapshot …` regeneration), and
  whether any committed golden manifest must be regenerated.
- [ ] **Step 3:** If any pin is load-bearing and cannot be regenerated this
  session (needs S3 / the deferred MCMC), say so explicitly and gate Task 13's
  schema bump behind that regeneration.

---

### Task 13: Drop dead BD covariate arrays + schema version bump (H-3)

**Files:**
- Modify: `packages/nfp-ingest/src/nfp_ingest/model_data.py` (stop computing/returning `birth_rate`, `bd_proxy`, `bd_qcew_lagged`)
- Modify: `packages/nfp-ingest/src/nfp_ingest/snapshots.py` (`GLOBAL_ARRAY_KEYS`, `SCHEMA_VERSION` bump + v-read fallback)
- Modify: `packages/nfp-model/src/nfp_model/data.py` (`from_snapshot` — stop rebuilding them)
- Test: `test_snapshots.py`, `nfp-model/tests/test_data.py`

**Context.** These three arrays are computed, serialized into every `.npz`, and
reconstructed on intake, but `model.py` reads none of them (`plans/5`:
`φ₁·X^birth` never wired; `plans/0`: `φ₂·BD^QCEW` pruned as indistinguishable
from zero). Posterior-neutral (case (a)) — the model never reads them — but they
imply a misleading contract. The codebase already does v1→v2 read fallbacks, so
a versioned drop is in-pattern.

- [ ] **Step 0: Gate on Task 12.** If Task 12 found a load-bearing un-regenerable
  pin, STOP and report; otherwise proceed.
- [ ] **Step 1: Verify posterior-neutrality.** Confirm in `model.py` (v2) AND the
  reference that none of the three arrays enters the likelihood. (Expected
  case (a).) If any IS read → STOP, Parity Policy.
- [ ] **Step 2: Write/adjust tests** — a snapshot built without the three arrays
  round-trips; `from_snapshot` rebuilds model inputs without them; an old-schema
  snapshot (with the arrays) still loads via the version fallback.
- [ ] **Step 3: Implement** — remove from `build_model_data` return, from
  `GLOBAL_ARRAY_KEYS`, and from `from_snapshot`; bump `SCHEMA_VERSION`; add the
  read fallback for prior-version snapshots (mirror the existing v1→v2 pattern).
- [ ] **Step 4: Run** the store-free snapshot/data tests green.
- [ ] **Step 5: DEFERRED VERIFICATION:** the golden-master / posterior parity
  confirmation requires the S3 store + a model run. Mark the task **implemented,
  parity-verification deferred** (note it in the final summary; do not archive
  the audit until confirmed).
- [ ] **Step 6: Commit** — `refactor(ingest,model): drop dead BD covariate arrays + bump SCHEMA_VERSION (H-3)`.

---

### Task 14: NaN sentinel for censored cyclicals + all-zero-load warning (H-4a)

**Files:**
- Modify: `packages/nfp-ingest/src/nfp_ingest/model_data.py` (`_load_cyclical_indicators`, the cyclical censor)
- Modify (only if the model side must distinguish NaN): `packages/nfp-model/src/nfp_model/data.py` / `model.py` gating — **Parity Policy applies**
- Test: store-free unit over the cyclical load/censor

**Context, two sub-issues:**
1. If indicator parquets are missing/mispathed, `_load_cyclical_indicators`
   yields all-zero arrays, the model's gating drops them, and `φ₃` is **silently
   never sampled** (`plans/5` records this footgun).
2. The censor sets `arr[i:] = 0.0`, colliding with a legitimately-zero centred
   indicator — "censored" and "present-but-zero" are indistinguishable. Births
   already use `np.nan` for this.

- [ ] **Step 1: Verify against the reference.** How does the reference represent
  a censored cyclical — `0.0` or NaN? How does the model gating treat each?
  Record. The *warning* (sub-issue 1) is a pure add — no value change, safe
  regardless. The *NaN sentinel* (sub-issue 2) changes the array the model sees:
  classify (a)/(b)/(c).
- [ ] **Step 2: Implement the safe part now (sub-issue 1):** when a configured
  cyclical covariate loads all-zero AND `as_of` post-dates its publication lag,
  `warnings.warn(...)` (or raise, matching the fail-loud posture) naming the
  indicator and the expected path. Unit-test it. This needs no parity decision.
- [ ] **Step 3: NaN sentinel (sub-issue 2):**
  - If the reference already uses NaN (case (b)) → switch the censor to NaN to
    match; ensure the model gating treats NaN as "absent" identically. Restores parity.
  - If the reference uses `0.0` (case (a)/(c)) → changing to NaN moves what the
    model sees. If the model's gating makes it posterior-neutral (masked
    identically whether 0.0 or NaN) → (a), proceed + note for golden confirmation.
    If it could move the posterior → **(c), STOP**, report for a user ruling;
    meanwhile keep `0.0` and only fix the docstring + add the warning.
- [ ] **Step 4: Run** store-free unit tests green.
- [ ] **Step 5: DEFERRED VERIFICATION** for any model-input change (Step 3) —
  same posture as Task 13.
- [ ] **Step 6: Commit** with classification in the message.

---

### Task 15: H-2b note — full model parity-in-CI (scoped, likely deferred)

**Files:** append a short scoped note to this plan (no code unless trivially achievable).

**Context.** The audit's H-2 ideal is a credential-free `fit_model(light) →
collect_parity_arrays` smoke vs a committed golden reduction. A *true*
parity-vs-frozen-reference golden must be produced by `~/Projects/alt_nfp`
(read-only) on a shared tiny fixture, and an MCMC even at `light` is a `slow`
test. Task 9 already closes the data-transform CI gap; the model-level smoke is
the remaining piece.

- [ ] **Step 1:** Decide feasibility this session: can a tiny ModelData fixture
  be run through BOTH the v2 model and the frozen reference without S3 and
  without a long MCMC? If yes and cheap → implement a `slow`-marked smoke that
  asserts v2 vs a committed reference reduction at generous tolerance.
- [ ] **Step 2:** If not cheap → write the scoped note (fixture shape, how to
  produce the reference golden, tolerance) and mark H-2b **deferred**. Do not
  fake a self-referential test (v2 vs v2 proves nothing about parity).

---

## Phase 3 — DEFERRED items (execute alongside the next golden-master / MCMC run)

*Status 2026-06-13: H-3 (T13) and H-4a's all-zero warning (T14) were implemented
and verified now (A2 goldens 9/9). The two items below change a model input or
need a frozen-reference MCMC, so they wait for the run the user deferred. Do them
together — both regenerate/extend the same golden surface.*

### Deferred-A — H-4a NaN sentinel for censored cyclicals
The censoring loop in `panel_to_model_data` writes `0.0` into the post-horizon
tail of each cyclical array (`model_data.py` ~line 466), which is
indistinguishable from a legitimately-zero centred value. Births already use
`np.nan`. To switch cyclicals to a NaN sentinel:
1. **Classify vs the frozen reference** (`~/Projects/alt_nfp`, READ-ONLY): does
   the reference censor cyclicals with `0.0` or `np.nan`? If `0.0` (likely),
   this is a v2 divergence → Parity Policy case (a)/(c).
2. **Model-side masking is required first.** `model.py`'s `phi_3·X_cycle` block
   must treat a NaN cyclical as *absent* (masked out of the likelihood), not
   propagate NaN. Without this, NaN breaks the model. This is a `nfp-model`
   change → re-run the A3 parity + the MCMC smoke.
3. **A2 goldens change** (the cyclical arrays' tail values change `0.0`→NaN), so
   the A2 golden fixtures must be regenerated (and the `_DROPPED`-style exclusion
   does NOT apply — these are real value changes). Confirm posterior parity via
   the A3/backtest run.
Why deferred: it's a model-input change whose parity verification is the MCMC run
itself. Do not land it on a green fast-suite alone.

### Deferred-B — H-2b model-level parity-in-CI smoke
Task 9 closed the *data-transform* CI gap (credential-free). The *model-level*
gap remains: a credential-free `fit_model(light) → collect_parity_arrays` smoke
vs a committed golden reduction. The golden must be produced by the **frozen
reference** (`~/Projects/alt_nfp`, READ-ONLY) on a shared tiny `ModelData`
fixture; an MCMC even at `light` is a `slow` test.
1. Build a tiny credential-free `ModelData` fixture (a few periods, 1–2
   providers, 1 cyclical) that both repos can consume.
2. Run it through the frozen reference's model to produce a golden posterior
   reduction (means/SDs of a few sites); commit fixture + golden.
3. Add a `slow`-marked test: v2 `fit_model(light)` → `collect_parity_arrays` →
   assert vs the golden at a generous (MCSE-aware) tolerance.
Do **not** fake a self-referential v2-vs-v2 test — it proves nothing about
parity. Why deferred: needs the reference MCMC to mint the golden.

---

## Sequencing & gates

1. **Phase 1 (Tasks 1–9)** runs start-to-finish via subagent-driven-development
   (implementer → spec review → code-quality review per task). Fully verifiable
   in CI now; no parity exposure. **Natural checkpoint after Phase 1.**
2. **Phase 2 (Tasks 10–11)** each begins with the frozen-reference comparison.
   Any **case (c)** pauses for a user ruling (Parity Policy). No MCMC needed.
3. **Phase 3 (Tasks 12–15)** implements model-input changes with unit tests, but
   **parity verification is deferred** to the next S3 golden-master / MCMC run
   (the run the user is already deferring). Task 12 (hash-ripple) gates Task 13.
4. **Final:** full code review of the branch; run
   `uv run pytest -m "not network" --no-cov` + `uv run ruff check .`; summarize
   what is verified vs deferred. Do **not** archive the audit specs until the
   deferred parity items are confirmed.

## Self-review (controller, against the two audit docs)

- C-1 ✓ T1 · H-2 ✓ T9 (data) + T15 (model, deferred) · H-3 ✓ T12–13 ·
  H-4a ✓ T14 · H-4b ✓ T4 · M-5 ✓ T5 · M-6 ✓ T3 · M-7 ✓ T6 · M-8 ✓ T7 ·
  L-9 ✓ T1 · L-10/11/12/13 ✓ T8.
- I-1 ✓ T2 · I-2 ✓ T10 · I-3 ✓ T8 · I-4 ✓ T11 · TestGap ✓ T9 · ingest
  low-cluster ✓ T8.
- Every Phase 2/3 task carries the reference-comparison step and the
  "`~/Projects/alt_nfp` is read-only" constraint inline. Parity Policy stated once,
  referenced by ID. No placeholders: each task has files, a test, and a commit.
