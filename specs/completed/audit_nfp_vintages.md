# Package Review — `nfp-vintages`

**Scope this turn (read first).** Project-knowledge retrieval is still
unavailable, so this reviews only the `nfp-vintages` source already in our
session. That is, fortunately, the package's highest-risk surface:

- **`build_store.py`** — complete (the merge/dedup/partition-write builder and
  its `main()`). This is where the Critical store-rebuild footgun lives.
- **`__main__.py`** — most of it (the Typer app, the bare-run callback,
  `download`, `download-indicators`, `current`, `build`, `snapshot`, and the
  head of `_build_release_calendar`). Missing the full bodies of `process` and
  `_build_release_calendar`.
- **`CLAUDE.md`** — complete (file map, patterns, data layout, test mapping).

**Deferred (not seen this turn — no findings asserted):** `views.py`
(`real_time_view`/`final_view`/`specific_vintage_view`), `evaluation.py`
(`vintage_diff`/`build_noise_multiplier_vector`), `processing/ces_triangular.py`,
`processing/qcew_bulk.py`, `processing/sae_states.py` (disabled),
`processing/combine.py`, `__init__.py`, `test_vintages.py`, and the unread
bodies in `__main__.py`. The `processing/` modules (CES triangular-CSV parsing,
the QCEW 4-stream transform with persons→thousands conversion and the
durable/nondurable + government-ownership hierarchy) are the most likely place
for *parsing* bugs — flag them for the next pass.

Findings new to this review are **[NEW]**; carried-and-sharpened ones are
**[SYS]**.

---

## Package role and one structural observation

`nfp-vintages` sits at the top of the data chain (it imports lookups, download,
and ingest) and is the only package with a `[project.scripts]` entry — it is the
*pipeline + CLI*. Acquisition was moved out to `nfp_download.bls.bulk` in the A2
seam, so this package is now orchestration (`__main__`), the store builder
(`build_store`), and pure-Polars query/eval helpers (`views`, `evaluation`).
Dependency direction is clean (all imports point downward; the
`_build_release_calendar` orchestration legitimately pulls from all three lower
packages).

**[NEW] D-1 (design, Medium) — the vintage store's *write* surface is split
across two packages with divergent safety policies.** The from-scratch rebuild
(`build_store`, here in nfp-vintages) and the incremental writers
(`append_to_vintage_store` / `compact_partition`, in nfp-ingest) are both "write
the vintage store," but they live in different packages and were evidently
maintained as separate concerns. The visible consequence is that the *same
logical uniqueness key* is deduplicated **three different ways**:

| Writer | Package | Key excludes `vintage_date`? | Tie-break on a key collision |
|---|---|---|---|
| `build_store` | nfp-vintages | yes | `current`-tier wins, then **arbitrary** (order-dependent) |
| `append_to_vintage_store` | nfp-ingest | yes | keep **existing** (first-seen) |
| `compact_partition` | nfp-ingest | yes | keep **max** `vintage_date` |

Three policies for one decision is how they drifted apart in the first place
(the system review's M-6 was two of them; here's the third). **Fix:** give the
store a single authoritative "write a partition with this dedup policy" function
in `nfp_ingest.vintage_store` (which already owns `VINTAGE_STORE_SCHEMA` and the
append/compact path), and have `build_store` call it. One key, one tie-break,
one place to reason about.

---

## `build_store.py`

Reads `revisions.parquet` + `releases.parquet`, merges (current releases win),
normalizes `industry_type`, dedups, and writes the Hive-partitioned store. Small
file, large blast radius.

### Correctness

**[SYS → restated as Critical for this package] B-1 — `build_store` silently and
irreversibly destroys the canonical store, and bare `alt-nfp` triggers it.** The
builder writes to the canonical store by default and clears each partition
before rewriting:

```python
out_path = store_path or VINTAGE_STORE_PATH          # default = canonical store
...
if partition_dir.exists():
    for f in partition_dir.glob('*.parquet'):
        f.unlink()                                    # wipe, then write one file
```

The intended store lifecycle is **build once, append forever**: the root
`CLAUDE.md` states "the canonical store only ever takes appends," and that
`s3://alt-nfp/store` holds live-captured release-day rows "that exist in no raw
input — a from-scratch `alt-nfp build` to that URI would silently destroy them."
So `build_store` is a one-time bootstrap that becomes a loaded gun the moment the
first monthly append lands. (Hedge: the exact reconstructibility depends on
whether `releases.parquet` accumulates captures or is overwritten each `current`
run — that's in `nfp_ingest.releases`, which I haven't read — but the
maintainer's own root-CLAUDE.md assessment that rebuild "would silently destroy
them" is the authoritative signal, and my finding holds regardless: the hazard
is documented and the code has no guard.)

The severity is **Critical for this package** because the trigger is the most
natural command: not just `alt-nfp build`, but **bare `alt-nfp`** (see V-1), run
against the documented default store, guarded only by a comment. "The author
remembers the rule" is not a code safeguard, and the loss is silent and
irreversible.

**Fix (pick one; the second is cleanest):**
```python
# (a) refuse the canonical store unless explicitly forced
if is_remote(out_path) and str(out_path).rstrip('/').endswith('/store') and not allow_canonical:
    raise RuntimeError(
        'refusing to rebuild the canonical store in place — it holds '
        'irreproducible live-captured rows; write to a scratch prefix '
        '(…/store-rebuild) or pass allow_canonical=True'
    )
# (b) better: route build_store through the append+dedup writer (D-1) so a
#     "rebuild" merges into, rather than replaces, the existing partitions.
```

**[NEW] B-2 (Medium) — the dedup is order-sensitive and under-keyed.** The merge
resolves overlaps by:

```python
combined = (
    combined.sort('current')
    .unique(subset=key_cols, keep='last')   # key_cols excludes vintage_date & employment
    .drop('current')
)
```

Two issues. First, `keep='last'` keeps the last row *in current row order*;
after `sort('current')`, current=1 (releases) rows land last, so a release wins a
collision with a revision — that's the intent. But for two rows that share the
key **and** the `current` tier (differing only in `vintage_date`/`employment`),
which survives depends on whether Polars' `sort` is stable for ties on `current`
— engine/version-dependent, and `maintain_order` is not set. That makes the
store contents potentially **nondeterministic** for same-tier duplicates, which
would silently break the "byte-identical reproduction" property the A0 gate
relied on. Second, because `key_cols` excludes `vintage_date`, a release row
*replaces* a revision row's `vintage_date` for the same `(…, revision,
benchmark_revision)` — defensible (the release is the authoritative real-time
vintage) but undocumented, and it's the same vintage_date-tie-break ambiguity as
D-1. **Fix:** sort by a deterministic tiebreaker before `unique`
(`['current', 'vintage_date']` or similar), or use an explicit
"prefer current, then latest vintage" expression, and document the rule.

**[NEW] B-3 (Medium) — the combined frame is written without schema validation.**
`pl.concat([revisions, releases], how='diagonal_relaxed')` fills missing columns
with null and relaxes dtype mismatches, then the frame is normalized, deduped,
and written — with **no cast to `VINTAGE_STORE_SCHEMA` and no column check**. A
malformed `releases.parquet` (an extra column, a coerced dtype) flows straight
into the canonical store. `read_vintage_store` then scans it with an *explicit*
`schema=VINTAGE_STORE_SCHEMA`, so a write/read schema disagreement surfaces later
as a read error or silent column injection rather than at the (much cheaper)
write. The store is the foundational artifact; the write side should be at least
as strict as the read side. **Fix:** `combined.cast(VINTAGE_STORE_SCHEMA)` (or a
`validate`-style guard) before the partition loop, and reject unexpected columns.

**[NEW] B-4 (Low) — reads ignore `storage_options`, writes don't.**
`pl.read_parquet(rev_path)` / `pl.read_parquet(rel_path)` are called with no
`storage_options`, while the writes correctly use `storage_options_for(out_path)`.
This is fine for the CLI (inputs are always the local `INTERMEDIATE_DIR`/
`DATA_DIR`), but if a caller passes a remote `releases_path`, the read silently
lacks credentials/options. Minor asymmetry; tighten if remote inputs ever become
a thing.

**[NEW] B-5 (Low–Medium) — `build_store.main()` is a redundant, weaker second
CLI for the same function.** It hand-parses `sys.argv`:

```python
if '--releases' in args:
    idx = args.index('--releases')
    if idx + 1 < len(args):
        releases_path = Path(args[idx + 1])
```

If `--releases` is the final arg (value forgotten), the guard is False and it
**silently** builds from the default location instead of erroring; all other
args are ignored. Meanwhile the Typer `build` command (`alt-nfp build
--releases PATH`) handles this properly. Two entry points to one function, the
`python -m` one being the inferior copy. **Fix:** drop `main()` in favor of the
Typer command, or at minimum error on a missing `--releases` value.

### Notes

- The shared `f'v_{vmin}_{vmax}.parquet'` naming is collision-prone (the I-1 bug
  in nfp-ingest's `append`), but here it's *masked* by the preceding wipe —
  build_store writes exactly one file per partition. Worth knowing the two
  writers share the scheme: a build-then-append sequence can land two
  `v_X_Y.parquet` files that collide cross-tool (I-1's scenario).

---

## `__main__.py`

The Typer CLI. Clean and readable; the issues are orchestration-shaped.

**[NEW] V-1 (Medium) — bare `alt-nfp` runs the destructive `build` (compounds
B-1), and the callback invokes Typer commands as plain functions.**

```python
@app.callback(invoke_without_command=True)
def main(ctx):
    load_dotenv()
    if ctx.invoked_subcommand is None:
        download(); download_indicators(); process(); current(); build(None)
```

Two problems in one place. (1) The most natural invocation of the tool — bare
`alt-nfp`, no subcommand — ends in `build(None)` → `build_store(releases_path=
None)` → a wipe-and-rebuild of the **canonical** store. This is the concrete
realization of B-1's danger; it is not merely that a power-user might run
`alt-nfp build`, it's that *typing the program name with no arguments* does. (2)
Calling `build(None)` and the other commands directly works only because none of
the bare-run commands has a *required* Typer option — a command like `snapshot`
(whose `as_of` is `typer.Option(...)`, required) would receive the `OptionInfo`
sentinel as its value if called this way, with no validation. The bare-run
sequence rests on an implicit, unenforced contract. **Fix:** factor each step's
real work into a plain function (`_download()`, `_process()`, …) that both the
Typer command and the callback call; never invoke a Typer-decorated function
directly. (And once B-1 is guarded, the bare run is no longer a footgun.)

**[NEW] V-2 (Low) — the documented "each step is idempotent" is false for
`build`.** Re-running `download`/`current` overwrites their outputs (idempotent
in effect). But re-running `build` does **not** preserve the store's state — it
reverts the store to "revisions + current `releases.parquet`," discarding any
rows added by direct appends since the last build. The "idempotent" claim in
`CLAUDE.md` papers over exactly the B-1 hazard; correct it to flag `build` as a
one-time bootstrap.

**[NEW] V-3 (Low) — `snapshot --as-of` doesn't enforce (or consistently apply)
the day-12 convention.** The help text says "(day-12 convention)," but:

```python
if grid_end is None:
    dates = [start]                       # uses the literal as_of, any day
else:
    while _date(y, m, 12) <= end:         # grid snaps to the 12th
        dates.append(_date(y, m, 12)); ...
```

So `alt-nfp snapshot --as-of 2026-01-15` (single) snapshots at a non-12th date
silently, while the grid path snaps to the 12th — and a grid started at
`2026-01-15` would *include* `2026-01-12`, before the requested start. Minor, but
since the day-12 convention is load-bearing for censoring, **validate that
`as_of.day == 12`** (or document the single-date exception explicitly).

**[SYS, H-4b] V-4 — graceful fetch degradation is good; the parse-drift gap lives
just downstream.** The visible head of `_build_release_calendar` catches
`FetchError` and falls back to release pages already on disk — a genuinely nice
defense against BLS's recurring 403s (I praised this in the system review). The
unguarded case is `parse_index_page` (in nfp-download) returning an empty/partial
list on HTML-structure drift: the *fetch* is protected, the *parse* is not, and
wrong/missing release dates feed straight into the censoring layer's vintage
dates. The call site is here; the parser and the fix (a cardinality/sanity
assertion on parse output) belong to nfp-download.

**[Deferred] V-5 — `process()` and the full `_build_release_calendar` body are
not in context.** I can see `process` calls `_build_release_calendar` and runs
revision processing (per CLAUDE.md), and that `_build_release_calendar` wires
`collect_release_dates` + `build_vintage_dates` + `SUPPLEMENTAL_RELEASE_DATES`
into the intermediate parquets — but not the bodies. No findings asserted on
them.

---

## Test coverage

`CLAUDE.md`'s test mapping lists `test_vintages.py` for **views & evaluation**,
and notes download-transport tests moved to nfp-download. There is **no listed
test for `build_store.py` and none for the CLI orchestration** — the two highest-
risk pieces in the package. The A0 gate verified `build_store` once, **manually**
(a from-scratch build into a scratch S3 prefix, row counts compared to the
reference store), not as pytest. So B-1/B-2/B-3 all live in code that has no
automated test, and the dangerous bare-run path (V-1) is exercised by nothing.
This is the same pattern as nfp-ingest (the careful, query-shaped code is tested;
the risky, side-effecting code is not), and it's why these issues persist
undetected. A `tmp_path`/local-store test of `build_store` (build → assert
schema + row count; build twice → assert determinism; build over a store with an
extra appended row → assert it's preserved *or* the guard fires) would cover
B-1/B-2/B-3 and run in CI with no credentials.

---

## Prioritized findings (package-scoped)

| ID | Sev | Location | One-liner |
|----|-----|----------|-----------|
| **B-1** | **Critical** | `build_store` (+ `__main__.build`) | Wipe-and-rebuild defaults to the canonical store; bare `alt-nfp` triggers it; guarded only by a comment ⇒ silent, irreversible loss of irreproducible live-captured rows. **[SYS]** |
| **D-1** | Medium | store write surface (cross-package) | Three writers, three `vintage_date` tie-break policies on one key. **[NEW]** |
| **B-2** | Medium | `build_store` dedup | Order-sensitive `keep='last'` over a vintage_date-excluding key ⇒ potential nondeterminism; undocumented current-wins-then-arbitrary tie-break. **[NEW]** |
| **B-3** | Medium | `build_store` write | No schema validation/cast before writing the foundational artifact; `diagonal_relaxed` can leak columns/dtypes. **[NEW]** |
| **V-1** | Medium | `__main__` callback | Bare `alt-nfp` runs destructive `build` (compounds B-1); callback invokes Typer commands directly (fragile for required-option commands). **[NEW]** |
| **TestGap** | Medium | `build_store` + CLI | Highest-risk code has no automated test (build_store verified once, manually). **[NEW]** |
| **H-4b** | Medium | `_build_release_calendar` → nfp-download parser | Fetch degrades gracefully; parse drift fails silently into vintage dates. **[SYS]** |
| **B-5** | Low–Med | `build_store.main()` | Redundant `python -m` CLI with silent-failure arg parsing. **[NEW]** |
| Low | Low | `__main__` / `build_store` | `snapshot` day-12 not enforced (V-3); "idempotent" claim false for `build` (V-2); reads ignore storage_options (B-4); shared collision-prone filename (note). **[NEW]** |

---

## Synthesis

**What's good.** The CLI is clean, readable, and well-documented; the
subcommand decomposition is sensible; and `_build_release_calendar`'s graceful
degradation to cached release pages when BLS's bot-detection 403s is genuinely
thoughtful defensive engineering. The package is *thin* in the right way — most
heavy lifting is delegated to the lower packages, and acquisition was correctly
pushed out in the A2 seam.

**The pattern under the findings (consistent with the prior two reviews).** The
package's risk is concentrated in one small, side-effecting module that the test
suite doesn't touch, and the most dangerous operation is reachable by the most
casual command. `build_store` is where the system's single Critical issue lives,
and `__main__`'s bare run wires it to the program name itself. The store's write
logic being split across two packages (D-1) is *why* its safety policies drifted
— and why "never rebuild in place" had to become a written rule instead of a
code invariant. Everything here reduces to the same prescription from the system
review: **turn the documented hard rule into a guard, and put the side-effecting
code under tests that run without credentials.**

**Top changes for `nfp-vintages`, in order.**

1. **Guard `build_store` against the canonical store (B-1)** — refuse the
   canonical URI without an explicit flag, or route the rebuild through an
   append/merge. This is the single highest-leverage change in the *entire*
   codebase; do it before anything else in this package.
2. **De-fang bare `alt-nfp` (V-1):** factor each pipeline step into a plain
   `_work()` function the callback calls, so the bare run can't invoke a
   half-resolved Typer command — and so guarding B-1 actually covers the bare
   path.
3. **Unify the store-write surface (D-1)** and make the dedup deterministic and
   schema-checked (B-2, B-3): one `write_partition(..., dedup_policy)` in
   `nfp_ingest.vintage_store`, one tie-break, a `cast(VINTAGE_STORE_SCHEMA)`
   before write.
4. **Add a store-free `build_store` test** (schema + row count; determinism on
   re-run; preservation-or-guard when an extra row was appended). Closes the
   B-1/B-2/B-3 blind spot in CI.
5. **Clean the CLI nits:** enforce `as_of.day == 12` (V-3), correct the
   "idempotent" claim (V-2), and remove or harden `build_store.main()` (B-5).

**Bottom line.** `nfp-vintages` is a thin, readable orchestration layer whose one
heavy module — `build_store` — carries the codebase's single most dangerous
operation, untested, defaulting to the canonical store, and reachable by typing
`alt-nfp` with no arguments. The package is otherwise in good shape; the work is
almost entirely "wire the known hazard shut and test the builder."

---

## Deferred to next pass (retrieval permitting)

`views.py` (the real-time / final / specific-vintage queries — verify they honor
the same as-of semantics as the censoring layer), `evaluation.py`
(`vintage_diff`, `build_noise_multiplier_vector` — the empirical multipliers that
feed back into the model's QCEW/CES noise), the entire `processing/` package
(CES triangular-CSV parsing; the QCEW 4-stream transform with persons→thousands
and the durable/nondurable + government-ownership hierarchy; `combine.py`), the
full bodies of `process` and `_build_release_calendar`, `__init__.py`, and
`test_vintages.py`. The `processing/` parsers are the most likely remaining home
for correctness bugs in this package. Re-run project-knowledge retrieval and I'll
complete them.