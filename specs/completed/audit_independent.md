# Independent Audit — `alt-nfp` workspace

**Provenance.** Independent, source-grounded multi-agent audit (workflow run
`wwock9wsk`): 7 deep finders (one per package + cross-cutting), **every finding
adversarially verified** against the actual current source on `main` and against
the frozen reference at `~/Projects/alt_nfp`. 40 agents, ~2.5M tokens, ~19 min.
Read-only throughout — no code changed, no store written, no MCMC run.

This is distinct from the four memory-based audits (`audit_alt_nfp.md`,
`audit_nfp_ingest.md`, `audit_nfp_lookup.md`, `audit_nfp_vintages.md`,
`audit_nfp_model.md`) which were written *without* retrieval and explicitly
deferred the highest-risk surfaces. This pass had full source access and an
adversarial verification stage whose job was to **refute** each finding.

---

## Executive summary

**The codebase is in genuinely good shape.** The adversarial pass refuted,
downgraded, or classified-as-parity-faithful the large majority of findings. The
memory-based audits' big-ticket items all verified **clean / parity-faithful**:
`build_store` B-1 is already guarded (C-1), the `processing/` parsers (CES
triangular, QCEW persons→thousands) and `nowcast.py` level-reconstruction
arithmetic are byte-faithful ports, the QCEW revision schedule is internally
consistent, and the import-boundary / no-secrets discipline holds. The verifier
even overturned a *finder* error (IND-LK-2 below).

**Disposition tally (25 findings, by verified verdict):**

| Verdict | N | Meaning |
|---|---|---|
| `refuted` / `already_handled` | 5 | not a live bug, or already fixed |
| `confirmed` (fix now, parity-neutral) | 9 | real + safe to fix without diverging from frozen ref |
| `parity_intentional` (backlog) | 11 | real but byte-faithful to the frozen ref → fixing breaks parity |

**Only two findings are materially substantive (both MEDIUM, both genuinely new):**

- **IND-IMD-1** — `_fetch_ces_releases` stamps the post-benchmark (benchmark-revised)
  CES level onto the **rev1/bmr0 ordinary-print track** as well as the correct
  rev2/bmr1 reprint track. After a benchmark lands, the rev1/bmr0 row carries a
  level ~900k off. Live-capture path (not golden-mastered → fixable); bounded
  to scratch/local rebuilds today. **The correct value is not in the source**, so
  the only fix is to stop emitting the wrong level — a semantics call. → needs a ruling.
- **IND-XC-2** — `scripts/mirror_store.py` is an **unguarded direct-s3fs write** to
  whatever `NFP_STORE_URI` points at, bypassing both the `build_store` canonical
  guard and the pytest safety net. Same class as the wipe incident, through a door
  with none of the post-incident protections. Manual trigger, bounded blast radius
  (clobbers the base partition, not the hash-suffixed live-capture appends). → clean fix.

Everything else is a small parity-neutral cleanup or a parity-faithful backlog item.

---

## nfp-lookups (LK)

> *Finder summary:* genuinely clean. All parity-sensitive reference DATA
> (revision schedules, schemas, benchmark dates, industry hierarchy, FIPS maps,
> series-ID grammar) is byte-identical to the frozen reference. The
> QCEW-schedule × `_select_qcew_at_horizon` cross-check is internally consistent.

| ID | Title | sev (finder→verified) | verdict | disposition |
|---|---|---|---|---|
| IND-LK-1 | `storage_options_for`/`_store_location` don't validate the S3 env contract | low→**none** | `refuted` | none — trigger self-defeating (URI+creds share one `.env`); mooted by conftest net + Task #18 ruling |
| IND-LK-2 | "2025 benchmark `-862` is stale" | low→**none** | `refuted` | none — **premise inverted**: `-862` IS the correct Feb-2026 final (prelim was `-911`) |
| IND-LK-3 | `en_series_id` emits QCEW-invalid IDs + wrong docstring (dead in v2) | low→low | `parity_intentional` | backlog — dead code, byte-faithful |
| IND-LK-4 | `validate_panel` declares `_REQUIRED_NON_NULL`/`_VALID_*` it never enforces | low→low | `parity_intentional` | backlog — dead validation, byte-faithful (A2-adjacent) |

## nfp-download (DL)

> *Finder summary:* faithful, well-tested port with three landed improvements
> (curl_cffi impersonation, `ParseError` cardinality guard, transport retries).
> Store-feeding QCEW filter constants byte-identical to the reference. No secrets.

| ID | Title | sev | verdict | disposition |
|---|---|---|---|---|
| IND-DL-1 | `fetch_qcew*` return empty frame on total request failure (silent-ish) | med→low | `parity_intentional` | backlog — warns already; byte-faithful |
| IND-DL-2 | `download_qcew_bulk` returns path without writing when no data | low→low | `parity_intentional` | backlog — mechanism partly overstated; byte-faithful |
| IND-DL-3 | FRED `_request_with_retry` masks final transport error with stale 5xx | low→low | `parity_intentional` | backlog — diagnostics-only; byte-faithful |
| IND-DL-4 | `get_with_retry` raises `UnboundLocalError` when `max_retries=0` | low→low | `confirmed` | backlog — real but zero exposure (all callers ≥3); byte-faithful |
| IND-DL-5 | `download_qcew_bulk` default `end_year=2025` silently truncates fresh runs | low→low | `parity_intentional` | backlog — freshness gap; byte-faithful |
| IND-DL-6 | `parse_index_page` has no partial-cardinality assertion | low→**none** | `refuted` | none — on-disk rebuild bounds it; optional logging only |

## nfp-ingest — store / censoring (IST)

> *Finder summary:* the A2 parity heart (CES triangular selection, QCEW rank
> rules, benchmark sentinel, dedup key) is byte-for-byte identical to the frozen
> reference. The two deliberate divergences (append content-suffix, compact
> tie-break) are well-tested fixes (T2/T3) and don't threaten A0.

| ID | Title | sev | verdict | disposition |
|---|---|---|---|---|
| IND-IST-1 | CES fallback can leak a `benchmark_revision>0` row as a spurious `revision_number=-1` obs | med→low | `parity_intentional` | backlog — latent (=I-2), byte-faithful (A2) |
| IND-IST-2 | `_validate_censored_selection` never inspects `benchmark_revision` | low→low | `confirmed` | backlog — pairs with IST-1; validator-only (parity-neutral) |
| IND-IST-3 | append/compact "earliest vintage wins" comments overstate equivalence | low→low | `confirmed` | **fix now** — comment-only |
| IND-IST-4 | validator check #7 comment ("≥90%") doesn't match code (absolute `<10`) | low→low | `parity_intentional` | backlog — verbatim port |
| IND-IST-5 | append content-suffix filename hashing correctly closes the clobber bug | low→low | `already_handled` | none — confirms T2 |

## nfp-ingest — model data / snapshots / releases (IMD)

> *Finder summary:* largely clean and faithfully ported. BD covariates fully
> removed (SCHEMA_VERSION=3). `compositing.py` byte-identical. Provider `g_pp`
> lookahead checked — NOT a leak.

| ID | Title | sev | verdict | disposition |
|---|---|---|---|---|
| **IND-IMD-1** | `_fetch_ces_releases` stamps benchmark-revised level onto the rev1/bmr0 track post-benchmark | **med→med** | `confirmed` | **fix now (needs ruling)** — value bug, live-capture path |
| IND-IMD-2 | orphaned dead `bd_qcew_lag` config field (BD-removal residue) | low→low | `confirmed` | **fix now** — parity-none |
| IND-IMD-3 | unused `date_list` param in `_qcew_series_with_meta` | low→low | `parity_intentional` | backlog — verbatim port |
| IND-IMD-4 | QCEW compositing weights use latest vintage, no as-of (weight lookahead) | low→low | `parity_intentional` | backlog — only cell-level providers; default 'G' national unaffected |

**IND-IMD-1 detail.** `fetch_ces_national` reads BLS `ce.data.0.AllCESSeries` —
one (latest-published) level per (series, date). After a benchmark lands that is
the benchmark-revised level. `_latest_ces_vintage_dates` now correctly keys on
`(ref_date, benchmark_revision)` (spec-blessed, §4(c)-i), so a benchmarked
December emits two metadata rows: `(rev=1,bmr=0)` and `(rev=2,bmr=1)`. The join
`raw.join(latest_vdates, on='ref_date', how='left')` (releases.py:172) is on
`ref_date` **alone**, fanning the single post-benchmark level onto *both* rows.
The reprint row is right; the rev1/bmr0 row gets the benchmark-revised level when
its true value is the pre-benchmark second print (spec §5). Defect is
post-benchmark **re-capture** only (on the real rev1 release day the benchmark
row doesn't yet exist). Existing regressions assert the metadata triples, never
the stamped **level** — so it's uncaught. Reach to canonical is gated behind the
`build_store` guard + the (unwired) append path; `mirror_store.py` is the one
manual door (see IND-XC-2).

## nfp-vintages (VT)

> *Finder summary:* parsers (`ces_triangular`, `qcew_bulk`, `combine`,
> `evaluation`) are byte-for-byte ports (the persons→thousands `/1000.0` factor,
> the triangular diagonal, the 4-stream hierarchy all match). The `build_store`
> canonical guard is correct and not bypassable. `views.py`/`evaluation.py` have
> divergent as-of semantics but ZERO production callers. No noise-multiplier
> double-application.

| ID | Title | sev | verdict | disposition |
|---|---|---|---|---|
| IND-VT-1 | `snapshot --as-of` doesn't enforce the documented day-12 convention | low→low | `confirmed` | **fix now** — parity-none |
| IND-VT-2 | `build_store` dedup key omits `vintage_date` (latent nondeterminism) | low→low | `parity_intentional` | backlog — latent, byte-faithful (A0) |
| IND-VT-3 | `build_store` writes without casting to `VINTAGE_STORE_SCHEMA` | low→low | `parity_intentional` | backlog — latent, byte-faithful (A0) |
| IND-VT-4 | `views.py` as-of semantics diverge; dead/legacy code | low→low | `parity_intentional` | backlog — unused, byte-faithful (delete or docstring) |
| IND-VT-5 | `build_store.main()` redundant entry; silent `--releases` no-op | low→low | `confirmed` | **fix now** — parity-none |

## nfp-model (MD)

> *Finder summary:* faithful port. PyMC→NumPyro parametrizations match
> (InverseGamma verified numerically); `nowcast.py` level reconstruction agrees
> across reference / serial / batched. Documented don't-fix traps (phi clip, g0
> innovation-SD, non-centered Fourier, medium preset) correctly inherited.

| ID | Title | sev | verdict | disposition |
|---|---|---|---|---|
| IND-MD-1 | parity-gate z-test escape hatch weakens at low-ESS sites | med→low | `parity_intentional` | backlog — documented plan-of-record design; "passes any effect size" overstated (new reparam side is high-ESS) |
| IND-MD-2 | data intake has no consistency checks; `jnp.take` clamps OOB | med→low | `refuted` | none/backlog — unreachable by construction (`np.where` → in-bounds) |
| IND-MD-3 | `from_snapshot` carries dead `births`/`births_obs` the model never reads | low→low | `confirmed` | **fix now** — parity-none |
| IND-MD-4 | `nowcast` `nanmean` vs `batch` plain `mean` | low→**none** | `refuted` | none — compared fields already use identical aggregators |

## Cross-cutting / architecture (XC)

> *Finder summary:* architectural discipline is strong — zero upward-import
> violations, no cross-package private imports, nfp-model imports no `nfp_*`, no
> leaked secrets, consistent dependency bounds. The store-write safety net is
> comprehensive and self-tested.

| ID | Title | sev | verdict | disposition |
|---|---|---|---|---|
| IND-XC-1 | all three parity gates (A1/A2/A3) self-skip in CI → "DONE" never runs automatically | high→med | `confirmed` | **fix now (partial)** — CI note now; synthetic smoke aligns with deferred H-2b |
| **IND-XC-2** | `mirror_store.py` unguarded direct-s3fs write to canonical store | high→med | `confirmed` | **fix now** — store-safety, incident-adjacent |
| IND-XC-3 | `VINTAGE_STORE_SCHEMA` re-declared in nfp-ingest; lookups copy is dead | med→low | `parity_intentional` | backlog — byte-identical, byte-faithful |
| IND-XC-4 | QCEW quarter→revision schedule hardcoded in 2 sites, not imported | med→low | `parity_intentional` | backlog — byte-identical + currently agree, byte-faithful |
| IND-XC-5 | A2 golden gate excludes 3 arrays from comparison | low→low | `confirmed` | **fix now** — one-line absence assertion |

---

## What the memory-based audits got wrong (corrected here)

- **vintages B-1 / V-1 (Critical)** — already remediated (C-1/T1); the guard is
  correct and not bypassable.
- **lookups L-1** (`_find_base_dir` "silent degradation") — the fallback resolves
  correctly and is an *improvement* over the reference (markers are committed
  files, not gitignored `data/`).
- **lookups L-2** (S3 env validation) — refuted (IND-LK-1).
- **lookups benchmark currency** — `-862` is the correct final, not stale (IND-LK-2).
- **model MOD-2** (intake asserts) — the OOB path is unreachable by construction (IND-MD-2).
- **model MOD-4** (gate power) — real but the "weakest where bugs hide" framing is
  overstated; the reparametrized side is high-ESS (IND-MD-1).
- The deferred high-risk surfaces (parsers, `nowcast.py`, series-ID grammar, QCEW
  schedule) are all **clean and parity-faithful** — the original audits' worry
  about them was unfounded.

Implementation plan: see [`plans/completed/9-audit_independent.md`](../plans/completed/9-audit_independent.md) (retired as stale 2026-06-21).
