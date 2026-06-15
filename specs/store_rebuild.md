# Vintage-store rebuild — private hierarchy + total anchor, NSA, vintage-aware

**Status:** design spec for a clean rebuild of the vintage store. Supersedes the
"irreplaceable / never rebuild" premise in the root `CLAUDE.md`. Companions:
`store_audit_findings.md` (current state), `ces_qcew_industry.md` (the QCEW→CES
private crosswalk), `size_classes.md` (the size-class dimension),
`ces_growth_convention.md` (vintage/print semantics), `bloomberg_consensus.md`
(the downstream `00` scoring target).

Structural changes vs today's store: an **`ownership` axis** added (private /
total / government), the `00` total-nonfarm series kept as a **stored-not-modeled
scoring anchor**, `size_class` columns added, raw-NAICS provenance dropped, and
the `industry_type='national'` tag retired (it collided with
`geographic_type='national'`).

---

## 1. Premise — the store is replaceable; rebuild it deliberately

The store holds only **public CES + QCEW** data, reconstructable from the BLS
triangular/bulk files. The historical core (≤2023) is complete and correct; the
live-capture era has gaps (2024-12 missing the benchmark reprint; 2025-12 missing
the second/final prints; and below the supersector level, `06`/`08`/all sectors
are frozen at 2025-01 — see `store_audit_findings.md` + §10). So:

- **Retire "never rebuild in place."** Rebuilds run to a **scratch prefix**
  (`s3://alt-nfp/store-rebuild`), are **validated**, then **promoted** to canonical
  as a deliberate cutover. The canonical store never takes an unvalidated in-place
  build.
- **Parity-vs-frozen-reference is retired** for this layer — the rebuild
  *intentionally* diverges (ownership axis, total anchor, NSA, QCEW-mapped).
  A1/A2 goldens are **re-baselined** against the rebuilt store, not held.
- **Everything is materialized in the store** (no on-the-fly reconstruction): the
  QCEW→CES private series and the size-class cross-product are computed at build
  time and stored. The store holds **levels** only; growth is a downstream read.

---

## 2. Scope

**In scope.** A national, **not-seasonally-adjusted**, vintage-aware store. Two
sources, one shared schema. It carries:

- the **private industry hierarchy** (`ownership='private'`) — what the model
  nowcasts, and
- the **`00` total-nonfarm anchor** (`ownership='total'`) — **stored, not
  modeled**: the actual against which the nowcast's reconstructed total
  (private `05` + a downstream government estimate) is compared.

| `source` | role | content |
|---|---|---|
| `ces` | actuals + anchor | CES published prints (1st/2nd/3rd) for the private hierarchy **and** the `00` total anchor, from triangular + bulk files |
| `qcew` | input | QCEW employment **mapped to CES private codes** (`ownership='private'`), monthly, vintage-tagged |

**Coverage:** **2017-01 →** present (for now). **Units:** thousands of persons.

**Out of scope.** Government and total-services codes (`07`, `90`–`93`;
`ownership` `government`/`total`) and the downstream `00` composition, **see §11**; seasonal adjustment (downstream),
**see §11**; geography beyond national, **see §11**; live capture, **see §11**;
the provider store (separate repo/object-store), **see §8**.

---

## 3. Industry & ownership

The CES top aggregates encode **two axes** the old single `industry_code` column
conflated. The rebuild splits them: an `industry_type` (level within the industry
partition) and an `ownership` (private / total / government, QCEW-native).

| `industry_type` | `ownership` | code(s) | meaning |
|---|---|---|---|
| `total` | `total` | `00` | total nonfarm — **anchor, stored not modeled** |
| `total` | `private` | `05` | total private — **private-tree root** |
| `domain` | `private` | `06`, `08` | goods-producing / private service-providing |
| `supersector` | `private` | `10 20 30 40 50 55 60 65 70 80` | |
| `sector` | `private` | `11 21 22 23 31 32 42 44 48 51 52 53 54 55 56 61 62 71 72 81` | |
| — | `total` / `government` | `07`, `90`–`93` | `07` total service-providing (`ownership='total'`) + government `90`–`93` (`ownership='government'`) — **deferred, not stored** (§11) |

**The private hierarchy nests additively** and is **additively closed**:
`05 = 06 + 08`, each domain sums its supersectors, each supersector its sectors.
This matters because the model will eventually read industry as a nested tree
(`total → domain → supersector → sector`); a nested model needs children that sum
to their parent. NSA levels satisfy this by BLS construction (§6). The `00`
anchor sits **outside** the private tree (`00 = 05 + government`, and government
is not stored), which is exactly why `ownership='total'` flags it as a separate
anchor rather than a tree node.

The "model the private subtree, score against the total" boundary is then a clean
`ownership` filter: the model-input pipeline reads `ownership == 'private'`; the
`00` anchor (`ownership == 'total'`) is automatically excluded from model input.

The CES-internal vs NAICS-2-digit sector reconciliation (`ces_qcew_industry.md`
§2) and the two structural sums — Mining-and-Logging (`10` = `21` + Logging
`1133`) and Durable/Nondurable (`31`/`32` from 3-digit subsectors) — are applied
at build time per that spec.

> **`nfp_lookups` note.** The industry hierarchy/grammar lives in `nfp-lookups`.
> The rebuild adds the `ownership` column to `VINTAGE_STORE_SCHEMA`, retires the
> `industry_type='national'` value, and adds the `total` level. These are
> schema/grammar changes owned by `nfp-lookups`, scheduled in the implementation
> plan.

---

## 4. The two sources

### 4.1 `source=ces` — CES published prints (private hierarchy + `00` anchor)

`benchmark_revision` is **provenance-keyed, not basis-keyed** — each `(rev, bmr)`
cell has exactly one input source:

| `(rev, bmr)` | source | when |
|---|---|---|
| `(0,0) (1,0) (2,0)` | **triangular** (`cesvinall`) | the real-time 1st/2nd/3rd print history |
| `(0,0) (1,0) (2,0)` | **bulk** | the un-benchmarked **tail** (latest published month back to the first month after the last applied benchmark; currently 2026-01 → 2026-05) |
| `(2,1)` | **bulk** | **everything else** (benchmarked history), `vintage_date` = that year's February benchmark release |

So `bmr=1` is **only ever `rev=2`**, and is produced **only** by the bulk
benchmarked file — the triangular real-time prints are always `bmr=0`, even when a
print already sits on a post-benchmark basis (e.g. a December second print
published with the January release *after* the annual benchmark stays `(1,0)`;
its benchmarked partner is the separate `(2,1)` row). This reproduces the store's
exact four-combo population `(0,0)/(1,0)/(2,0)/(2,1)` and is what makes the §10
history gate hold; it is also how the missing 2024-12 / 2025-12 `(2,1)` rows get
filled (bulk benchmarked input). The next annual benchmark lands **early February
2027** (published with the January 2027 Employment Situation). *(Precedence
footnote for the impl plan: where triangular and bulk both cover a tail month,
bulk wins for the tail, triangular for established history.)*

Ownership tagging: the private hierarchy (§3) is `ownership='private'`; the `00`
series is the `ownership='total'` anchor, built by the same print/benchmark rules.
The deferred `07` (total services) and `90`–`93` (government) codes are never written.

### 4.2 `source=qcew` — QCEW mapped to CES private codes (input)

- **Crosswalk applied at build time** (`ces_qcew_industry.md`): `own_code=='5'`
  (→ `ownership='private'`), national (`area_fips=='US000'`), aggregate the listed
  `(industry_code, agglvl)` cells into the CES private codes of §3.
  `industry_code` is therefore the **CES code** (`05/06/08/10-80/sector`); **raw
  NAICS provenance is dropped**. (`own_code` 0/1/2/3 — total/government — are
  deferred with §11; QCEW writes `ownership='private'` only.)
- **Monthly levels:** use `month1_emplvl`, `month2_emplvl`, `month3_emplvl` of each
  quarter → three monthly rows, each **÷ 1000** (persons → CES-comparable
  thousands). Sum the measure column over cells; never sum a rate.
- **Per-vintage aggregation (critical, see §5).** Each CES-code aggregate is summed
  only over leaf cells sharing one QCEW `(revision, vintage_date)`; the result
  inherits that `(revision, vintage_date)`.

---

## 5. Vintage & as-of semantics

Both sources are vintage-aware so the real-time censoring (`vintage_store.py`
Layer-1 + `panel_adapter` Layer-2) works unchanged.

- **CES:** each store row's **vintage identity** is `(revision ∈ {0,1,2},
  benchmark_revision ∈ {0,1}, vintage_date)`, populated per §4.1. This is the
  row-identity key, **not** the growth-cohort key: per-cohort log-growth is
  computed downstream by grouping on `(geo, industry_type, industry_code,
  ownership, revision, benchmark_revision)` and sorting by `ref_date` —
  `industry_type` is **load-bearing** in the group, not optional, because code
  `55` exists at two levels (supersector `55` Financial activities vs sector `55`
  Management of companies); dropping it would collapse them into one cohort and
  difference the two series together. **Never** group on `vintage_date`
  (stamps misalign across the months being differenced,
  `ces_growth_convention.md` §4c). The store commits to neither growth form (§6);
  it stores levels.
- **QCEW — single-vintage (`revision = 0`) per industry.** BLS publishes QCEW
  revision history **only at the national total** (`qcew-revisions.csv` is
  area×field with no industry breakdown — verified 2026-06-15), and the open-data
  API serves only current values, so per-industry rev-1..4 levels **do not exist
  as a public source**. Each per-industry cell is therefore stored as a single
  `revision = 0` row (`benchmark_revision = 0` always — QCEW has no annual
  benchmark), `vintage_date` = the QCEW quarterly publication date for that
  ref-quarter from `nfp_lookups.revision_schedules.get_qcew_vintage_date` (exact
  from the calendar when available, else the lag-based approximation). The stored
  level is the current (latest-revised) value; QCEW **revision uncertainty is
  carried model-side** by the per-quarter noise schedule (`QCEW_REVISIONS` noise
  multipliers; nominal depth Q1=4/Q2=3/Q3=2/Q4=1), **not** as stored revision rows
  — a deliberate accepted trade-off (**decision A**, 2026-06-15; the per-industry
  rev-0 row holds a benchmarked value tagged at its initial publication date, a
  small accepted as-of lookahead). See `store_rebuild_acquire.md`. The crosswalk
  sum still **never crosses a QCEW vintage** (here trivially one rev-0 vintage per
  ref-quarter).
- **Publication lag.** QCEW lands ~5–6 months after the reference quarter; the
  reconstructed series inherits QCEW's `vintage_date`, so as-of knowability is
  honored automatically. The `(2,1)` CES benchmark rows carry the February
  benchmark-release `vintage_date`, so at historical as-ofs they censor out and
  the as-of layer falls back to the real-time prints (the existing lookahead
  block).

---

## 6. Representation — NSA, log-growth

The store holds **NSA levels** (`employment`, thousands). Growth is computed
per-cohort downstream and the model observes **`log(growth)`** — per-period
`log(emp_t / emp_{t-1})` (the expected choice) — **or**, alternatively,
`log(cumprod(growth))` (reconstructed log levels). The two are mutually
exclusive; the store commits to neither (it stores levels).

**NSA is required, not incidental.** NSA employment nests additively across the
industry hierarchy (§3) by BLS construction; SA aggregates are each independently
seasonally adjusted and do **not** sum, so a nested model on SA data would see
children that disagree with their parent. The **SA `00` headline** scored against
Bloomberg consensus is composed **downstream** from the private nowcast +
government estimate, then seasonally adjusted — outside this store.

---

## 7. Schema

`source=ces` and `source=qcew` share one schema (Hive-partitioned by
`(source, seasonally_adjusted)`; `seasonally_adjusted=false` only):

| column | type | notes |
|---|---|---|
| `geographic_type` | str | `national` (only, for now) |
| `geographic_code` | str | `00` |
| `ownership` | str | **new** — `private` / `total`; `government` deferred (§11) |
| `industry_type` | str | `total` / `domain` / `supersector` / `sector` (`national` retired) |
| `industry_code` | str | CES codes: `00` (total anchor), `05/06/08`, `10-80`, sectors (§3) |
| `ref_date` | date | day-12 convention |
| `vintage_date` | date | per §5 |
| `revision` | u8 | CES 0-2; QCEW 0-4 |
| `benchmark_revision` | u8 | CES 0/1; QCEW 0 |
| `employment` | f64 | thousands |
| `size_class_type` | str? | `size_classes.md` — null for CES + QCEW Q2/Q3/Q4 |
| `size_class_code` | str? | **singular**; null where `size_class_type` is null |

**Removed vs today:** raw-NAICS provenance; `industry_type='national'`.
**Added:** `ownership`, `size_class_type`, `size_class_code`, the `00` anchor
(as `ownership='total'`).

**All-sizes selection.** The headline (all-sizes) industry level is the
**null-size** row for CES + QCEW Q2/Q3/Q4 and the **`total`/`size_class_code='0'`**
row for QCEW Q1 (§8). A continuous all-sizes series must therefore select
`size_class_type IS NULL OR size_class_code = '0'` — a bare `size_class_type IS
NULL` silently drops every Q1 month.

---

## 8. Size-class dimension

Per `size_classes.md`: `size_class_type ∈ {total, small, medium, large}` (1/3/5/9
nested buckets over native QCEW `size_code` 1–9), `size_class_code` singular.
Lives in `source=qcew`, populated **only for QCEW Q1 — ref_dates with month ∈
{01, 02, 03}** (size is a Q1 establishment-size product, assigned by March
employment); **null for CES and for QCEW Q2/Q3/Q4** (months 04–12).

**Full cross-product on Q1.** For those Q1 months, every industry cell carries
every size scheme — the rows are the cross-product `industry_code ×
size_class_type`. A given `industry_code` at a given `(ref_date∈Q1, rev,
vintage_date)` expands to **one row per scheme code**: `total` (code `'0'` =
all-sizes), `small` (codes `'1'`–`'3'`), `medium` (codes `'1'`–`'5'`), and `large`
(codes `'1'`–`'9'`, the native QCEW `size_code`s). Each row stores a single code
(that is what "singular" means — the ranges above enumerate the rows, not a
stored value). `geographic_type == 'national'` for now. Size-class rows inherit
their industry parent's QCEW `(rev, vintage_date)` and `ownership='private'`.

The **provider store** (separate repo/object-store) bins its microdata to the
*same* scheme (March/third-month employment) so the two line up.

---

## 9. Build pipeline (to scratch)

1. **Acquire** (public): CES triangular (`cesvinall`) + CES bulk current; QCEW bulk
   (all quarters, national, private). 2017+.
2. **CES** → emit, per the §4.1 source table: triangular real-time prints as
   `(rev∈{0,1,2}, bmr=0)`; the bulk un-benchmarked tail as `(rev∈{0,1,2}, bmr=0)`;
   the bulk benchmarked history as `(rev=2, bmr=1)` at the February benchmark
   `vintage_date`. Tag the private hierarchy `ownership='private'` and the `00`
   series `ownership='total'`.
3. **QCEW** → per `(rev, vintage_date)`: apply the `ces_qcew_industry` crosswalk
   (`own_code=='5'` → `ownership='private'`), explode `month1/2/3_emplvl` to
   monthly, ÷1000, write the CES-coded private aggregates. For **Q2/Q3/Q4** the
   all-sizes industry level is written as a **null-size** row (per §7). For **Q1**
   ref_dates, ingest the size-class file at **native `size_code`** (`'1'`–`'9'` =
   the `large` scheme), then **derive** `small`/`medium` by the `size_class_members`
   rollup (`size_classes.md`) and `total` (`'0'`) by summing all native codes —
   never source or join `small`/`medium` codes directly from raw QCEW. On Q1 the
   all-sizes level is the `total`/`'0'` row **only** — emit **no** null-size row
   for Q1 (else it double-counts under §7's `IS NULL OR size_class_code='0'`
   selector).
4. **Write** both to `NFP_STORE_URI=s3://alt-nfp/store-rebuild` (the canonical guard
   refuses `…/store`; `is_canonical_store` from the audit branch).

---

## 10. Validation & promotion

**Acceptance gates** (replace frozen-reference parity; key on `industry_type +
industry_code + ownership + (rev, bmr) + values`, applying an explicit old→new
`industry_type` remap for the ≤2023 history join — `national/00`→`(total, total)`,
`domain/05`→`(total, private)`, supersectors/sectors unchanged. `industry_type`
stays in the key because code `55` is the lone cross-level collision —
supersector `55` (Financial activities) vs sector `55` (Management of companies)
— that `industry_code + ownership` alone cannot disambiguate):

- **History consistency:** rebuilt `source=ces` prints match the current store's
  rows where both exist (≤2023, the known-good core) — for the private hierarchy
  **and** the `00` anchor. The four-combo `(rev,bmr)` population must reproduce.
- **Gap fill — priority-ordered:**
  - *Hard gate:* `05` + the supersectors current to the published frontier (what
    the nowcast and the supersector narrative read today), and the 2024-12 /
    2025-12 December `(2,1)` cohorts complete.
  - *Reconstruct-and-validate (not launch-blocking):* `06`/`08`/sectors (frozen at
    2025-01 in the live store) are refilled from the triangular pass; validate the
    **additive nesting** (`05 = 06 + 08`, supersectors sum, sectors sum) where
    present, but a missing sector-month does not block promotion.
- **Reconstruction accuracy:** rebuilt `source=qcew` `05/06/08/…` vs *published*
  CES private at **benchmark months / annual averages**, in the *direction* the
  `ces_qcew_industry.md` §8 residuals predict — in particular the `81/80/08/05`
  residual must be **small and non-negative** (the 8131 religious-org inclusion);
  a negative residual there fails the gate. The numeric magnitude tolerance is
  **deferred to the implementation plan** (set a bound on `|residual|` before the
  gate is operational; do not hardcode equality).
- **Vintage integrity:** `_validate_censored_selection`-style fail-fast checks pass
  on an as-of slice (no dups, no cross-vintage sums, no nulls/zeros).

**Promotion:** once gates pass, cut over deliberately (repoint `NFP_STORE_URI` to
the validated prefix, or copy scratch→canonical with the explicit
`--allow-canonical` escape hatch). Document the cutover; keep the prior canonical
snapshot until the new one is confirmed in the model.

---

## 11. Deferred / future

- **Government & total composition:** government codes `90`–`93`
  (`ownership='government'`, QCEW `own_code` 1/2/3) and total service-providing
  `07` (`ownership='total'`). The downstream step adds a government estimate to the
  private `05` nowcast to reconstruct `00`; that estimate, and any stored government
  series, are out of this rebuild's scope.
- **Live capture (D4):** cron a daily read of the BLS feed (`https://www.bls.gov/feed/`)
  to keep the frontier fresh. Until then a **stale store is acceptable** (rebuild
  periodically). `_fetch_ces_releases` is now import-fixed (on `main`) but not yet
  load-bearing.
- **NAICS-vintage drift:** use the NAICS-2022 crosswalk for all years initially
  (2017+ coverage limits exposure); add vintage-aware crosswalks later.
- **Geography** beyond national; **seasonal adjustment** (composed downstream, not
  in this store). **Births-deaths:** unchanged by this rebuild.

---

## 12. Open decisions

None outstanding. Resolved this round:

- **`benchmark_revision` is provenance-keyed** (§4.1): triangular → `bmr=0`; bulk
  benchmarked history → `(2,1)`; bulk tail → `bmr=0`. `bmr=1` ⟺ `rev=2`.
- **Ownership axis added** (§3, §7): `industry_type ∈ {total, domain, supersector,
  sector}` × `ownership ∈ {total, private, government}`; `05`=`(total, private)`,
  `00`=`(total, total)`, `06/08`=`(domain, private)`. `industry_type='national'`
  retired.
- **`00` total-nonfarm kept as a stored-not-modeled anchor** (reverses the earlier
  "no need to store 00" — needed for the nowcast-vs-actual comparison).
- **Nested additive hierarchy** (§3, §6), NSA-required; sectors future-facing.
- **QCEW per-industry is single-vintage (`rev=0`)** — **decision A** (2026-06-15):
  BLS publishes no per-industry revision history, so per-industry QCEW is stored
  rev-0 (current value) and revision uncertainty stays model-side
  (`QCEW_REVISIONS` noise). Only the national total has rev 0–4. (Rejected: B,
  proportional synthesis from total-level revision ratios.)
- Size cross-product keyed on `industry_code` (§8); all-sizes selection convention
  (§7); reconstruction-accuracy tolerance deferred (§10).

The spec is ready for an implementation plan.
