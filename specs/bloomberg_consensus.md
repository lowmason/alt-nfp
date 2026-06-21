# Bloomberg consensus integration (independent spec)

Status: **independent / portable; adapter built, file pending (2026-06-19).** The
`alt-nfp` consensus adapter (`nfp_vintages/competitors/consensus.py`:
`load_consensus`, `Consensus`) is **built and wired** — it is consumed by the
**Total-track (Track B)** backtest `run_a5_backtest.py:cmd_total`, which scores the
assembled Total NFP (private nowcast + government wedge) against this consensus.
What remains is the **file itself**: it is proprietary Bloomberg-derived data, absent
from this public repo, so locally the consensus column renders `—` until the file
lands (the `None`-tolerant path below). This spec stays self-contained so it can be
lifted into a Bloomberg-terminal workspace. It defines (a) the **contract** the
adapter expects — the part this repo owns and pins — and (b) a **Bloomberg-side
retrieval recipe** to produce a file in that contract. Companions:
`specs/completed/a5_real_competitors.md` (the harness this consensus scores in) and
`specs/completed/government_wedge.md` (the government forecast that makes the Total contest
valid — consensus is a Total object, meaningless against the private nowcast alone).

The split matters: **the contract is normative; the retrieval recipe is a
reference implementation.** As long as the file matches the contract, the
A5 harness consumes it unchanged regardless of how it was produced (Bloomberg
Excel add-in, `blpapi`/`xbbg` in Python, a manual export, or a hand-built
CSV).

## TL;DR

1. Produce a parquet/CSV with one row per BLS reference month:
   `ref_month, consensus_median_change_k, survey_date, release_date, source`
   (+ optional dispersion fields). Units: **thousands of jobs, MoM net
   change in total nonfarm payrolls, SA** — identical to `alt-nfp`'s
   `change_k`.
2. The consensus value for reference month *M* is the **median forecast as
   recorded at release-eve of M** (the final survey before BLS prints), so it
   aligns to A5's *release-eve* regime.
3. Bloomberg source: economic-release ticker **`NFP TCH Index`** (US Nonfarm
   Payrolls Total MoM Net Change, SA), survey field **`BN_SURVEY_MEDIAN`**,
   release date **`ECO_RELEASE_DT`**; optional **`BN_SURVEY_AVERAGE`**,
   **`BN_SURVEY_HIGH`**, **`BN_SURVEY_LOW`**, **`BN_SURVEY_NUMBER_OBSERVATIONS`**,
   **`FORECAST_STANDARD_DEVIATION`**.
4. The `alt-nfp` side is a pluggable adapter `load_consensus(path=None)` that
   returns the contract DataFrame if the file exists, else `None` (the
   scoreboard then renders the consensus column as `—`). No code change is
   needed when the file lands — only the file.

## 1. The contract (normative — `alt-nfp` owns this)

The consensus adapter reads a single file (parquet preferred; CSV accepted)
with exactly these columns:

| column | type | meaning | required |
|---|---|---|---|
| `ref_month` | date | first of the BLS **reference** month being forecast (e.g. `2025-06-01` for the June payrolls report) | ✅ |
| `consensus_median_change_k` | float | median economist forecast of the MoM net change in total nonfarm payrolls, **thousands, SA** | ✅ |
| `survey_date` | date | date the median snapshot was taken (≈ release-eve of M) | ✅ |
| `release_date` | date | BLS release date for `ref_month`'s first print | ✅ |
| `source` | str | provenance tag, e.g. `"bloomberg"` | ✅ |
| `consensus_mean_change_k` | float | survey average | optional |
| `consensus_high_change_k` | float | survey high | optional |
| `consensus_low_change_k` | float | survey low | optional |
| `n_forecasts` | int | number of contributing forecasters | optional |
| `consensus_std_k` | float | cross-forecaster std dev | optional |

Rules the adapter enforces (fail-fast, mirroring the store's
`_validate_censored_selection` discipline):

- `ref_month` unique, day = 1, monotonic.
- `consensus_median_change_k` in thousands (sanity bound: `|x| < 2000`
  outside obvious COVID months; warn, don't drop).
- `survey_date < release_date` and `survey_date` within ~10 days before
  `release_date` (the median must be a pre-release snapshot, not a backfill of
  the actual).
- Units are **net change**, not the level and not a percentage.

**Default location.** `data/competitors/consensus.parquet`
(gitignored, like all of `data/`), overridable via `NFP_CONSENSUS_PATH`. The
file is proprietary Bloomberg-derived data and must never be committed (this
repo is public).

**Adapter behavior.** `load_consensus(path=None) -> pl.DataFrame | None`:
resolve `path` → `NFP_CONSENSUS_PATH` → default; return the validated frame
if the file exists, else `None`. A `None` return is a first-class state: the
A5 scoreboard prints the consensus column as `—` and the gate is still
structurally satisfied (the join/scoring path is exercised by a small
committed fixture under `tests/`).

## 2. Bloomberg-side retrieval (reference recipe — verify in-terminal)

The goal is the **historical point-in-time median** for each past NFP
release: the consensus *as it stood just before* each release, not a single
current snapshot. There are two standard ways; pick whichever your workspace
supports. Field mnemonics are confirmed against Bloomberg's economic-survey
field set; the exact historical-retrieval mechanism is the part to validate
against your terminal's entitlements.

### 2a. The ticker and fields

- **Release ticker:** `NFP TCH Index` — "US Employees on Nonfarm Payrolls
  Total MoM Net Change SA" (the realized print; thousands).
- **Survey fields:** `BN_SURVEY_MEDIAN` (→ `consensus_median_change_k`),
  `ECO_RELEASE_DT` (→ `release_date`), and optionally `BN_SURVEY_AVERAGE`,
  `BN_SURVEY_HIGH`, `BN_SURVEY_LOW`, `BN_SURVEY_NUMBER_OBSERVATIONS`,
  `FORECAST_STANDARD_DEVIATION`, `ACTUAL_RELEASE` (the realized value, useful
  for a self-check against the store).
- **Reference-period field:** the indicator's observation/reference period
  (often surfaced as `ECO_RELEASE_PERIOD` or via the release row's
  period label) → maps to `ref_month`. Confirm the exact mnemonic; if absent,
  derive `ref_month` as the calendar month **before** `ECO_RELEASE_DT`
  (NFP for month M releases in M+1).

### 2b. Excel add-in (no API entitlement needed)

Use the economic-release/`ECO` data. The pattern (verify field availability):

```
' Per release event, the survey median + release date:
=BDP("NFP TCH Index","BN_SURVEY_MEDIAN")          ' current/next release median
=BDP("NFP TCH Index","ECO_RELEASE_DT")

' Historical release list + per-release survey snapshot (bulk):
=BDS("NFP TCH Index","ECO_FUTURE_RELEASE_DATE_LIST")   ' enumerate release dates
' then per release date, override to pull that release's median.
```

For a full back-history the robust route is the **ECO** function on the
terminal (Economic Calendar → NFP → export the survey-median column with
release dates), or the community Excel scripts that wrap this (e.g. the
`BBG-ECO-EXCEL` pattern). Export to the contract columns.

### 2c. Python (`blpapi` / `xbbg`)

```python
from xbbg import blp

# Enumerate historical release dates, then pull the as-recorded survey median
# per release (point-in-time). The exact override key for "median as of the
# release" depends on entitlements — validate against `ACTUAL_RELEASE` to
# confirm you are reading the pre-release median, not the realized print.
rel = blp.bds("NFP TCH Index", "ECO_FUTURE_RELEASE_DATE_LIST")
# For each release date d in rel: pull BN_SURVEY_MEDIAN with the appropriate
# reference-date override, plus ECO_RELEASE_DT, ACTUAL_RELEASE, n-forecasts.
```

Whatever the mechanism, the **acceptance test** below is what certifies the
file — not the API path.

## 3. Transformation and alignment

- **Units:** `NFP TCH Index` is already thousands, MoM net change, SA — no
  conversion. This matches `alt-nfp` `change_k` exactly. Do **not** use a
  level series or a percentage.
- **`ref_month`:** the month BLS is reporting on, *not* the release month.
  NFP for reference month M releases in M+1. Set `ref_month` to the first of
  M. If derived from `ECO_RELEASE_DT`, subtract one month.
- **`survey_date`:** the date the median snapshot was taken; should be
  release-eve (the final pre-print survey). This is what aligns the consensus
  to A5's release-eve regime, where the model is censored to the same day.
- **Coverage:** aim for 2012-01 onward (the model's sample start). Pre-2012
  rows are harmless but unused. COVID months (2020–2021) are kept in the file
  but excluded from headline metrics downstream.

## 4. Wiring into `alt-nfp`

The adapter lives at `nfp_vintages/competitors/consensus.py` (per
`a5_real_competitors.md` §6). It is consumed on the **Total track (Track B)**, not
the private one: `run_a5_backtest.py:cmd_total` calls `load_consensus()`, wraps it in
`Consensus`, and for each release-eve target scores
`consensus_median_change_k` against the **assembled Total** (private nowcast +
government wedge) and the **Total `00` first print** — consensus forecasts the Total
number, so it is meaningless against the private nowcast alone. (It does **not**
appear on the private scoreboard or in the private MZ.) Until the file exists,
`load_consensus()` returns `None`, `Consensus.predict` returns `None`, and the
consensus column renders `—`; the join and scoring are still exercised by committed
synthetic fixtures (null + populated) so the path cannot rot.

## 5. Acceptance test (certifies any file, any source)

Spot-check the produced file against known published Bloomberg medians and
realized prints (independent of how it was pulled):

- 2025-01 reference month (Dec-2024 jobs, released early Jan-2025): Bloomberg
  median ≈ **+165k** (widely reported pre-release).
- A handful of rows where `ACTUAL_RELEASE` is available: confirm
  `consensus_median_change_k` ≠ `ACTUAL_RELEASE` (i.e. you captured the
  *forecast*, not the realized print) and that `survey_date < release_date`.
- Cross-check 3–5 `release_date` values against the BLS Employment Situation
  schedule and against `alt-nfp`'s own `_ces_publication_date(M)`.

If those pass, the file is contract-valid and A5 consumes it with no code
change.

## 6. Caveats

- **Point-in-time, not revised consensus.** Bloomberg may carry a revised or
  late-updated median; capture the median **as of release-eve**, or the
  comparison leaks information the street did not have.
- **Survey timing varies.** The final survey usually locks a day or two
  before the print; `survey_date` records the actual snapshot date so the
  release-eve alignment is auditable.
- **Relevance/coverage drift.** `n_forecasts` (and `RELEVANCE_VALUE`) vary
  over time; carry them if available so thin-survey months can be flagged.
- **Provenance.** Tag `source="bloomberg"` (or the actual vendor). Keep the
  raw pull alongside the contract file for reproducibility; never commit
  either (proprietary; public repo).

## Sources

- Bloomberg economic-survey fields (`BN_SURVEY_MEDIAN`, `BN_SURVEY_AVERAGE`,
  `BN_SURVEY_HIGH`, `BN_SURVEY_LOW`, `BN_SURVEY_NUMBER_OBSERVATIONS`,
  `ECO_RELEASE_DT`, `FORECAST_STANDARD_DEVIATION`): university Bloomberg
  Excel add-in guides (Yale, Dartmouth, Penn, MSU) and the `BBG-ECO-EXCEL`
  project — verify exact mnemonics and entitlements against your terminal.
- `NFP TCH Index` = US Nonfarm Payrolls Total MoM Net Change, SA.
