# TODO: BLS Publication Calendar — Exact Release Dates for CES, QCEW, and SAE

## Context

The model's vintage-tracking infrastructure currently relies on approximate
publication dates derived from structural lag patterns (`RevisionSpec.lag_months`
in `src/alt_nfp/lookups/revision_schedules.py`).  The `PublicationCalendar`
dataclass exists but is unpopulated — its `from_parquet` method expects a
file that doesn't yet exist.

Exact publication dates matter for two reasons: (1) real-time backtesting
must reconstruct the precise information set available on any given date,
and (2) the nowcasting window between payroll provider data availability
(~1st of the month) and official release is the primary value proposition.
Off-by-a-week errors in vintage dating can make backtests look artificially
good or bad.

BLS publishes forward-looking release schedules on fixed URLs.  We have
scraped the current schedules for three programs:

- **CES** (Employment Situation): https://www.bls.gov/schedule/news_release/empsit.htm
- **QCEW** (County Employment and Wages): https://www.bls.gov/schedule/news_release/cew.htm
- **SAE** (State Employment and Unemployment): https://www.bls.gov/schedule/news_release/laus.htm

This TODO builds the exact-date calendar, extends `PublicationCalendar` to
cover SAE, and wires it into the vintage lookup pipeline so that
`get_ces_vintage_date` / `get_qcew_vintage_date` prefer exact dates when
available and fall back to lag-based approximations otherwise.

### Government shutdown note

The CES and SAE schedules for this period show Oct 2025 missing.  Per
`bls_estimates/release_dates/vintage_dates.py`, Oct 2025 CES/SAE data was
released together with Nov 2025 on Dec 16, 2025 (CES) and Dec 11, 2025
(SAE).  The hard-coded supplemental dates in that module handle this.  The
calendar built here should include these combined-release entries.

## References

- `src/alt_nfp/lookups/revision_schedules.py` — `PublicationCalendar`,
  `RevisionSpec`, `get_ces_vintage_date()`, `get_qcew_vintage_date()`
- `src/bls_estimates/release_dates/` — existing scraper pipeline
  (parser, vintage_dates, config) for historical release dates
- `src/bls_estimates/release_dates/vintage_dates.py` — supplemental
  dates, shutdown handling constants

## Code Style

- Python 3.12, line length 100, formatter: black, linter: ruff
- Use **Polars** (not pandas) for all DataFrames
- Prefer **single quotes** over double quotes for strings
- **Two blank lines** after class and function definitions (PEP 8)
- Type hints on all public functions
- Docstrings on all public functions and classes (numpy-style)

---

## Task 1: Hard-Code Scraped Release Dates

### What to create

Create `src/alt_nfp/lookups/publication_dates.py` containing three
dictionaries of `(reference_period, release_date)` tuples, one per program.
These are the **first-print (revision 0)** publication dates scraped from
the BLS schedule pages.

### CES (Employment Situation) — first-print release dates

All releases at 08:30 AM ET.

```python
from datetime import date

# CES first-print (revision 0) release dates.
# Source: https://www.bls.gov/schedule/news_release/empsit.htm
# Key: ref_month (1st of month), Value: publication date
CES_RELEASE_DATES: dict[date, date] = {
    date(2024, 10, 1): date(2024, 11, 1),
    date(2024, 11, 1): date(2024, 12, 6),
    date(2024, 12, 1): date(2025, 1, 10),
    date(2025, 1, 1): date(2025, 2, 7),
    date(2025, 2, 1): date(2025, 3, 7),
    date(2025, 3, 1): date(2025, 4, 4),
    date(2025, 4, 1): date(2025, 5, 2),
    date(2025, 5, 1): date(2025, 6, 6),
    date(2025, 6, 1): date(2025, 7, 3),
    date(2025, 7, 1): date(2025, 8, 1),
    date(2025, 8, 1): date(2025, 9, 5),
    # Sep 2025: released Nov 20 (gov shutdown delayed Oct release)
    date(2025, 9, 1): date(2025, 11, 20),
    # Oct 2025: released with Nov on Dec 16 (gov shutdown)
    date(2025, 10, 1): date(2025, 12, 16),
    date(2025, 11, 1): date(2025, 12, 16),
    date(2025, 12, 1): date(2026, 1, 9),
}
```

### QCEW (County Employment and Wages) — release dates

All releases at 10:00 AM ET.  QCEW is quarterly; key is `(year, quarter)`.

```python
# QCEW release dates.
# Source: https://www.bls.gov/schedule/news_release/cew.htm
# Key: (ref_year, ref_quarter as int 1-4), Value: publication date
QCEW_RELEASE_DATES: dict[tuple[int, int], date] = {
    (2025, 2): date(2025, 12, 19),
    (2025, 3): date(2026, 3, 10),
    (2025, 4): date(2026, 6, 2),
    (2026, 1): date(2026, 8, 28),
    (2026, 2): date(2026, 12, 2),
}
```

### SAE (State Employment and Unemployment) — first-print release dates

All releases at 10:00 AM ET.

```python
# SAE first-print (revision 0) release dates.
# Source: https://www.bls.gov/schedule/news_release/laus.htm
# Key: ref_month (1st of month), Value: publication date
SAE_RELEASE_DATES: dict[date, date] = {
    date(2024, 10, 1): date(2024, 11, 19),
    date(2024, 11, 1): date(2024, 12, 20),
    date(2024, 12, 1): date(2025, 1, 28),
    date(2025, 1, 1): date(2025, 3, 17),
    date(2025, 2, 1): date(2025, 3, 28),
    date(2025, 3, 1): date(2025, 4, 18),
    date(2025, 4, 1): date(2025, 5, 21),
    date(2025, 5, 1): date(2025, 6, 24),
    date(2025, 6, 1): date(2025, 7, 18),
    date(2025, 7, 1): date(2025, 8, 19),
    date(2025, 8, 1): date(2025, 9, 19),
    # Sep 2025: released Dec 11 (gov shutdown delayed Oct release)
    date(2025, 9, 1): date(2025, 12, 11),
    # Oct 2025: released with Nov (gov shutdown)
    date(2025, 10, 1): date(2025, 12, 16),
}
```

### Implementation notes

- Use `date(YYYY, M, 1)` as the ref_month key for CES and SAE (not the
  12th; the 12th convention is internal to the scraper's `ref_date`).
  Document this choice in the module docstring.
- Add a module-level `BLS_SCHEDULE_URLS` dict for provenance:
  ```python
  BLS_SCHEDULE_URLS = {
      'ces': 'https://www.bls.gov/schedule/news_release/empsit.htm',
      'qcew': 'https://www.bls.gov/schedule/news_release/cew.htm',
      'sae': 'https://www.bls.gov/schedule/news_release/laus.htm',
  }
  ```
- Include a `LAST_SCRAPED: date` constant so we know when the data was
  last refreshed (set to `date(2026, 2, 26)`).
- Add type alias: `ReleaseDateMap = dict[date, date]` for CES/SAE and
  `QCEWReleaseDateMap = dict[tuple[int, int], date]` for QCEW.

### Verification

Write a simple `assert` block (guarded by `if __name__ == '__main__'`)
that checks:
- All CES release dates fall on a Friday (BLS convention for Employment
  Situation), except when holidays force a different day.
- All QCEW release dates are after the end of the reference quarter +
  at least 4 months.
- All SAE release dates are after CES release dates for the same
  reference month (SAE always lags CES by ~2-3 weeks).
- No duplicate reference periods.

---

## Task 2: Extend `PublicationCalendar` to Include SAE

### What exists

`PublicationCalendar` in `revision_schedules.py` currently holds only:
- `ces_release_dates: pl.DataFrame`
- `qcew_release_dates: pl.DataFrame`

### What to change

Add a third field:
```python
sae_release_dates: pl.DataFrame
# Columns: ref_month (Date), revision_number (Int32), pub_date (Date)
```

Update `from_parquet` to also filter for `source == 'sae'` and populate
the new field, following the same pattern as the CES branch.

### Add `from_dicts` classmethod

Add a classmethod that builds the calendar directly from the hard-coded
dicts in Task 1, without requiring a parquet file:

```python
@classmethod
def from_dicts(cls) -> 'PublicationCalendar':
    """Build calendar from hard-coded BLS schedule dates.

    Uses the scraped release dates in publication_dates.py.
    Only populates revision 0 (first-print) dates.  Revision 1+
    dates are computed by adding the structural lag offsets from
    CES_REVISIONS to the revision-0 date.

    Returns
    -------
    PublicationCalendar
    """
```

The logic for CES/SAE revision > 0 dates: take the revision-0 pub_date
and add the *incremental* lag.  For CES:
- revision 0: exact date from dict
- revision 1: revision 0 pub_date + ~30 days (next month's release)
- revision 2: revision 0 pub_date + ~60 days (month after that)

Rather than hard-coding +30/+60, look up the *next* CES release date in
the dict.  For example, if Feb 2025 CES revision-0 is Mar 7, 2025, then
Feb 2025 revision-1 is the CES release date for Mar 2025 (Apr 4, 2025),
and revision-2 is the release date for Apr 2025 (May 2, 2025).  This
correctly handles irregular spacing.

For QCEW, revision > 0 dates follow the annual file structure (see
`QCEW_REVISIONS` in `revision_schedules.py`).  Since the hard-coded
QCEW dates only cover initial releases, revision > 0 dates should fall
back to the lag-based approximation.

### Where to put it

Modify `src/alt_nfp/lookups/revision_schedules.py` in-place.  Import
from the new `publication_dates.py` module.

---

## Task 3: Wire Calendar into Vintage Date Lookups

### What exists

`get_ces_vintage_date(ref_month, revision)` and
`get_qcew_vintage_date(ref_quarter, ref_year, revision)` return
approximate dates using `RevisionSpec.lag_months`.

### What to change

Add an optional `calendar: PublicationCalendar | None = None` parameter
to both functions (and create a new `get_sae_vintage_date`).  When a
calendar is provided and contains an exact date for the requested
`(ref_period, revision)`, return that date.  Otherwise, fall back to
the lag-based approximation.

```python
def get_ces_vintage_date(
    ref_month: date,
    revision: int,
    calendar: PublicationCalendar | None = None,
) -> date:
    """Compute publication date for a CES vintage.

    Parameters
    ----------
    ref_month : date
        Reference month (1st of month).
    revision : int
        Revision number (0 = first print, 1 = second, 2 = third,
        -1 = benchmark).
    calendar : PublicationCalendar | None
        If provided, prefer exact dates over lag-based approximations.

    Returns
    -------
    date
        Publication date (exact if available, approximate otherwise).
    """
```

Same pattern for `get_sae_vintage_date` and `get_qcew_vintage_date`.

### Add module-level default calendar

Add a lazily-initialized module-level calendar so callers don't have
to pass it every time:

```python
_DEFAULT_CALENDAR: PublicationCalendar | None = None


def get_default_calendar() -> PublicationCalendar:
    """Return the default PublicationCalendar, built from hard-coded dates.

    Lazily initialized on first call.
    """
    global _DEFAULT_CALENDAR
    if _DEFAULT_CALENDAR is None:
        _DEFAULT_CALENDAR = PublicationCalendar.from_dicts()
    return _DEFAULT_CALENDAR
```

Update `get_ces_vintage_date` and `get_qcew_vintage_date` to use the
default calendar when `calendar is None`.

---

## Task 4: Build `publication_calendar.parquet`

### What to create

A script `scripts/build_publication_calendar.py` (or a CLI subcommand)
that merges two sources into a single parquet file:

1. **Historical dates** from `bls_estimates/release_dates/` — the
   scraper pipeline covers CES, SAE, and QCEW going back to ~2007.
2. **Forward-looking dates** from the hard-coded dicts in Task 1.

### Output schema

```
source:       Utf8    ('ces', 'qcew', 'sae')
ref_period:   Utf8    ('2025-01' for monthly, '2025Q1' for quarterly)
revision_number: Int32 (0, 1, 2, -1 for benchmark)
publication_date: Date
```

This is the schema `PublicationCalendar.from_parquet` already expects.

### Logic

1. Load `release_dates.parquet` from `bls_estimates` (if available).
2. Convert to the output schema, computing revision numbers via the
   same logic in `vintage_dates.py`.
3. Load the hard-coded dicts from `publication_dates.py`.
4. Merge, preferring the hard-coded dates for overlapping periods
   (they're more carefully vetted).
5. Write to `data/publication_calendar.parquet`.

### Graceful degradation

If `release_dates.parquet` doesn't exist (fresh clone without running
the scraper), the script should still produce a valid parquet from
just the hard-coded dates.  Log a warning that historical coverage is
limited.

---

## Task 5: Add `update_schedule` CLI Command

### What to create

A lightweight command (e.g., `python -m alt_nfp.lookups.update_schedule`)
that fetches the three BLS schedule URLs, parses the HTML tables, and
prints updated Python dict literals that can be pasted into
`publication_dates.py`.

### Why not auto-update the source file

The schedule pages occasionally have anomalies (government shutdowns,
combined releases) that need manual review.  The command should output
candidate updates for human review, not silently modify code.

### HTML parsing

Each BLS schedule page has a table with three columns: Reference Month
(or Quarter), Release Date, Release Time.  Use basic regex or
`html.parser` — no BeautifulSoup dependency.  The existing
`bls_estimates/release_dates/parser.py` has patterns for extracting
dates from BLS HTML that can be adapted.

### Output format

Print to stdout in copy-paste-ready format:

```
# CES — fetched 2026-02-26
# New entries not in publication_dates.py:
    date(2026, 1, 1): date(2026, 2, 6),
    date(2026, 2, 1): date(2026, 3, 6),
```

---

## Task 6: Tests

### Unit tests for `publication_dates.py`

File: `tests/test_publication_dates.py`

- `test_ces_dates_are_after_ref_month` — every CES release date is
  at least 20 days after its reference month.
- `test_sae_lags_ces` — for every month with both CES and SAE dates,
  SAE comes after CES.
- `test_qcew_lags_quarter_end` — every QCEW release date is at least
  120 days after end of reference quarter.
- `test_no_duplicate_ref_periods` — no duplicates in any dict.
- `test_shutdown_entries_present` — Oct 2025 CES and SAE entries exist
  and match the known combined-release dates.

### Unit tests for `PublicationCalendar`

File: `tests/test_publication_calendar.py`

- `test_from_dicts_builds_all_three_sources` — CES, QCEW, SAE
  DataFrames are all non-empty.
- `test_ces_revision_chain` — for a month with known dates, revision 0
  < revision 1 < revision 2.
- `test_exact_beats_approximate` — `get_ces_vintage_date` with a
  calendar returns a different (more precise) date than without.
- `test_fallback_when_missing` — for a ref_month not in the calendar,
  the lag-based approximation is returned.
- `test_roundtrip_parquet` — write calendar to parquet via
  `build_publication_calendar.py`, reload via `from_parquet`, and
  verify dates match.

---

## Dependency Graph

```
Task 1 (hard-code dates)
  └──> Task 2 (extend PublicationCalendar)
         └──> Task 3 (wire into vintage lookups)
         └──> Task 4 (build parquet)
  └──> Task 5 (update CLI, independent)
Task 6 (tests) depends on Tasks 1–4
```

Tasks 1 and 5 can be done in parallel.  Task 2 depends on Task 1.
Tasks 3 and 4 depend on Task 2.  Task 6 depends on Tasks 1–4.