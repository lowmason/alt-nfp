"""CES triangular-revision builder for the store rebuild (store_rebuild T2).

Parses the BLS CES ``cesvinall`` triangular CSVs (``tri_{code6}_NSA.csv``) into
``VINTAGE_STORE_SCHEMA``-shaped rows for ``source='ces'``, **NSA only**. The
``_SA`` companions are ignored. The function returns a DataFrame and **never**
touches the vintage store.

Triangle layout
---------------
Each file is a triangle whose **rows are release vintages** (``year``/``month``)
and whose **columns are reference months** named ``Mon_YY`` (e.g. ``Jun_23``).
Reading *down* a ref-month column gives that month's print-then-revision history
in publication order. Cells are employment levels in thousands of persons.

Emitted ``(revision, benchmark_revision)`` per ref-month
--------------------------------------------------------
- ``(0,0)`` / ``(1,0)`` / ``(2,0)`` — the first three non-null values down the
  column (first print, second, third), all on the current annual-benchmark
  basis at the time (``benchmark_revision=0``). These are the k=0,1,2 diagonals.
- ``(2,1)`` — **one row per distinct annual-benchmark basis**. Walking the
  ``(Y, 1)`` January-data vintage rows (``Y = ref_year+1, ref_year+2, …`` up to
  the latest release in the triangle), a ``(2,1)`` row is emitted for each ``Y``
  whose restated cell differs from the previously emitted benchmark value (the
  first non-null one is always emitted; later benchmarks that leave the month
  unchanged are skipped). Both the value and the date come from the **same**
  ``(Y, 1)`` benchmark release, so each row is internally consistent and
  lookahead-free.

Vintage dates come from
:func:`nfp_lookups.revision_schedules.get_ces_vintage_date`, keyed on the day-12
``ref_date`` so the exact BLS calendar is hit. Each ``(2,1)`` benchmark date
uses the package convention that the annual benchmark for ref-year ``Y-1``
publishes with the January-``Y`` first print, i.e.
``get_ces_vintage_date(date(Y, 1, 12), 0)`` — the same convention as the
``benchmark_revision=1`` rows produced by
:func:`nfp_ingest.release_dates.vintage_dates.build_vintage_dates`. A frontier
``as_of`` cutoff drops ``(2,1)`` rows whose benchmark release has not yet
happened.

Industry taxonomy
-----------------
Each 6-digit CES code maps via :data:`nfp_lookups.industry.INDUSTRY_MAP` to its
2-digit ``industry_code`` and the rebuilt ``(industry_type, ownership)`` axes
(store_rebuild §3): ``00``→``(total,total)``, ``05``→``(total,private)``,
``06``/``08``→``(domain,private)``, supersectors/sectors → ``private`` at their
level. Sector CES codes are recoded to NAICS (41→42, 42→44, 43→48) so they key
into the taxonomy. Government / total-services codes ``07``/``90``/``91``/``92``/
``93`` are dropped entirely. ``ownership`` is derived through
:func:`nfp_lookups.industry.ownership_for` rather than hardcoded.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
from nfp_lookups.industry import (
    CES_SECTOR_TO_NAICS,
    INDUSTRY_MAP,
    ownership_for,
)
from nfp_lookups.paths import DOWNLOADS_DIR
from nfp_lookups.revision_schedules import get_ces_vintage_date

CESVINALL_DIR = DOWNLOADS_DIR / "ces" / "cesvinall"

# Output column order = VINTAGE_STORE_SCHEMA; used for the empty-input frame.
_OUTPUT_SCHEMA: dict[str, pl.DataType] = {
    "geographic_type": pl.Utf8,
    "geographic_code": pl.Utf8,
    "ownership": pl.Utf8,
    "industry_type": pl.Utf8,
    "industry_code": pl.Utf8,
    "ref_date": pl.Date,
    "vintage_date": pl.Date,
    "revision": pl.UInt8,
    "benchmark_revision": pl.UInt8,
    "employment": pl.Float64,
    "size_class_type": pl.Utf8,
    "size_class_code": pl.Utf8,
    "source": pl.Utf8,
    "seasonally_adjusted": pl.Boolean,
}

# Earliest reference month retained (store_rebuild coverage: 2017+).
_MIN_REF_DATE = date(2017, 1, 12)

# CES 6-digit codes dropped entirely (government + total service-providing).
_DROPPED_CES_CODES: frozenset[str] = frozenset(
    {"070000", "900000", "909100", "909200", "909300"}
)

_MONTH_NAMES = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)
_MONTH_TO_NUM = {name: i + 1 for i, name in enumerate(_MONTH_NAMES)}


def _taxonomy_for(entry) -> tuple[str, str, str]:
    """Map an :class:`IndustryEntry` to ``(industry_type, ownership, code)``.

    Recodes sector CES codes to NAICS (41→42, 42→44, 43→48) and resolves the
    rebuilt axes via :func:`ownership_for`. ``00``/``05`` are the special
    ``total``-type anchors.
    """
    code = entry.industry_code
    if code == "00":
        return "total", ownership_for("total", "00"), "00"
    if code == "05":
        return "total", ownership_for("total", "05"), "05"
    industry_type = entry.industry_type
    if industry_type == "domain":
        # 06 / 08 are private domains in the rebuilt taxonomy.
        return "domain", ownership_for("domain", code), code
    if industry_type == "sector":
        code = CES_SECTOR_TO_NAICS.get(code, code)
    return industry_type, ownership_for(industry_type, code), code


def _diagonals(tri: pl.DataFrame) -> pl.DataFrame:
    """Extract the per-ref-month print history from one triangular frame.

    Parameters
    ----------
    tri : pl.DataFrame
        A raw triangular frame with integer ``year``/``month`` vintage columns
        and ``Mon_YY`` reference-month columns (values cast to ``Float64``).

    Returns
    -------
    pl.DataFrame
        One row per emitted ``(ref_date, revision, benchmark_revision,
        bench_year)`` carrying ``employment`` (thousands). For each ref-month
        column it yields the first three non-null values as
        ``(0,0)``/``(1,0)``/``(2,0)`` (``bench_year=0``, a join-key sentinel —
        these prints don't depend on a benchmark year), plus one ``(2,1)`` row
        **per distinct annual-benchmark basis**: walking the ``(Y, 1)``
        January-data vintage rows for ``Y = ref_year+1, …``, a row is emitted
        for each ``Y`` whose cell differs from the previously emitted benchmark
        value (always the first such ``Y``). ``bench_year`` carries ``Y`` so
        each ``(2,1)`` row gets its own benchmark ``vintage_date`` downstream.
        ``bench_year`` is a non-null ``Int32`` throughout (``0`` for bmr=0) so
        it can serve as a join key — Polars left-joins treat null keys as
        non-matching, which would null out the print vintage dates.
    """
    vintage_year = tri["year"].to_list()
    vintage_month = tri["month"].to_list()
    # (year, month) → row index, to locate each ref-month's first-print row.
    vintage_idx = {
        (y, m): i
        for i, (y, m) in enumerate(zip(vintage_year, vintage_month, strict=True))
    }
    # Latest release year present in the triangle bounds the benchmark walk.
    max_vintage_year = max(vintage_year)

    ref_dates: list[date] = []
    revisions: list[int] = []
    bmrs: list[int] = []
    values: list[float] = []
    bench_years: list[int] = []

    for col in tri.columns:
        if col in ("year", "month"):
            continue
        mon_name, yy = col.split("_")
        ref_month = _MONTH_TO_NUM[mon_name]
        # ASSUMPTION: every retained ``Mon_YY`` column is a 20xx ref-month. The
        # raw triangle includes 1939+ columns (``Jan_39``…); those have no
        # matching 20xx vintage row so they are skipped below, and the 2017+
        # ``_MIN_REF_DATE`` filter drops anything that slips through. Revisit
        # this 2000-offset if coverage ever predates 2000.
        ref_year = 2000 + int(yy)
        column = tri[col].to_list()

        # rev-0 lives in the vintage row whose (year, month) == this ref-month;
        # rev-1/rev-2 are the next two vintage rows down the column.
        start = vintage_idx.get((ref_year, ref_month))
        if start is None:
            continue

        ref_date = date(ref_year, ref_month, 12)

        # First three non-null prints down the column → revisions 0, 1, 2.
        for k in range(3):
            row = start + k
            if row >= len(column):
                break
            v = column[row]
            if v is None:
                break
            ref_dates.append(ref_date)
            revisions.append(k)
            bmrs.append(0)
            values.append(float(v))
            bench_years.append(0)  # sentinel: prints carry no benchmark year

        # One (2,1) row per distinct benchmark basis: walk the (Y, 1)
        # January-data vintage rows for Y = ref_year+1, …, emitting whenever
        # the restated value differs from the previously emitted benchmark
        # value (always emitting the first non-null one).
        prev_bench: float | None = None
        for y in range(ref_year + 1, max_vintage_year + 1):
            row = vintage_idx.get((y, 1))
            if row is None:
                continue
            v = column[row]
            if v is None:
                continue
            v = float(v)
            if prev_bench is not None and v == prev_bench:
                continue
            ref_dates.append(ref_date)
            revisions.append(2)
            bmrs.append(1)
            values.append(v)
            bench_years.append(y)
            prev_bench = v

    return pl.DataFrame(
        {
            "ref_date": ref_dates,
            "revision": revisions,
            "benchmark_revision": bmrs,
            "employment": values,
            "bench_year": bench_years,
        },
        schema={
            "ref_date": pl.Date,
            "revision": pl.UInt8,
            "benchmark_revision": pl.UInt8,
            "employment": pl.Float64,
            "bench_year": pl.Int32,
        },
    )


def _vintage_dates(combos: pl.DataFrame) -> pl.DataFrame:
    """Map distinct ``(ref_date, revision, benchmark_revision, bench_year)`` to dates.

    ``benchmark_revision=0`` uses ``get_ces_vintage_date(ref_date, revision)``;
    ``benchmark_revision=1`` uses the January-``bench_year`` first-print date —
    the release that published the ``bench_year-1`` annual benchmark. Keying on
    ``bench_year`` (rather than a single ``ref.year + 1``) is what gives each
    per-benchmark ``(2,1)`` row of one ref-month its own, consistent vintage
    date instead of collapsing them to a lookahead-prone first-benchmark date.
    """

    def _resolve(s: dict) -> date:
        ref: date = s["ref_date"]
        if s["benchmark_revision"] == 1:
            return get_ces_vintage_date(date(int(s["bench_year"]), 1, 12), 0)
        return get_ces_vintage_date(ref, int(s["revision"]))

    return combos.with_columns(
        vintage_date=pl.struct(
            "ref_date", "revision", "benchmark_revision", "bench_year"
        ).map_elements(_resolve, return_dtype=pl.Date)
    )


def build_ces_panel(
    cesvinall_dir: Path | None = None, *, as_of: date | None = None
) -> pl.DataFrame:
    """Build CES NSA vintage-store rows from the triangular ``cesvinall`` CSVs.

    Parameters
    ----------
    cesvinall_dir : Path or None
        Directory holding ``tri_{code6}_NSA.csv`` files. Defaults to
        :data:`CESVINALL_DIR` (``DOWNLOADS_DIR/ces/cesvinall``).
    as_of : datetime.date or None
        Frontier cutoff for benchmark ``(2,1)`` rows: only those whose
        benchmark ``vintage_date`` is ``<= as_of`` are retained (the first-basis
        ``(0,0)``/``(1,0)``/``(2,0)`` prints are always kept). Defaults to
        ``date.today()`` evaluated once at entry; passing an explicit ``as_of``
        makes the result deterministic and unit-testable.

    Returns
    -------
    pl.DataFrame
        ``VINTAGE_STORE_SCHEMA``-conformant rows (partition cols carried as
        plain ``source``/``seasonally_adjusted`` columns): NSA, ``source='ces'``,
        national geography, null size class, for ref_date ≥ 2017-01-12. Each
        retained industry/ref-month yields ``(0,0)``/``(1,0)``/``(2,0)`` plus one
        ``(2,1)`` row **per distinct annual-benchmark basis** (value and date
        both drawn from the same ``(Y, 1)`` benchmark release — no lookahead).

    Raises
    ------
    FileNotFoundError
        If *cesvinall_dir* does not exist.
    """
    if as_of is None:
        as_of = date.today()

    path = cesvinall_dir or CESVINALL_DIR
    if not path.exists():
        raise FileNotFoundError(f"cesvinall directory not found: {path}")

    # NSA only — ignore _SA companions.
    entries_by_code = {e.ces_code: e for e in INDUSTRY_MAP}

    parts: list[pl.DataFrame] = []
    for csv_path in sorted(path.glob("tri_*_NSA.csv")):
        code6 = csv_path.stem[len("tri_") : -len("_NSA")]
        if code6 in _DROPPED_CES_CODES:
            continue
        entry = entries_by_code.get(code6)
        if entry is None:
            continue  # unknown code (not in the canonical industry map)

        industry_type, ownership, industry_code = _taxonomy_for(entry)

        # ``year``/``month`` are integer vintage keys; ref-month columns are
        # mixed text in the raw files, so coerce every non-key column to Float64.
        # Read only the header first (all Utf8) to discover columns without
        # tripping type inference on a float-in-an-int-looking column.
        header = pl.read_csv(csv_path, n_rows=0, infer_schema_length=0).columns
        schema_overrides: dict[str, pl.DataType] = {
            c: pl.Float64 for c in header if c not in ("year", "month")
        }
        schema_overrides.update({"year": pl.Int32, "month": pl.Int32})
        tri = pl.read_csv(csv_path, schema_overrides=schema_overrides)

        diag = _diagonals(tri).filter(pl.col("ref_date") >= _MIN_REF_DATE)
        if diag.height == 0:
            continue

        parts.append(
            diag.with_columns(
                industry_type=pl.lit(industry_type, pl.Utf8),
                industry_code=pl.lit(industry_code, pl.Utf8),
                ownership=pl.lit(ownership, pl.Utf8),
            )
        )

    if not parts:
        return pl.DataFrame(schema=_OUTPUT_SCHEMA)

    allrows = pl.concat(parts)

    # ``bench_year`` distinguishes per-benchmark (2,1) rows of one ref-month, so
    # it must be part of the dedup/join key — otherwise distinct benchmark bases
    # collapse to one combo and every (2,1) row is stamped with the first
    # benchmark's date (a lookahead mis-stamp).
    vdates = _vintage_dates(
        allrows.select(
            "ref_date", "revision", "benchmark_revision", "bench_year"
        ).unique()
    )
    result = allrows.join(
        vdates,
        on=["ref_date", "revision", "benchmark_revision", "bench_year"],
        how="left",
    )

    # Drop benchmark rows whose annual-benchmark publication post-dates ``as_of``
    # (not yet released as of the frontier) — keeps the frontier lookahead-safe.
    # The first-basis (0,0)/(1,0)/(2,0) rows are real past prints and retained.
    result = result.filter(
        (pl.col("benchmark_revision") == 0) | (pl.col("vintage_date") <= as_of)
    )

    return result.select(
        geographic_type=pl.lit("national", pl.Utf8),
        geographic_code=pl.lit("00", pl.Utf8),
        ownership=pl.col("ownership"),
        industry_type=pl.col("industry_type"),
        industry_code=pl.col("industry_code"),
        ref_date=pl.col("ref_date"),
        vintage_date=pl.col("vintage_date"),
        revision=pl.col("revision").cast(pl.UInt8),
        benchmark_revision=pl.col("benchmark_revision").cast(pl.UInt8),
        employment=pl.col("employment"),
        size_class_type=pl.lit(None, pl.Utf8),
        size_class_code=pl.lit(None, pl.Utf8),
        source=pl.lit("ces", pl.Utf8),
        seasonally_adjusted=pl.lit(False, pl.Boolean),
    ).sort(
        "industry_type",
        "industry_code",
        "ref_date",
        "revision",
        "benchmark_revision",
        # Per-benchmark (2,1) rows share every other sort key; vintage_date
        # disambiguates them so output ordering is deterministic.
        "vintage_date",
    )
