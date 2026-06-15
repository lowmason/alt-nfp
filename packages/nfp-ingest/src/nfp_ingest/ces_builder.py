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
- ``(2,1)`` — the **latest** value in the column: one row per (industry,
  ref-month), the current annual-benchmark basis. Only this single latest value
  is emitted (no per-benchmark history, by design).

Vintage dates come from
:func:`nfp_lookups.revision_schedules.get_ces_vintage_date`, keyed on the day-12
``ref_date`` so the exact BLS calendar is hit. The ``(2,1)`` benchmark date uses
the package convention that the annual benchmark for ref-year ``Y`` publishes
with the January-``Y+1`` first print, i.e.
``get_ces_vintage_date(date(ref_year + 1, 1, 12), 0)`` — identical to the
``benchmark_revision=1`` rows produced by
:func:`nfp_ingest.release_dates.vintage_dates.build_vintage_dates`.

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
        return "total", "total", "00"
    if code == "05":
        return "total", "private", "05"
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
        One row per ``(ref_date, revision, benchmark_revision)`` carrying
        ``employment`` (thousands). For each ref-month column it yields the
        first three non-null values as ``(0,0)``/``(1,0)``/``(2,0)`` and the
        last non-null value as ``(2,1)``.
    """
    vintage_year = tri["year"].to_list()
    vintage_month = tri["month"].to_list()
    # (year, month) → row index, to locate each ref-month's first-print row.
    vintage_idx = {
        (y, m): i
        for i, (y, m) in enumerate(zip(vintage_year, vintage_month, strict=True))
    }

    ref_dates: list[date] = []
    revisions: list[int] = []
    bmrs: list[int] = []
    values: list[float] = []

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

        # Latest non-null value in the column → (revision=2, benchmark_revision=1).
        latest: float | None = None
        for v in column[start:]:
            if v is not None:
                latest = float(v)
        if latest is not None:
            ref_dates.append(ref_date)
            revisions.append(2)
            bmrs.append(1)
            values.append(latest)

    return pl.DataFrame(
        {
            "ref_date": ref_dates,
            "revision": revisions,
            "benchmark_revision": bmrs,
            "employment": values,
        },
        schema={
            "ref_date": pl.Date,
            "revision": pl.UInt8,
            "benchmark_revision": pl.UInt8,
            "employment": pl.Float64,
        },
    )


def _vintage_dates(combos: pl.DataFrame) -> pl.DataFrame:
    """Map distinct ``(ref_date, revision, benchmark_revision)`` to vintage dates.

    ``benchmark_revision=0`` uses ``get_ces_vintage_date(ref_date, revision)``;
    ``benchmark_revision=1`` uses the January-``Y+1`` first-print date (the
    annual-benchmark publication convention).
    """

    def _resolve(s: dict) -> date:
        ref: date = s["ref_date"]
        if s["benchmark_revision"] == 1:
            return get_ces_vintage_date(date(ref.year + 1, 1, 12), 0)
        return get_ces_vintage_date(ref, int(s["revision"]))

    return combos.with_columns(
        vintage_date=pl.struct("ref_date", "revision", "benchmark_revision").map_elements(
            _resolve, return_dtype=pl.Date
        )
    )


def build_ces_panel(cesvinall_dir: Path | None = None) -> pl.DataFrame:
    """Build CES NSA vintage-store rows from the triangular ``cesvinall`` CSVs.

    Parameters
    ----------
    cesvinall_dir : Path or None
        Directory holding ``tri_{code6}_NSA.csv`` files. Defaults to
        :data:`CESVINALL_DIR` (``DOWNLOADS_DIR/ces/cesvinall``).

    Returns
    -------
    pl.DataFrame
        ``VINTAGE_STORE_SCHEMA``-conformant rows (partition cols carried as
        plain ``source``/``seasonally_adjusted`` columns): NSA, ``source='ces'``,
        national geography, null size class, for ref_date ≥ 2017-01-12. Each
        retained industry/ref-month yields ``(0,0)``/``(1,0)``/``(2,0)`` plus a
        single latest ``(2,1)`` row.

    Raises
    ------
    FileNotFoundError
        If *cesvinall_dir* does not exist.
    """
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

    vdates = _vintage_dates(
        allrows.select("ref_date", "revision", "benchmark_revision").unique()
    )
    result = allrows.join(
        vdates, on=["ref_date", "revision", "benchmark_revision"], how="left"
    )

    # Drop benchmark rows whose annual-benchmark publication has not happened
    # yet (the Jan-Y+1 date is a future lag approximation) — mirrors the
    # ``vintage_date <= today`` filter in ``build_vintage_dates`` and keeps the
    # frontier lookahead-safe. The first-basis (0,0)/(1,0)/(2,0) rows are real
    # past prints and are retained.
    today = date.today()
    result = result.filter(
        (pl.col("benchmark_revision") == 0) | (pl.col("vintage_date") <= today)
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
    ).sort("industry_type", "industry_code", "ref_date", "revision", "benchmark_revision")
