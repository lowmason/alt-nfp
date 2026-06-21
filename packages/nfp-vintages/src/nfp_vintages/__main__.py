"""Unified CLI for the vintage-building pipeline.

Usage::

    alt-nfp                          # Run all steps (or: python -m nfp_vintages)
    alt-nfp download                  # Download CES + QCEW revision files
    alt-nfp download-indicators       # Download cyclical indicators from FRED
    alt-nfp process                   # Scrape BLS calendar + process revisions
    alt-nfp current                   # Fetch current BLS estimates
    alt-nfp build                     # Combine + build vintage_store
    alt-nfp build --releases PATH      # Build store using given releases.parquet
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import typer
from dotenv import load_dotenv

app = typer.Typer(help="Vintage-building pipeline for alt-nfp.")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Run all pipeline stages, or a single subcommand."""
    load_dotenv()
    if ctx.invoked_subcommand is None:
        download()
        download_indicators()
        process()
        current()
        build(None, allow_canonical=False)


@app.command()
def download() -> None:
    """Download CES and QCEW data files."""
    from nfp_download.bls.bulk import download_ces, download_qcew, download_qcew_bulk

    print('Downloading CES vintage data...')
    download_ces()
    print('Downloading QCEW revisions CSV...')
    download_qcew()
    print('Downloading QCEW bulk quarterly files (2003-2025)...')
    download_qcew_bulk()
    print('Done.')


@app.command("download-indicators")
def download_indicators() -> None:
    """Download cyclical indicators from FRED into data/indicators/."""
    from nfp_ingest.indicators import download_indicators

    print('Downloading cyclical indicators from FRED...')
    results = download_indicators()
    total = sum(results.values())
    print(f'Done: {total} total rows across {len(results)} indicators.')


@app.command()
def process() -> None:
    """Scrape BLS release calendar, then process CES/QCEW revisions."""
    from nfp_vintages.calendar import advance_release_calendar
    from nfp_vintages.processing.ces_triangular import main as ces_triangular_main
    from nfp_vintages.processing.combine import main as combine_main
    from nfp_vintages.processing.qcew_bulk import main as qcew_main

    print('=== Building BLS release calendar ===')
    advance_release_calendar()

    print('\n=== Processing CES national revisions ===')
    ces_triangular_main()
    print('\n=== Processing QCEW revisions ===')
    qcew_main()
    print('\n=== Combining revisions ===')
    combine_main()


@app.command()
def current() -> None:
    """Fetch current BLS estimates and write releases.parquet."""
    from nfp_ingest.releases import build_releases

    print('=== Fetching current BLS estimates ===')
    build_releases()


@app.command()
def build(
    releases_path: Path | None = typer.Option(
        None,
        "--releases",
        path_type=Path,
        help="Path to releases.parquet (default: use built-in location).",
    ),
    allow_canonical: bool = typer.Option(
        False,
        "--allow-canonical",
        help="Permit overwriting the canonical (production) store in place (DANGEROUS — prefer the plans/10 T8 backup-first cutover).",
    ),
) -> None:
    """Build the Hive-partitioned vintage store."""
    from nfp_vintages.build_store import build_store

    build_store(releases_path=releases_path, allow_canonical=allow_canonical)


@app.command("build-rebuild")
def build_rebuild(
    start_year: int = typer.Option(
        2017,
        "--start-year",
        help="First QCEW reference year to fetch (default 2017, the rebuild scope).",
    ),
    end_year: int | None = typer.Option(
        None,
        "--end-year",
        help="Last QCEW reference year (inclusive; default = current year). "
        "Set --start-year == --end-year for a one-year small-window smoke build.",
    ),
    allow_canonical: bool = typer.Option(
        False,
        "--allow-canonical",
        help=(
            "Permit writing to the canonical store in place "
            "(DANGEROUS — only for explicit maintainer override)."
        ),
    ),
) -> None:
    """Compose CES + QCEW panels into the scratch rebuild store.

    CES is built from the cached ``cesvinall/`` triangular CSVs (no network).
    QCEW levels and size are fetched from the BLS API slices (area + size
    endpoints) over plain httpx — so this command needs network access, and
    writes to ``NFP_STORE_URI`` (the scratch ``s3://alt-nfp/store-rebuild``
    prefix; the canonical store is refused unless ``--allow-canonical``).

    Use ``--start-year 2024 --end-year 2024`` for a one-year small-window smoke
    build (fetch → compose → scratch write) before the full 2017-present run.
    See specs/store_rebuild_acquire.md for the acquire design.
    """
    from nfp_ingest.ces_builder import build_ces_panel
    from nfp_ingest.qcew_acquire import (
        acquire_qcew_levels,
        acquire_qcew_size_native,
    )
    from nfp_ingest.qcew_crosswalk import build_qcew_panel
    from nfp_ingest.size_class import build_size_class_panel

    from nfp_vintages.rebuild_store import (
        compose_rebuild_panel,
        write_rebuild_store,
    )

    print("=== Build-rebuild: composing CES + QCEW panels ===")

    print("Building CES panel from cesvinall/ triangular CSVs...")
    ces = build_ces_panel()
    print(f"  CES: {ces.height:,} rows")

    # Fetch QCEW area-endpoint slices (all quarters) → crosswalk.
    print(f"Acquiring QCEW levels (BLS area API slices, {start_year}-{end_year or 'now'})...")
    raw_qcew = acquire_qcew_levels(start_year=start_year, end_year=end_year)
    qcew_levels = build_qcew_panel(raw_qcew)
    print(f"  QCEW levels: {qcew_levels.height:,} rows")

    # Fetch QCEW Q1 size-endpoint slices (size_code 1-9) → crosswalk.
    # acquire_qcew_size_native already crosswalks to CES codes (it is NOT raw CSV).
    print(f"Acquiring QCEW size native rows (BLS size API slices, {start_year}-{end_year or 'now'})...")
    size_native = acquire_qcew_size_native(start_year=start_year, end_year=end_year)
    size = build_size_class_panel(size_native)
    print(f"  QCEW size: {size.height:,} rows")

    print("Composing panels...")
    panel = compose_rebuild_panel(ces, qcew_levels, size)
    print(f"  Combined: {panel.height:,} rows")

    print("Writing rebuild store...")
    write_rebuild_store(panel, allow_canonical=allow_canonical)
    print("Done.")


def _run_snapshot(as_of: date, grid_end: date | None = None) -> None:
    """Write hash-pinned ModelData snapshot(s); plain helper (no Typer types)."""
    from nfp_ingest.snapshots import snapshot_model_data

    if as_of.day != 12:
        raise ValueError("--as-of must fall on the 12th (day-12 convention)")

    if grid_end is None:
        dates = [as_of]
    else:
        dates = []
        y, m = as_of.year, as_of.month
        while date(y, m, 12) <= grid_end:
            dates.append(date(y, m, 12))
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)

    for d in dates:
        path, digest = snapshot_model_data(d)
        print(f"  {d}: {path} (hash {digest[:12]})")


@app.command()
def status(
    as_of: str | None = typer.Option(
        None, "--as-of", help="Knowability cutoff for the UNCAPTURED alarm (YYYY-MM-DD)."
    ),
    store: str | None = typer.Option(
        None, "--store", help="Override the store URI/path (default: VINTAGE_STORE_PATH)."
    ),
) -> None:
    """Read-only store coverage + 'what's uncaptured' report (spec §8)."""
    from datetime import date as _date

    from nfp_lookups.paths import VINTAGE_STORE_PATH

    from nfp_vintages.store_status import compute_status, format_status

    if store is not None:
        if store.startswith(("s3://", "s3a://")):
            from upath import UPath

            store_path = UPath(store)
        else:
            store_path = Path(store)
    else:
        store_path = VINTAGE_STORE_PATH

    as_of_date = _date.fromisoformat(as_of) if as_of is not None else None
    report = compute_status(store_path, as_of=as_of_date)
    print(format_status(report))


@app.command()
def snapshot(
    as_of: str = typer.Option(..., "--as-of", help="Knowledge cutoff, YYYY-MM-DD (day-12)."),
    grid_end: str | None = typer.Option(
        None, "--grid-end", help="If set, snapshot every month's 12th from --as-of through here."
    ),
) -> None:
    """Write hash-pinned ModelData snapshot(s) for the given as-of date(s)."""
    from datetime import date as _date

    end = _date.fromisoformat(grid_end) if grid_end is not None else None
    try:
        _run_snapshot(_date.fromisoformat(as_of), end)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--as-of") from exc


def _run_update(
    as_of: date,
    *,
    only: str | None = None,
    refresh_calendar: bool = True,
    store_path=None,
) -> None:
    """Capture month-T prints into the store; plain helper (no Typer types)."""
    from nfp_lookups.paths import VINTAGE_STORE_PATH

    store_path = store_path if store_path is not None else VINTAGE_STORE_PATH

    if refresh_calendar:
        from nfp_vintages.calendar import advance_release_calendar

        advance_release_calendar()

    if only in (None, "ces"):
        from nfp_ingest.capture import capture_ces_print

        res = capture_ces_print(as_of, store_path=store_path)
        print(f"  CES: appended {res.appended}, skipped {res.skipped}")
        for c in res.corrected:
            print(f"  CORRECTED-LEVEL ces {c.ref_date} {c.industry_code} "
                  f"rev{c.revision}/bmr{c.benchmark_revision}: "
                  f"{c.stored_employment} -> {c.incoming_employment}")

    if only in (None, "qcew"):
        from nfp_ingest.capture import capture_qcew_quarter

        res = capture_qcew_quarter(as_of, store_path=store_path)
        print(f"  QCEW: appended {res.appended}, skipped {res.skipped}")
        for c in res.corrected:
            print(f"  CORRECTED-LEVEL qcew {c.ref_date} {c.industry_code} "
                  f"rev{c.revision}/bmr{c.benchmark_revision}: "
                  f"{c.stored_employment} -> {c.incoming_employment}")

    if only in (None, "indicators"):
        from nfp_ingest.indicators import download_indicators

        results = download_indicators()
        total = sum(results.values()) if results else 0
        print(f"  Indicators: {total} rows across {len(results or {})} series")

    # --- self-healing compaction (§6.2) -------------------------------------
    # A crash between append_to_vintage_store and compact_partition leaves a
    # partition with >1 fragment. Compact any such partition on the next run —
    # cheap and idempotent (compact is a no-op on a single-file partition).
    # FOLLOW-ON: remote (s3://) self-heal — enumerate partitions via UPath +
    # storage_options_for(store_path); compact_partition already deletes remote
    # fragments. Guarded out here so the local test stays hermetic.
    from nfp_ingest.vintage_store import compact_partition
    from nfp_lookups.paths import is_remote

    if not is_remote(store_path):
        for source_dir in sorted(store_path.glob("source=*")):
            source = source_dir.name.split("=", 1)[1]
            for sa_dir in sorted(source_dir.glob("seasonally_adjusted=*")):
                if len(list(sa_dir.glob("*.parquet"))) > 1:
                    sa = sa_dir.name.split("=", 1)[1] == "true"
                    compact_partition(store_path, source, sa)


@app.command()
def update(
    as_of: str = typer.Option(..., "--as-of", help="Knowledge cutoff, YYYY-MM-DD."),
    only: str | None = typer.Option(
        None, "--only", help="Limit to one source: ces | qcew | indicators."
    ),
    no_refresh_calendar: bool = typer.Option(
        False, "--no-refresh-calendar", help="Skip the release-calendar scrape (assume current)."
    ),
) -> None:
    """Advance the calendar, capture month-T prints, and append them to the store."""
    from datetime import date as _date

    if only is not None and only not in ("ces", "qcew", "indicators"):
        raise typer.BadParameter("must be ces, qcew, or indicators", param_hint="--only")
    _run_update(
        _date.fromisoformat(as_of), only=only, refresh_calendar=not no_refresh_calendar
    )


if __name__ == '__main__':
    app()
