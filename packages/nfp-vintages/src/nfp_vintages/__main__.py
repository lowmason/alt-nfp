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


def _build_release_calendar() -> None:
    """Scrape BLS publication schedule and build release/vintage date parquets.

    Produces ``release_dates.parquet`` and ``vintage_dates.parquet`` in the
    intermediate directory.  Called automatically by :func:`process`.
    """
    import asyncio

    import polars as pl
    from nfp_download.release_dates.config import PUBLICATIONS
    from nfp_download.release_dates.parser import collect_release_dates
    from nfp_download.release_dates.scraper import (
        FetchError,
        ParseError,
        create_session,
        download_all,
        fetch_index,
        parse_index_page,
    )
    from nfp_ingest.release_dates.vintage_dates import (
        SUPPLEMENTAL_RELEASE_DATES,
        build_vintage_dates,
    )
    from nfp_lookups.paths import (
        RELEASE_DATES_PATH,
        RELEASES_DIR,
        VINTAGE_DATES_PATH,
    )

    async def _download_all_publications() -> None:
        async with create_session() as session:
            for pub in PUBLICATIONS:
                print(f'Fetching index for {pub.name}...')
                try:
                    html = await fetch_index(session, pub.index_url)
                except FetchError as e:
                    # Safety net: if BLS's bot detection changes again, the
                    # calendar can still be built from release pages already
                    # on disk; only newly published pages are missed.
                    print(
                        f'  WARNING: index fetch failed for {pub.name} ({e}); '
                        f'using cached release pages only'
                    )
                    continue
                try:
                    entries = parse_index_page(
                        html, pub.name, pub.series, pub.frequency,
                    )
                except ParseError as e:
                    # Page structure may have drifted; fall back to cached
                    # release pages already on disk so the rest of the calendar
                    # build can proceed. Only newly-published pages are missed.
                    print(
                        f'  WARNING: index parse failed for {pub.name} ({e}); '
                        f'using cached release pages only'
                    )
                    continue
                print(f'  Found {len(entries)} releases for {pub.name}')
                try:
                    paths = await download_all(entries, pub.name)
                except FetchError as e:
                    print(
                        f'  WARNING: release download failed for {pub.name} '
                        f'({e}); using cached release pages only'
                    )
                    continue
                print(f'  Downloaded {len(paths)} new files for {pub.name}')

    asyncio.run(_download_all_publications())

    print('Building release_dates...')
    rows = []
    for pub in PUBLICATIONS:
        pub_dir = RELEASES_DIR / pub.name
        if not pub_dir.exists():
            continue
        for row in collect_release_dates(pub.name, pub_dir):
            rows.append(row)

    df = pl.DataFrame(
        rows,
        schema={'publication': pl.Utf8, 'ref_date': pl.Date, 'vintage_date': pl.Date},
        orient='row',
    )
    supplemental = pl.DataFrame(
        [
            {'publication': p, 'ref_date': ref, 'vintage_date': vint}
            for p, ref, vint in SUPPLEMENTAL_RELEASE_DATES
        ],
        schema={'publication': pl.Utf8, 'ref_date': pl.Date, 'vintage_date': pl.Date},
    )
    existing_keys = df.select('publication', 'ref_date').unique()
    supplemental = supplemental.join(
        existing_keys, on=['publication', 'ref_date'], how='anti',
    )
    if supplemental.height > 0:
        df = pl.concat([df, supplemental])
    df = df.sort('publication', 'ref_date')

    RELEASE_DATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(RELEASE_DATES_PATH)
    print(f'Wrote {RELEASE_DATES_PATH} ({len(df)} rows)')

    print('Building vintage_dates...')
    vdf = build_vintage_dates()
    VINTAGE_DATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    vdf.write_parquet(VINTAGE_DATES_PATH)
    print(f'Wrote {VINTAGE_DATES_PATH} ({len(vdf)} rows)')


@app.command()
def process() -> None:
    """Scrape BLS release calendar, then process CES/QCEW revisions."""
    from nfp_vintages.processing.ces_triangular import main as ces_triangular_main
    from nfp_vintages.processing.combine import main as combine_main
    from nfp_vintages.processing.qcew_bulk import main as qcew_main

    print('=== Building BLS release calendar ===')
    _build_release_calendar()

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
        help="Permit rebuilding the canonical store in place (DANGEROUS — destroys live-captured vintage rows).",
    ),
) -> None:
    """Build the Hive-partitioned vintage store."""
    from nfp_vintages.build_store import build_store

    build_store(releases_path=releases_path, allow_canonical=allow_canonical)


@app.command("build-rebuild")
def build_rebuild(
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
    endpoints, 2017-present) via the impersonating client — so this command
    needs network access, and writes to ``NFP_STORE_URI`` (the scratch
    ``s3://alt-nfp/store-rebuild`` prefix; the canonical store is refused unless
    ``--allow-canonical``). See specs/store_rebuild_acquire.md for the acquire
    design.
    """
    from nfp_ingest.ces_builder import build_ces_panel
    from nfp_ingest.qcew_crosswalk import build_qcew_panel
    from nfp_ingest.size_class import build_size_class_panel

    from nfp_vintages.rebuild_store import (
        _acquire_qcew_levels,  # noqa: PLC2701
        _acquire_qcew_size_native,  # noqa: PLC2701
        compose_rebuild_panel,
        write_rebuild_store,
    )

    print("=== Build-rebuild: composing CES + QCEW panels ===")

    print("Building CES panel from cesvinall/ triangular CSVs...")
    ces = build_ces_panel()
    print(f"  CES: {ces.height:,} rows")

    # Fetch QCEW area-endpoint slices (2017-present, all quarters) → crosswalk.
    print("Acquiring QCEW levels (BLS area API slices)...")
    raw_qcew = _acquire_qcew_levels()
    qcew_levels = build_qcew_panel(raw_qcew)
    print(f"  QCEW levels: {qcew_levels.height:,} rows")

    # Fetch QCEW Q1 size-endpoint slices (2017-present, size_code 1-9) → crosswalk.
    # _acquire_qcew_size_native already crosswalks to CES codes (it is NOT raw CSV).
    print("Acquiring QCEW size native rows (BLS size API slices)...")
    size_native = _acquire_qcew_size_native()
    size = build_size_class_panel(size_native)
    print(f"  QCEW size: {size.height:,} rows")

    print("Composing panels...")
    panel = compose_rebuild_panel(ces, qcew_levels, size)
    print(f"  Combined: {panel.height:,} rows")

    print("Writing rebuild store...")
    write_rebuild_store(panel, allow_canonical=allow_canonical)
    print("Done.")


@app.command()
def snapshot(
    as_of: str = typer.Option(
        ..., "--as-of", help="Knowledge cutoff, YYYY-MM-DD (day-12 convention)."
    ),
    grid_end: str | None = typer.Option(
        None,
        "--grid-end",
        help="If set, snapshot every month's 12th from --as-of through this date.",
    ),
) -> None:
    """Write hash-pinned ModelData snapshot(s) for the given as-of date(s)."""
    from datetime import date as _date

    from nfp_ingest.snapshots import snapshot_model_data

    start = _date.fromisoformat(as_of)
    if grid_end is None and start.day != 12:
        raise typer.BadParameter(
            "--as-of must fall on the 12th (day-12 convention)", param_hint="--as-of"
        )
    if grid_end is None:
        dates = [start]
    else:
        end = _date.fromisoformat(grid_end)
        dates = []
        y, m = start.year, start.month
        while _date(y, m, 12) <= end:
            dates.append(_date(y, m, 12))
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)

    for d in dates:
        path, digest = snapshot_model_data(d)
        print(f'  {d}: {path} (hash {digest[:12]})')


if __name__ == '__main__':
    app()
