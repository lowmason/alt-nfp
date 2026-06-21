"""Release-calendar advance — scrape BLS schedule, build the vintage calendar.

Lifted from ``nfp_vintages.__main__._build_release_calendar`` (spec §5.0). The
public ``advance_release_calendar`` is the §5.0 production dependency: ``update``
advances ``vintage_dates.parquet`` to the as-of cutoff before every capture, and
the bootstrap script reuses it. The scrape degrades gracefully on a BLS 403 by
falling back to cached release pages already on disk.

The release/vintage parquet writes thread ``storage_options_for`` + an
``is_remote`` mkdir guard + ``str(path)`` so they work against the S3 data store
on Bloomberg as well as the local ``data/`` fallback (plans/15 container contract).
"""

from __future__ import annotations


def advance_release_calendar() -> None:
    """Scrape the BLS publication schedule and build release/vintage parquets.

    Produces ``release_dates.parquet`` and ``vintage_dates.parquet`` in the
    intermediate directory. On a BLS 403/parse drift the per-publication scrape
    is skipped and the calendar is built from cached release pages on disk; only
    newly-published pages are missed.
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
        is_remote,
        storage_options_for,
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

    if not is_remote(RELEASE_DATES_PATH):
        RELEASE_DATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(
        str(RELEASE_DATES_PATH), storage_options=storage_options_for(RELEASE_DATES_PATH)
    )
    print(f'Wrote {RELEASE_DATES_PATH} ({len(df)} rows)')

    print('Building vintage_dates...')
    vdf = build_vintage_dates()
    if not is_remote(VINTAGE_DATES_PATH):
        VINTAGE_DATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    vdf.write_parquet(
        str(VINTAGE_DATES_PATH), storage_options=storage_options_for(VINTAGE_DATES_PATH)
    )
    print(f'Wrote {VINTAGE_DATES_PATH} ({len(vdf)} rows)')
