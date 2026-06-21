"""Production CLI for the alt-nfp vintage store.

Usage::

    alt-nfp update --as-of T [--only ces|qcew|indicators]  # capture knowable prints for T
    alt-nfp status [--as-of T] [--store URI]               # store coverage + uncaptured alarm
    alt-nfp watch [--source ces|qcew|all] [--snapshot]     # feed-driven trigger (cron)
    alt-nfp snapshot --as-of T [--grid-end E]              # hash-pinned ModelData (day-12)

One-time historical load is a SCRIPT, not a command::

    uv run python scripts/bootstrap_store.py --scratch s3://alt-nfp/store-rebuild \\
        --canonical s3://alt-nfp/store

The legacy stage pipeline (download / download-indicators / process / current /
build / build-rebuild and the bare-run chain) was retired in the production-workflow
reshape (specs/cli_production_workflow.md §10). The calendar scrape it used now lives in
nfp_vintages.calendar.advance_release_calendar (invoked by `update`); the rebuild compose/
write moved to scripts/bootstrap_store.py.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import typer
from dotenv import load_dotenv

app = typer.Typer(help="Production vintage-store CLI for alt-nfp.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Load environment config before any command resolves store paths."""
    load_dotenv()


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


def _watch_snapshot_anchor(ref_token: str) -> date:
    """Day-12 anchor for an uncaptured ref token — never the raw pubDate.

    ``ref_token`` is the string after the source prefix in ``StoreStatus.uncaptured``
    (e.g. ``"2025-05-01"`` for CES or ``"2025-Q2"`` for QCEW). Returns the 12th of
    the ref-period's closing month — the convention ``_run_snapshot`` enforces (§4a).

    Parameters
    ----------
    ref_token : str
        ISO ref date (CES, ``"2025-05-01"``) or QCEW quarter token (``"2025-Q2"``).

    Returns
    -------
    date
        The day-12 anchor as a ``datetime.date``.
    """
    if "-Q" in ref_token:
        year_str, q_str = ref_token.split("-Q")
        month = int(q_str) * 3  # Q1→Mar, Q2→Jun, Q3→Sep, Q4→Dec
        return date(int(year_str), month, 12)
    ref = date.fromisoformat(ref_token)
    return date(ref.year, ref.month, 12)


@app.command()
def watch(
    source: str = typer.Option(
        "all", "--source", help="Which feed(s) to poll: all | ces | qcew."
    ),
    snapshot_after: bool = typer.Option(
        False, "--snapshot", help="Also bake a ModelData snapshot for each new release."
    ),
    store: str | None = typer.Option(
        None, "--store", help="Override the store URI/path (default: VINTAGE_STORE_PATH)."
    ),
) -> None:
    """Poll the BLS release feed; trigger ``update`` on a newly-published release.

    Designed for a daily cron. The feed answers only "a release is out *now*" and
    supplies the publication day (``pubDate``); the **store** (via ``compute_status``)
    is the source of truth for which ref-month/quarter is still uncaptured. A clean
    no-op on days with nothing new. A same-day CES + QCEW co-release triggers both.
    """
    from nfp_download.release_dates.feed import (
        CEWQTR_FEED_URL,
        EMPSIT_FEED_URL,
        fetch_feed,
    )
    from nfp_lookups.paths import VINTAGE_STORE_PATH

    from nfp_vintages.store_status import compute_status

    if store is not None:
        if store.startswith(("s3://", "s3a://")):
            from upath import UPath

            store_path = UPath(store)
        else:
            store_path = Path(store)
    else:
        store_path = VINTAGE_STORE_PATH

    _feeds = {"ces": EMPSIT_FEED_URL, "qcew": CEWQTR_FEED_URL}
    if source == "all":
        wanted = ["ces", "qcew"]
    elif source in _feeds:
        wanted = [source]
    else:
        raise typer.BadParameter("must be one of: all, ces, qcew", param_hint="--source")

    for src in wanted:
        items = fetch_feed(_feeds[src])
        if not items:
            print(f"  {src}: feed empty — skipping")
            continue
        # Pick the newest item by pub_date — robust to feed ordering. BLS lists
        # newest-first, but don't depend on it: a non-newest items[0] would drive
        # an older as_of and falsely no-op a genuinely-new release (a missed
        # capture is gone — §9).
        latest = max(items, key=lambda it: it.pub_date)
        pub = latest.pub_date  # already a date object from parse_feed

        # The store decides whether this release's ref-period is captured.
        status = compute_status(store_path, as_of=pub)
        uncaptured = [u for u in status.uncaptured if u.startswith(f"{src}:")]
        if not uncaptured:
            print(f"  {src}: latest release ({pub}) already captured — no-op")
            continue

        # ref_token is the part after "src:" — ISO date (CES) or YYYY-Qn (QCEW).
        ref_token = uncaptured[0].split(":", 1)[1]
        print(f"  {src}: NEW release {pub} (uncaptured {ref_token}) — updating")
        _run_update(pub, only=src, store_path=store_path)

        if snapshot_after:
            anchor = _watch_snapshot_anchor(ref_token)
            print(f"  {src}: snapshot at day-12 anchor {anchor}")
            _run_snapshot(anchor)


if __name__ == '__main__':
    app()
