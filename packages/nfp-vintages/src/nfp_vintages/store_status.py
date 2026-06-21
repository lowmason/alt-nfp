"""Read-only store health + knowability report (spec §8).

Built on ``read_vintage_store`` (partition-prune + projection pushdown,
LazyFrame) ONLY — never ``transform_to_panel`` (the expensive growth/censoring
path) and never ``views.py`` (panel-grain, post-transform). Coverage is raw
row presence (no ``employment > 0`` filter) so the Oct-2025 ``-1`` "no print"
sentinel (``first_print.py:79-84``) counts as present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import polars as pl
from nfp_ingest.vintage_store import read_vintage_store
from nfp_lookups.paths import VINTAGE_STORE_PATH, is_canonical_store, is_remote
from nfp_lookups.revision_schedules import get_ces_vintage_date, get_qcew_vintage_date

# (source, seasonally_adjusted) partitions present in the rebuilt store.
_PARTITIONS: tuple[tuple[str, bool], ...] = (
    ("ces", True),
    ("ces", False),
    ("qcew", False),
)


@dataclass(frozen=True)
class PartitionCoverage:
    """Coverage of one ``(source, seasonally_adjusted)`` store partition."""

    source: str
    seasonally_adjusted: bool
    earliest_ref: date | None
    latest_ref: date | None
    row_count: int
    last_capture: date | None
    distinct_vintages: int


@dataclass(frozen=True)
class StoreStatus:
    """The full ``status`` report — header flags, coverage, and alarms."""

    store_uri: str
    is_remote: bool
    is_canonical: bool
    per_partition: list[PartitionCoverage] = field(default_factory=list)
    uncaptured: list[str] = field(default_factory=list)
    missing_months: list[str] = field(default_factory=list)
    corrected: list[str] = field(default_factory=list)


def _partition_coverage(store_path, source: str, sa: bool) -> PartitionCoverage | None:
    """Aggregate one partition via ``read_vintage_store``; None if empty/absent."""
    lf = read_vintage_store(store_path, source=source, seasonally_adjusted=sa)
    agg = lf.select(
        pl.len().alias("row_count"),
        pl.col("ref_date").min().alias("earliest_ref"),
        pl.col("ref_date").max().alias("latest_ref"),
        pl.col("vintage_date").max().alias("last_capture"),
        pl.col("vintage_date").n_unique().alias("distinct_vintages"),
    ).collect()
    row_count = int(agg.item(0, "row_count"))
    if row_count == 0:
        return None
    return PartitionCoverage(
        source=source,
        seasonally_adjusted=sa,
        earliest_ref=agg.item(0, "earliest_ref"),
        latest_ref=agg.item(0, "latest_ref"),
        row_count=row_count,
        last_capture=agg.item(0, "last_capture"),
        distinct_vintages=int(agg.item(0, "distinct_vintages")),
    )


# Known shutdown months (employment -1 sentinel in the store): these are expected
# interior "gaps" caused by BLS shutdown delays, not missing captures.
_KNOWN_SHUTDOWN_MONTHS: frozenset[date] = frozenset({
    date(2025, 10, 1),  # Oct-2025: BLS shutdown delayed Sep+Oct CES prints
})


def _next_month(d: date) -> date:
    """First of the month after *d*."""
    return date(d.year + 1, 1, 1) if d.month == 12 else date(d.year, d.month + 1, 1)


def _ces_uncaptured(latest_ref: date | None, as_of: date) -> list[str]:
    """CES ref-months whose rev0 was published <= as_of but are absent from the store.

    Returns entries in the format ``"ces:<YYYY-MM-DD>"`` (Phase 8 contract).
    The date is the first of the ref-month so Phase 8 can reconstruct it via
    ``date.fromisoformat(ref_token)``.
    """
    if latest_ref is None:
        return []
    out: list[str] = []
    candidate = _next_month(latest_ref)
    while candidate <= as_of:
        try:
            v0 = get_ces_vintage_date(candidate, 0)
        except ValueError:
            # Defensive: rev0 is always in the CES schedule, so this never fires
            # for the current call. It guards the forward walk against an as_of
            # that outruns the known revision calendar — stop rather than crash.
            break
        if v0 <= as_of:
            out.append(f"ces:{candidate.isoformat()}")
        candidate = _next_month(candidate)
    return out


def _qcew_uncaptured(latest_ref: date | None, as_of: date) -> list[str]:
    """QCEW quarters whose rev0 was published <= as_of but are absent from the store.

    Returns entries in the format ``"qcew:<YYYY>-Q<n>"`` (Phase 8 contract).
    Phase 8 detects ``"-Q" in ref_token`` then splits on ``"-Q"`` to recover
    year and quarter number.
    """
    if latest_ref is None:
        return []
    out: list[str] = []
    # Advance to the first month of the quarter after latest_ref's quarter.
    q_start_month = ((latest_ref.month - 1) // 3) * 3 + 1
    year, month = latest_ref.year, q_start_month + 3
    if month > 12:
        year, month = year + 1, month - 12
    while date(year, month, 1) <= as_of:
        q_num = (month - 1) // 3 + 1
        ref_quarter = f"Q{q_num}"
        try:
            v0 = get_qcew_vintage_date(ref_quarter, year, 0)
        except ValueError:
            # Defensive (mirrors _ces_uncaptured): Q1–Q4 rev0 is always scheduled,
            # so this never fires for the current call. It guards the forward walk
            # against an as_of beyond the known calendar — stop rather than crash.
            break
        if v0 <= as_of:
            out.append(f"qcew:{year}-Q{q_num}")
        month += 3
        if month > 12:
            year, month = year + 1, month - 12
    return out


def _missing_headline_months(store_path) -> list[str]:
    """Interior CES-SA ref-month gaps over the headline series (geo 00, ind 00/05).

    Raw row presence (no ``employment > 0`` filter) so a -1 sentinel row counts
    as present and known shutdown months are annotated rather than flagged as errors.
    A month is "present" if either the total (``00``) or private (``05``) headline
    row exists. Returns ``"YYYY-MM"`` strings; shutdown months are suffixed with
    ``" [known-shutdown]"`` instead of a bare flag.
    """
    lf = read_vintage_store(
        store_path,
        source="ces",
        seasonally_adjusted=True,
        geographic_type="national",
        geographic_code="00",
    )
    present = (
        lf.filter(pl.col("industry_code").is_in(["00", "05"]))
        .select(pl.col("ref_date").dt.truncate("1mo"))
        .unique()
        .collect()
        .get_column("ref_date")
        .sort()
        .to_list()
    )
    if len(present) < 2:
        return []
    have = set(present)
    out: list[str] = []
    cursor = present[0]
    last = present[-1]
    while cursor < last:
        cursor = _next_month(cursor)
        if cursor not in have:
            label = f"{cursor:%Y-%m}"
            if cursor in _KNOWN_SHUTDOWN_MONTHS:
                label += " [known-shutdown]"
            out.append(label)
    return out


def format_status(status: StoreStatus) -> str:
    """Render a StoreStatus as a human-readable multi-line report."""
    lines: list[str] = []
    flags = []
    if status.is_remote:
        flags.append("REMOTE")
    else:
        flags.append("LOCAL")
    if status.is_canonical:
        flags.append("CANONICAL")
    lines.append(f"store: {status.store_uri}  [{'/'.join(flags)}]")
    if not status.is_remote:
        # Cause-agnostic: a local store_uri arises either from NFP_STORE_URI being
        # unset (the .env gotcha) OR from an explicit local `--store` — the path is
        # printed above, so warn about the consequence without asserting the cause.
        lines.append(
            "  WARNING: LOCAL FALLBACK — reading a local store, not the canonical "
            "S3 store. (Set NFP_STORE_URI or pass --store s3://… for the remote store.)"
        )

    lines.append("")
    lines.append("coverage (source, seasonally_adjusted):")
    for p in status.per_partition:
        lines.append(
            f"  {p.source:<5} sa={str(p.seasonally_adjusted):<5} "
            f"rows={p.row_count:>8,} "
            f"ref=[{p.earliest_ref}..{p.latest_ref}] "
            f"last_capture={p.last_capture} vintages={p.distinct_vintages}"
        )

    if status.uncaptured:
        lines.append("")
        lines.append("UNCAPTURED (published per calendar, absent from store):")
        lines.extend(f"  {u}" for u in status.uncaptured)

    if status.missing_months:
        lines.append("")
        lines.append("missing headline months (interior gaps):")
        lines.extend(f"  {m}" for m in status.missing_months)

    if status.corrected:
        lines.append("")
        lines.append("CORRECTED-LEVEL (incoming != stored employment):")
        lines.extend(f"  {c}" for c in status.corrected)

    return "\n".join(lines)


def compute_status(
    store_path=VINTAGE_STORE_PATH,
    as_of: date | None = None,
) -> StoreStatus:
    """Read-only coverage + knowability report for the vintage store.

    Reads via ``read_vintage_store`` only. ``as_of`` (default: today) bounds the
    forward UNCAPTURED alarm (Task 7.2). Never calls ``transform_to_panel``.
    """
    if as_of is None:
        as_of = date.today()

    per_partition: list[PartitionCoverage] = []
    for source, sa in _PARTITIONS:
        cov = _partition_coverage(store_path, source, sa)
        if cov is not None:
            per_partition.append(cov)

    # --- Task 7.2: alarm computation ---
    by_key = {(p.source, p.seasonally_adjusted): p for p in per_partition}
    ces_sa = by_key.get(("ces", True))
    qcew_nsa = by_key.get(("qcew", False))

    uncaptured: list[str] = []
    uncaptured.extend(_ces_uncaptured(ces_sa.latest_ref if ces_sa else None, as_of))
    uncaptured.extend(_qcew_uncaptured(qcew_nsa.latest_ref if qcew_nsa else None, as_of))

    missing_months = _missing_headline_months(store_path)

    return StoreStatus(
        store_uri=str(store_path),
        is_remote=is_remote(store_path),
        is_canonical=is_canonical_store(store_path),
        per_partition=per_partition,
        uncaptured=uncaptured,
        missing_months=missing_months,
        # Intentionally inert here: corrected-level drift is detected at capture
        # time (capture.py → _run_update prints CORRECTED-LEVEL lines), not
        # reconstructable from the store after the fact, so the read-only status
        # report carries the field (rendered by format_status when populated by a
        # future persisted-correction-log source) but never fills it itself.
        corrected=[],
    )
