# packages/nfp-ingest/src/nfp_ingest/first_print.py
"""A5 first-print target: the within-release headline change BLS announces.

Option A from ``specs/ces_growth_convention.md`` §5 — an *additive* read over
store **levels**. Touches no golden-mastered path: it computes a new derived
series and never alters the panel ``growth`` column or any selection logic.

The first print for reference month ``p`` is

    change_k(p) = L(p, rev0, bmr0) − L(p−1, partner)

with ``partner`` = the prior month's second print ``(rev1, bmr0)`` as
published alongside ``p``'s first print; at benchmark months where that row
is absent/shadowed, fall back to the prior month's latest published level
(highest ``(benchmark_revision, revision, vintage_date)``). CES employment is
in thousands, so the level difference is ``change_k`` directly.

Rebuilt-store accommodations: a release's revisions carry schedule-derived
``vintage_date`` stamps staggered across the release week, so the partner is
censored to ``fp_vintage`` widened by ``_RELEASE_WINDOW_DAYS`` rather than a
strict ``<=`` (still no lookahead — see the constant); and shutdown-skipped
release slots carry a ``-1`` "no print" sentinel that is dropped, leaving those
months unscorable (an absent or null change).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
from nfp_lookups.paths import VINTAGE_STORE_PATH

from nfp_ingest.vintage_store import read_vintage_store

# Schedule-derived ``vintage_date`` drift in the rebuilt store: a single logical
# release is stamped on staggered days across its release week — e.g. the "August
# release" carries Jul rev0 on 2025-08-01, Jun rev1 on 2025-08-03 and May rev2 on
# 2025-08-06. The co-released prior-month rev1 (the partner) can therefore land a
# few days *after* this month's first print, so a day-granular
# ``vintage_date <= fp_vintage`` censor wrongly drops it and collapses the partner
# onto a month-old print. Widen the censor by one release window so the
# co-released partner survives. The window must exceed the largest within-release
# stagger (~7 days: Nov rev1 2026-01-16 vs Dec rev0 2026-01-09) and stay well
# below the smallest inter-release gap (~24 days during the 2025 shutdown), so it
# admits the logically-simultaneous co-release and nothing genuinely later.
#
# This is NOT lookahead: ``first_print`` defines the A5 scoring *target*, not part
# of ``build_model_data``'s as-of input censoring — there is no knowability
# contamination. The co-released rev1 is published in the same release as the
# first print; only the synthetic schedule-derived stamp drifted it later.
_RELEASE_WINDOW_DAYS = 15


def first_print_changes(
    *,
    store_path: Path = VINTAGE_STORE_PATH,
    geographic_type: str = "national",
    geographic_code: str = "00",
    industry_type: str = "total",
    industry_code: str = "00",
) -> pl.DataFrame:
    """Per reference month: the first-print headline change and growth.

    Returns a DataFrame sorted by ``ref_date`` with columns
    ``ref_date``, ``first_print_growth``, ``first_print_change_k``,
    ``vintage_date`` (the first-print release date). Months with no partner
    (history edge) get null growth/change.
    """
    levels = (
        read_vintage_store(
            store_path,
            source="ces",
            seasonally_adjusted=True,
            geographic_type=geographic_type,
            geographic_code=geographic_code,
            industry_type=industry_type,
            industry_code=industry_code,
        )
        .select("ref_date", "vintage_date", "revision", "benchmark_revision", "employment")
        # Drop the -1.0 "no print" sentinel the rebuilt store writes for
        # shutdown-skipped release slots (e.g. Oct-2025 rev0): an invalid level,
        # not a real employment count. Dropping (rather than nulling) lets the
        # fallback below recover a valid earlier revision when only a later one
        # is sentinel.
        .filter(pl.col("employment") > 0)
        .with_columns(pl.col("ref_date").dt.truncate("1mo").alias("period"))
        .collect()
    )

    # Each month's first print: rev0/bmr0 (one vintage per month in practice;
    # take the latest defensively). ``fp_vintage`` is the release these levels
    # were published in — used below to censor the partner to this first print's
    # release (± one release window) so we never use a genuinely later prior-month
    # revision (no lookahead).
    first = (
        levels.filter((pl.col("revision") == 0) & (pl.col("benchmark_revision") == 0))
        .sort("vintage_date")
        .group_by("period")
        .last()
        .select(
            "period",
            pl.col("employment").alias("L_p"),
            pl.col("vintage_date").alias("fp_vintage"),
        )
        .with_columns(prev_period=pl.col("period").dt.offset_by("-1mo"))
    )

    # Prior-month level rows as known by the first print's release, widened by one
    # release window (see ``_RELEASE_WINDOW_DAYS``) so the co-released rev1 — which
    # the schedule-derived stamp may drift a few days past ``fp_vintage`` — is not
    # dropped. After this join ``period`` is the first-print month (the right
    # frame's ``period`` is consumed as the join key).
    cand = (
        first.select("period", "fp_vintage", "prev_period")
        .join(levels, left_on="prev_period", right_on="period", how="left")
        .filter(
            pl.col("vintage_date") <= pl.col("fp_vintage").dt.offset_by(f"{_RELEASE_WINDOW_DAYS}d")
        )
    )

    # Primary partner: prior month's original second print (rev1, bmr0).
    # ``.first()`` after sorting by vintage takes the earliest (original) print,
    # not a later correction — the level as published alongside p's first print.
    primary = (
        cand.filter((pl.col("revision") == 1) & (pl.col("benchmark_revision") == 0))
        .sort("vintage_date")
        .group_by("period")
        .first()
        .select("period", pl.col("employment").alias("L_prev_primary"))
    )

    # Fallback (benchmark months, where the rev1 partner is absent/shadowed):
    # prior month's latest published level known by p's release.
    fallback = (
        cand.sort("benchmark_revision", "revision", "vintage_date")
        .group_by("period")
        .last()
        .select("period", pl.col("employment").alias("L_prev_fallback"))
    )

    out = (
        first.join(primary, on="period", how="left")
        .join(fallback, on="period", how="left")
        .with_columns(L_prev=pl.coalesce("L_prev_primary", "L_prev_fallback"))
        .with_columns(
            first_print_change_k=(pl.col("L_p") - pl.col("L_prev")),
            first_print_growth=(pl.col("L_p").log() - pl.col("L_prev").log()),
        )
        .select(
            pl.col("period").alias("ref_date"),
            "first_print_growth",
            "first_print_change_k",
            pl.col("fp_vintage").alias("vintage_date"),
        )
        .sort("ref_date")
    )
    return out


__all__ = ["first_print_changes"]
