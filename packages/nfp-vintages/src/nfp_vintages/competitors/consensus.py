"""Consensus survey-median competitor (pluggable, Bloomberg-sourced).

Reads the contract file defined in ``specs/bloomberg_consensus.md`` §1, or
returns ``None`` when unconfigured (the staged state — the scoreboard then
renders the consensus column as ``-``). A T-1-only competitor: the street
median locks ~release-eve.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

_REQUIRED = ("ref_month", "consensus_median_change_k", "survey_date",
             "release_date", "source")


def _as_path(p: str | Path | Any) -> Path | Any:
    """Coerce *p* to a Path or UPath.

    S3 URIs are built via the public :func:`nfp_lookups.paths.upath_for` so that
    credentials flow in and ``Path()`` does not mangle ``s3://`` → ``s3:/``.
    """
    s = str(p)
    if s.startswith(("s3://", "s3a://")):
        from nfp_lookups.paths import upath_for  # public credentialed-UPath builder

        return upath_for(s)
    return Path(p)


def consensus_path(path: str | Path | None = None) -> Path | Any:
    """Resolve path -> arg -> ``NFP_CONSENSUS_PATH`` -> COMPETITORS_DIR/consensus.parquet."""
    if path is not None:
        return _as_path(path)
    env = os.environ.get("NFP_CONSENSUS_PATH")
    if env:
        return _as_path(env)
    from nfp_lookups.paths import COMPETITORS_DIR

    return COMPETITORS_DIR / "consensus.parquet"


def load_consensus(path: str | Path | None = None) -> pl.DataFrame | None:
    """Load + validate the consensus file, or ``None`` if it does not exist."""
    from nfp_lookups.paths import storage_options_for

    p = consensus_path(path)
    if not p.exists():
        return None
    df = pl.read_parquet(str(p), storage_options=storage_options_for(p))
    missing = set(_REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"consensus file missing required columns: {sorted(missing)}")
    if df["ref_month"].n_unique() != df.height:
        raise ValueError("consensus ref_month must be unique")
    bad = df.filter(pl.col("survey_date") >= pl.col("release_date"))
    if bad.height:
        raise ValueError("consensus survey_date must precede release_date")
    return df.sort("ref_month")


class Consensus:
    """T-1-only competitor: returns the median once the survey has locked."""

    name = "consensus"

    def __init__(self, table: pl.DataFrame | None) -> None:
        self.table = table

    def predict(self, ref_month: date, *, as_of: date) -> float | None:
        if self.table is None:
            return None
        # The file's ``ref_month`` is month-start (day=1, per
        # ``specs/bloomberg_consensus.md``); the harness keys targets on the model
        # date axis (the CES ref day, the 12th). Month-bucket so the lookup is
        # agnostic to the caller's day convention.
        ref_m = date(ref_month.year, ref_month.month, 1)
        row = self.table.filter(pl.col("ref_month") == ref_m)
        if row.height == 0:
            return None
        survey = row["survey_date"][0]
        if as_of < survey:  # not locked yet (e.g. T-7)
            return None
        return float(row["consensus_median_change_k"][0])


__all__ = ["consensus_path", "load_consensus", "Consensus"]
