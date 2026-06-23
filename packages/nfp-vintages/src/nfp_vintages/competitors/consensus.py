"""Consensus survey-median competitor (pluggable, Bloomberg-sourced).

Reads the contract file defined in ``specs/bloomberg_consensus.md`` §1, or
returns ``None`` when unconfigured (the staged state — the scoreboard then
renders the consensus column as ``-``). A T-1-only competitor: the street
median locks ~release-eve.

The file carries **both** Total NFP (``industry_code == "00"``) and private
(``"05"``) series, each with a survey ``consensus_mean`` and ``consensus_median``
keyed on ``ref_date`` (the BLS reference month, on the CES ref day — the 12th).
``Consensus`` selects one series (industry code + statistic); Track B scores the
Total median against the assembled Total NFP.
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl

_REQUIRED = ("ownership", "industry_type", "industry_code", "ref_date",
             "release_date", "consensus_mean", "consensus_median")

# No survey_date in the file: the median locks at release-eve, so a value is
# withheld until ``as_of`` reaches ``release_date - _LOCK_LAG`` (no lookahead).
_LOCK_LAG = timedelta(days=1)


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
    # Each (industry_code, ref_date) is one survey snapshot — must be unique so a
    # lookup resolves to a single value (mirrors the store's censored-selection
    # discipline).
    if df.select("industry_code", "ref_date").is_duplicated().any():
        raise ValueError("consensus (industry_code, ref_date) pairs must be unique")
    return df.sort("industry_code", "ref_date")


class Consensus:
    """T-1-only competitor: returns one series' value once it has locked.

    Parameters
    ----------
    table : pl.DataFrame or None
        The loaded consensus file, or ``None`` when unconfigured.
    industry_code : str, default ``"00"``
        Which series to read: ``"00"`` (Total NFP — the Track B default) or
        ``"05"`` (private). Kept as a string so leading zeros are not lost.
    statistic : ``"median"`` or ``"mean"``, default ``"median"``
        Which survey statistic to return.
    """

    name = "consensus"

    def __init__(self, table: pl.DataFrame | None, *, industry_code: str = "00",
                 statistic: str = "median") -> None:
        if statistic not in ("median", "mean"):
            raise ValueError(f"statistic must be 'median' or 'mean', got {statistic!r}")
        self.table = table
        self.industry_code = industry_code
        self._col = f"consensus_{statistic}"

    def predict(self, ref_month: date, *, as_of: date) -> float | None:
        """Return the consensus value for ``ref_month``'s series once it has locked.

        Parameters
        ----------
        ref_month : date
            The reference month being predicted; month-bucketed before lookup so
            the caller's day convention (month-start vs the CES 12th) does not
            matter. The file's ``ref_date`` is likewise bucketed to its month.
        as_of : date
            Censoring date; the value is withheld until ``as_of`` reaches
            release-eve (``release_date - _LOCK_LAG``), so consensus stays a
            T-1-only competitor with no lookahead.

        Returns
        -------
        float or None
            The selected series' value, or ``None`` when no table is loaded, the
            month/series is absent, or it has not locked by ``as_of``.
        """
        if self.table is None:
            return None
        ref_m = date(ref_month.year, ref_month.month, 1)
        row = self.table.filter(
            (pl.col("industry_code") == self.industry_code)
            & (pl.col("ref_date").dt.truncate("1mo") == ref_m)
        )
        if row.height == 0:
            return None
        if as_of < row["release_date"][0] - _LOCK_LAG:  # not locked yet
            return None
        return float(row[self._col][0])


class ImpliedGovernment:
    """T−1-only competitor for the government wedge: Total − Private consensus.

    The street's implied monthly government contribution. Same release-eve lock as
    :class:`Consensus`, so it is ``None`` at t7 and present at t1. ``None`` when no
    table is loaded or the month/series is absent.
    """

    name = "implied_govt"

    def __init__(self, table: pl.DataFrame | None, *, statistic: str = "median") -> None:
        if statistic not in ("median", "mean"):
            raise ValueError(f"statistic must be 'median' or 'mean', got {statistic!r}")
        if table is None:
            self._table = None
        else:
            from ..diagnostics import implied_government_consensus

            self._table = implied_government_consensus(table, statistic=statistic)

    def predict(self, ref_month: date, *, as_of: date) -> float | None:
        """Implied government consensus (Total − Private) for ``ref_month``.

        Returns the value once it has locked (``as_of >= release_date − _LOCK_LAG``),
        else ``None`` — so it is ``None`` at t7 and present at t1. Also ``None`` when no
        table is configured or the month is absent from both consensus series.
        """
        if self._table is None:
            return None
        month = date(ref_month.year, ref_month.month, 1)
        hit = self._table.filter(pl.col("ref_date").dt.truncate("1mo") == month)
        if hit.height == 0:
            return None
        row = hit.row(0, named=True)
        if as_of < row["release_date"] - _LOCK_LAG:
            return None
        return float(row["implied_govt_k"])


__all__ = ["consensus_path", "load_consensus", "Consensus", "ImpliedGovernment"]
