"""Naive baseline competitors: sanity floors, never gates."""
from __future__ import annotations

from datetime import date

import polars as pl


def _known(history: pl.DataFrame, as_of: date) -> pl.DataFrame:
    """First-print rows released on or before ``as_of``, oldest->newest."""
    return (
        history.filter(pl.col("vintage_date") <= as_of)
        .drop_nulls("first_print_change_k")
        .sort("ref_date")
    )


class RandomWalk:
    """Predict the last published first-print change."""

    name = "naive_rw"

    def __init__(self, history: pl.DataFrame) -> None:
        self.history = history

    def predict(self, ref_month: date, *, as_of: date) -> float | None:
        known = _known(self.history, as_of)
        if known.height == 0:
            return None
        return float(known["first_print_change_k"][-1])


class TrailingMean:
    """Predict the mean of the last ``window`` published first-print changes."""

    name = "naive_mean"

    def __init__(self, history: pl.DataFrame, window: int = 12) -> None:
        self.history = history
        self.window = window

    def predict(self, ref_month: date, *, as_of: date) -> float | None:
        known = _known(self.history, as_of)
        if known.height == 0:
            return None
        tail = known["first_print_change_k"][-self.window :]
        return float(tail.mean())


__all__ = ["RandomWalk", "TrailingMean"]
