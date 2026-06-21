"""A5 competitors: each maps (ref_month, as_of) -> predicted change_k.

Competitors are scored against the first-print target across the T-7 and T-1
regimes (``specs/a5_real_competitors.md``). The protocol is keyed on
ref_month + as_of so the same adapters extend to supersector series later
(B1) without a rebuild.
"""
from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable


@runtime_checkable
class Competitor(Protocol):
    """Structural protocol for an A5 competitor: a ``name`` plus an as-of-aware ``predict``."""

    name: str

    def predict(self, ref_month: date, *, as_of: date) -> float | None:
        """Predicted change_k for ``ref_month`` using only data known by
        ``as_of``; ``None`` if the competitor has no value there."""
        ...


__all__ = ["Competitor"]
