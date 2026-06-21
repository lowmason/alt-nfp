"""§7 guardrail: month-T capture is idempotent and tie-breaks via min(vintage_date).

Characterization tests — the 10-col ukey + min-vintage compact already give these
properties. The first run pins them so a future change to the dedup rules trips here.
"""

from __future__ import annotations

from nfp_ingest.vintage_store import (
    append_to_vintage_store,
    compact_partition,
    read_vintage_store,
)
from nfp_vintages.tests._fixtures import make_ces_rows

_UKEY = [
    "ref_date", "industry_type", "industry_code", "geographic_type",
    "geographic_code", "revision", "benchmark_revision", "ownership",
    "size_class_type", "size_class_code",
]


def _relation(store) -> dict:
    """Map the 10-col dedup ukey -> employment for the (ces, true) partition."""
    df = read_vintage_store(store, source="ces", seasonally_adjusted=True).collect()
    return {
        tuple(r[c] for c in _UKEY): r["employment"]
        for r in df.iter_rows(named=True)
    }


class TestIdempotence:
    def test_capture_append_compact_twice_same_relation(self, tmp_path):
        store = tmp_path / "store"
        rows = make_ces_rows(ref_month="2026-01-12", vintage="2026-02-06")

        append_to_vintage_store(rows, store)
        compact_partition(store, "ces", True)
        first = _relation(store)

        # Second run: identical rows must add 0; compact must be a no-op.
        added = append_to_vintage_store(rows, store)
        compact_partition(store, "ces", True)
        second = _relation(store)

        assert added == 0  # re-append of identical rows is fully deduped
        assert first == second

    def test_same_ukey_later_vintage_keeps_min_vintage_level(self, tmp_path):
        store = tmp_path / "store"
        part = store / "source=ces" / "seasonally_adjusted=true"
        part.mkdir(parents=True)
        early = make_ces_rows(
            ref_month="2026-01-12", vintage="2026-02-06", employment=150_000.0
        )
        late = make_ces_rows(
            ref_month="2026-01-12", vintage="2026-03-06", employment=151_000.0
        )
        # Same 10-col ukey, different vintage_date. Write TWO fragment files
        # directly: append's anti-join would drop `late` (same ukey) before it
        # reached a second file, and compact no-ops on a single file — so a single
        # batch append never exercises the tie-break. Two files is the cross-file
        # state compact's min-vintage rule exists to resolve (§7 landmine).
        early.drop(["source", "seasonally_adjusted"]).write_parquet(part / "a.parquet")
        late.drop(["source", "seasonally_adjusted"]).write_parquet(part / "b.parquet")
        assert len(list(part.glob("*.parquet"))) == 2
        compact_partition(store, "ces", True)
        rel = _relation(store)
        # compact keeps MIN(vintage_date) per ukey → the early real-time level wins
        assert set(rel.values()) == {150_000.0}
