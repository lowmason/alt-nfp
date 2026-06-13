"""Tests for panel_to_model_data precondition warning (I-4).

Verifies that passing a frame with a 'benchmark_revision' column containing
values > 0 emits a UserWarning advising callers to use build_panel() or
build_model_data().  PANEL_SCHEMA has no such column, so real panels never
trigger this — it fires only when a raw vintage-store frame is passed by
mistake.
"""

from __future__ import annotations

import warnings
from datetime import date

import polars as pl
import pytest
from nfp_ingest.model_data import panel_to_model_data
from nfp_lookups.schemas import PANEL_SCHEMA

# ---------------------------------------------------------------------------
# Minimal synthetic panel helpers
# ---------------------------------------------------------------------------

def _base_rows(n: int = 6, source: str = "ces_sa", rev: int = 2) -> list[dict]:
    """Build n monthly rows of PANEL_SCHEMA-compatible data."""
    rows = []
    for i in range(n):
        yr = 2023 + (i // 12)
        mo = (i % 12) + 1
        rows.append(
            {
                "period": date(yr, mo, 1),
                "geographic_type": "national",
                "geographic_code": "US",
                "industry_code": "00",
                "industry_level": "domain",
                "source": source,
                "source_type": "official_sa",
                "growth": 0.001 * (i + 1),
                "employment_level": 150_000.0 + i * 100,
                "is_seasonally_adjusted": True,
                "vintage_date": date(yr, mo, 10),
                "revision_number": rev,
                "is_final": True,
                "publication_lag_months": 1,
                "coverage_ratio": None,
            }
        )
    return rows


def _make_panel(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=PANEL_SCHEMA)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPanelToModelDataBenchmarkRevisionWarning:
    """Defensive warning fires when benchmark_revision > 0 is present."""

    def test_warning_fires_when_benchmark_revision_gt_zero(self):
        """Frame with benchmark_revision > 0 should emit UserWarning."""
        rows = _base_rows(6, source="ces_sa", rev=2)
        df = _make_panel(rows)
        # Graft a benchmark_revision column (simulates a raw vintage-store frame)
        df = df.with_columns(pl.lit(1).cast(pl.Int32).alias("benchmark_revision"))

        with pytest.warns(UserWarning, match="benchmark_revision"):
            panel_to_model_data(df, providers=[])

    def test_no_warning_when_benchmark_revision_all_zero(self):
        """benchmark_revision column present but all-zero should NOT warn."""
        rows = _base_rows(6, source="ces_sa", rev=2)
        df = _make_panel(rows)
        df = df.with_columns(pl.lit(0).cast(pl.Int32).alias("benchmark_revision"))

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # Should not raise — benchmark_revision == 0 everywhere
            panel_to_model_data(df, providers=[])

    def test_no_warning_without_benchmark_revision_column(self):
        """Normal PANEL_SCHEMA panel (no benchmark_revision) should NOT warn."""
        rows = _base_rows(6, source="ces_sa", rev=2)
        df = _make_panel(rows)
        assert "benchmark_revision" not in df.columns

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            panel_to_model_data(df, providers=[])

    def test_warning_message_mentions_build_model_data(self):
        """Warning text should guide callers toward build_model_data."""
        rows = _base_rows(6, source="ces_sa", rev=2)
        df = _make_panel(rows).with_columns(
            pl.lit(2).cast(pl.Int32).alias("benchmark_revision")
        )

        with pytest.warns(UserWarning, match="build_model_data"):
            panel_to_model_data(df, providers=[])
