"""Tests for the `snapshot` CLI subcommand — validation-only paths.

IMPORTANT: These tests MUST NOT invoke snapshot_model_data.  They only
exercise validation logic that fires *before* any store I/O.  All tests
use a non-12th --as-of date (or otherwise trigger early errors) so the
typer.BadParameter guard short-circuits before snapshot_model_data is ever
*called* (its lazy import does no store I/O, so reaching the import is safe).
"""

from nfp_vintages.__main__ import app
from typer.testing import CliRunner


def test_snapshot_as_of_must_be_day_12():
    """--as-of on a non-12th day with no --grid-end must be rejected."""
    result = CliRunner().invoke(app, ["snapshot", "--as-of", "2026-01-05"])
    assert result.exit_code != 0
    assert "12" in result.output
