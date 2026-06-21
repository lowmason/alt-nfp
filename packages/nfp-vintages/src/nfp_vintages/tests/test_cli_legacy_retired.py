"""Legacy CLI retirement: retired commands gone, production surface intact.

Phase 9 of specs/cli_production_workflow.md §10. The bare `alt-nfp` must no
longer chain a store build; the retired stage commands must not be registered.
"""

from __future__ import annotations

from nfp_vintages.__main__ import app
from typer.testing import CliRunner

runner = CliRunner()

_RETIRED = {
    "download",
    "download-indicators",
    "process",
    "current",
    "build",
    "build-rebuild",
}
_KEPT = {"update", "status", "watch", "snapshot"}


def _registered_command_names() -> set[str]:
    names: set[str] = set()
    for cmd in app.registered_commands:
        # Typer derives the CLI name from the function name (underscores -> hyphens)
        # unless an explicit name was passed to @app.command(...).
        names.add(cmd.name or cmd.callback.__name__.replace("_", "-"))
    return names


def test_legacy_commands_are_gone():
    registered = _registered_command_names()
    leaked = _RETIRED & registered
    assert not leaked, f"retired commands still present: {leaked}"


def test_production_commands_present():
    registered = _registered_command_names()
    missing = _KEPT - registered
    assert not missing, f"expected production commands missing: {missing}"


def test_retired_command_invocation_errors():
    result = runner.invoke(app, ["build"])
    assert result.exit_code != 0, "`alt-nfp build` should no longer be a command"


def test_bare_invocation_does_not_run_a_build():
    # No subcommand: the old behavior chained download->...->build. After
    # retirement the bare run must NOT silently rebuild the store; with
    # no_args_is_help it shows usage and exits non-zero.
    result = runner.invoke(app, [])
    assert result.exit_code != 0
    assert "Usage" in result.output
