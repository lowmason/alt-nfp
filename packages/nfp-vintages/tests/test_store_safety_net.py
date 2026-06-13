"""Self-test for the repo-wide live-store safety net.

The root ``conftest.py`` autouse fixture ``_block_live_store`` blanks the
store-access environment for every test NOT marked ``real_store``. This severs
the **call-time** access path: Polars I/O (``read_vintage_store``,
``build_panel``, ``build_model_data``) and all store *writes*
(``build_store``/``append_to_vintage_store``/``compact_partition``) resolve
credentials via ``storage_options_for(path)`` at call time, so blanked env =
no creds = fail closed. That covers the incident class (a ``build_store``
write once destroyed a year of irreplaceable data).

KNOWN RESIDUAL (see ``test_object_method_access_residual``): ``VINTAGE_STORE_PATH``
is an import-time ``upath.UPath`` constructed with ``key=``/``secret=`` baked
in (``nfp_lookups.paths._store_location``). fsspec binds those creds into the
UPath's filesystem instance, so direct UPath *object methods*
(``.exists``/``.glob``/``.iterdir``) keep reaching the live store after a
runtime ``delenv`` — clearing the fsspec instance cache does not help. This is
read-only metadata access (existence/partition listing), not the data path and
not writes; it is the surface used by ``_store_available()`` availability
probes. Closing it requires a design change beyond the delenv fixture (e.g. a
single s3fs choke point, or re-pointing the constant on every binding module +
the captured function defaults) and is left as a documented gap.
"""

import os

import pytest


def test_unmarked_test_has_creds_blanked():
    """An unmarked test cannot see live-store credentials (env-level)."""
    assert os.environ.get("AWS_ACCESS_KEY_ID") is None
    assert os.environ.get("NFP_STORE_URI") is None


@pytest.mark.real_store
def test_real_store_marked_test_keeps_creds():
    """A real_store-marked test keeps whatever creds the session has.

    CI runs with no .env, so the var may be absent entirely — nothing to
    blank, nothing to assert. When present (a dev session with .env), it must
    survive: the marker exempts this test from the blanking fixture.
    """
    if "AWS_ACCESS_KEY_ID" not in os.environ:
        pytest.skip("no AWS_ACCESS_KEY_ID in session env (e.g. CI); nothing to assert")
    assert os.environ.get("AWS_ACCESS_KEY_ID") is not None


def test_unmarked_polars_read_fails_closed():
    """The data path is severed: an unmarked Polars read cannot reach the store.

    Skips when the store is local in this session (CI/local fallback) — the
    property is only meaningful when ``VINTAGE_STORE_PATH`` is a remote UPath.
    """
    from nfp_lookups.paths import VINTAGE_STORE_PATH, is_remote

    if not is_remote(VINTAGE_STORE_PATH):
        pytest.skip("store is local in this session; remote-only property")

    from nfp_ingest.vintage_store import read_vintage_store

    with pytest.raises(Exception):  # noqa: B017 - object-store OSError, no creds
        read_vintage_store(source="ces", seasonally_adjusted=True).collect()


@pytest.mark.xfail(
    strict=True,
    reason="KNOWN RESIDUAL: UPath object methods use creds baked into the "
    "import-time constant; delenv does not sever them. Strict-xfail flips to a "
    "failure the day this is closed, as a built-in reminder.",
)
def test_object_method_access_residual():
    """Documents the residual: unmarked object-method access still reaches the store.

    When closed, this should assert ``... .exists() is False`` and the
    ``strict=True`` xfail will turn the resulting xpass into a failure,
    prompting removal of the xfail marker.
    """
    from nfp_lookups.paths import VINTAGE_STORE_PATH, is_remote

    if not is_remote(VINTAGE_STORE_PATH):
        pytest.skip("store is local in this session; remote-only property")
    # The secure target is `is False`; today it is True (the residual), so this
    # assertion fails -> xfail. The day the net severs object access it passes
    # -> strict xpass -> failure -> remove the marker.
    assert (VINTAGE_STORE_PATH / "source=ces").exists() is False
