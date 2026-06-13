"""Self-test for the repo-wide live-store safety net.

The root ``conftest.py`` autouse fixture ``_block_live_store`` blanks the
store-access environment for every test NOT marked ``real_store``. This severs
the **call-time** access path: Polars I/O (``read_vintage_store``,
``build_panel``, ``build_model_data``) and all store *writes*
(``build_store``/``append_to_vintage_store``/``compact_partition``) resolve
credentials via ``storage_options_for(path)`` at call time, so blanked env =
no creds = fail closed. That covers the incident class (a ``build_store``
write once destroyed a year of irreplaceable data).

The **object-method** path is now closed too (see
``test_object_method_access_residual``). ``VINTAGE_STORE_PATH`` is an
import-time ``upath.UPath`` constructed with ``key=``/``secret=`` baked in
(``nfp_lookups.paths._store_location``); fsspec binds those creds into the
UPath's filesystem instance, so direct UPath *object methods*
(``.exists``/``.glob``/``.iterdir``) reached the live store after a runtime
``delenv`` — and clearing the fsspec instance cache did not help. The fixture
now severs that surface at a single class-level choke for unmarked tests:
``s3fs.S3FileSystem._call_s3`` (the funnel for every S3 API call) is patched
to raise, and ``_ls_from_cache`` is forced to miss so a dircache entry from a
prior ``real_store`` test cannot serve a stale hit. ``.exists()`` therefore
degrades to ``False`` — the read-only metadata surface used by
``_store_available()`` availability probes no longer touches production.
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


def test_object_method_access_residual():
    """The object-method path is severed: unmarked access cannot reach the store.

    Formerly a documented residual (a strict-xfail tripwire): the creds baked
    into ``VINTAGE_STORE_PATH`` let UPath object methods reach the live store
    despite the env-blanking fixture. ``_block_live_store`` now patches
    ``s3fs.S3FileSystem._call_s3`` (raise) and ``_ls_from_cache`` (force a
    cache miss) for unmarked tests, so ``.exists()`` degrades to ``False``
    instead of hitting production. Skips when the store is local in this
    session (remote-only property).
    """
    from nfp_lookups.paths import VINTAGE_STORE_PATH, is_remote

    if not is_remote(VINTAGE_STORE_PATH):
        pytest.skip("store is local in this session; remote-only property")
    assert (VINTAGE_STORE_PATH / "source=ces").exists() is False


def test_sever_blocks_any_remote_upath():
    """Session-independent proof the class-level s3fs sever fails closed.

    The two tests above gate on ``is_remote(VINTAGE_STORE_PATH)``, which resolves
    *local* in the standard pytest run (a ``load_dotenv`` timing effect) and so
    they skip — vacuous coverage. This test does not depend on the session store:
    the fixture patches ``s3fs.S3FileSystem`` at the *class* level, so ANY s3
    UPath an unmarked test touches must fail closed. ``s3fs`` is a declared
    dependency, so this runs everywhere (incl. CI), giving real regression
    coverage that the object-method path is severed.
    """
    pytest.importorskip("s3fs")
    from upath import UPath

    # Unmarked test → fixture active → _call_s3 raises → .exists() degrades to False.
    assert UPath("s3://alt-nfp/store/source=ces").exists() is False
