"""Tests for nfp_lookups.paths — base-dir discovery, store location, layout."""

import os
from pathlib import Path

import pytest
from nfp_lookups import paths
from nfp_lookups.paths import _find_base_dir, _store_location, is_canonical_store


@pytest.fixture(autouse=True)
def _isolate_paths_module():
    """Undo any ``reload(nfp_lookups.paths)`` a test in this module performs.

    Two tests below reload the module to exercise ``data_location()``'s env
    switch. These tests are unmarked, so the root ``conftest._block_live_store``
    fixture has blanked the store creds (``AWS_*``/``NFP_STORE_URI``) for them —
    which means the reload rebuilds the module-level ``VINTAGE_STORE_PATH`` with a
    *blanked-credential* s3fs instance baked in. Left in place, that poisoned
    path leaks to later ``@pytest.mark.real_store`` tests (e.g. ``test_diagnostics``),
    whose dynamic ``from nfp_lookups.paths import VINTAGE_STORE_PATH`` then reads
    an empty store and fails ``assert height > 0``. Snapshot the module dict before
    the test and restore it after, reinstating the real-credential objects (and
    every other env-derived constant) regardless of the cred state at teardown.
    """
    snapshot = dict(paths.__dict__)
    yield
    paths.__dict__.clear()
    paths.__dict__.update(snapshot)


def test_is_canonical_store_matches_canonical_uri():
    assert is_canonical_store("s3://alt-nfp/store") is True
    assert is_canonical_store("s3://alt-nfp/store/") is True
    assert is_canonical_store("s3://alt-nfp/store-rebuild") is False
    assert is_canonical_store("s3://alt-nfp/store-rebuild/") is False
    # s3a:// scheme is handled identically (docstring claims it)
    assert is_canonical_store("s3a://alt-nfp/store") is True
    assert is_canonical_store("s3a://alt-nfp/store-rebuild") is False


def test_is_canonical_store_false_for_local_paths(tmp_path):
    assert is_canonical_store(tmp_path) is False
    assert is_canonical_store(str(tmp_path / "store")) is False


class TestFindBaseDir:
    def test_env_override_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("NFP_BASE_DIR", str(tmp_path))
        assert _find_base_dir() == tmp_path.resolve()

    def test_env_override_expands_user(self, monkeypatch):
        monkeypatch.setenv("NFP_BASE_DIR", "~/somewhere")
        assert _find_base_dir() == (Path.home() / "somewhere").resolve()

    def test_marker_discovery_finds_workspace_root(self, monkeypatch):
        monkeypatch.delenv("NFP_BASE_DIR", raising=False)
        root = _find_base_dir()
        assert (root / "packages").is_dir()
        assert (root / "pyproject.toml").is_file()

    def test_fallback_depth_matches_editable_layout(self):
        # parents[4] above src/nfp_lookups/paths.py is the workspace root,
        # so the fixed-depth fallback agrees with marker discovery.
        assert Path(paths.__file__).resolve().parents[4] == paths.BASE_DIR


class TestDerivedLayout:
    def test_paths_hang_off_base_dir(self):
        assert paths.DATA_DIR == paths.BASE_DIR / "data"
        assert paths.STORE_DIR == paths.DATA_DIR / "store"
        assert paths.DOWNLOADS_DIR == paths.DATA_DIR / "downloads"
        assert paths.INTERMEDIATE_DIR == paths.DATA_DIR / "intermediate"
        assert paths.INDICATORS_DIR == paths.DATA_DIR / "indicators"
        assert paths.OUTPUT_DIR == paths.BASE_DIR / "output"

    # real_store here is the escape hatch, not a store read: VINTAGE_STORE_PATH is
    # an import-time constant derived from session env. The safety net blanks
    # NFP_STORE_URI at runtime, but the constant is already remote — so without
    # this exemption the skip would not fire and the assertion would fail on the
    # stale remote UPath. Keep the session's NFP_STORE_URI so the skip behaves as
    # designed.
    @pytest.mark.real_store
    def test_vintage_store_path_is_store_dir(self):
        if os.environ.get("NFP_STORE_URI"):
            pytest.skip("NFP_STORE_URI set; store is remote in this session")
        assert paths.VINTAGE_STORE_PATH == paths.STORE_DIR

    def test_release_dates_artifacts(self):
        assert paths.RELEASES_DIR == paths.DOWNLOADS_DIR / "releases"
        assert paths.RELEASE_DATES_PATH == paths.INTERMEDIATE_DIR / "release_dates.parquet"
        assert paths.VINTAGE_DATES_PATH == paths.INTERMEDIATE_DIR / "vintage_dates.parquet"


_S3_ENV = {
    "NFP_STORE_URI": "s3://test-bucket/store",
    "AWS_ACCESS_KEY_ID": "test-key",
    "AWS_SECRET_ACCESS_KEY": "test-secret",
    "AWS_ENDPOINT_URL": "http://127.0.0.1:9000",
}


def _set_s3_env(monkeypatch):
    for k, v in _S3_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("AWS_REGION", raising=False)


class TestStoreLocation:
    def test_default_is_local_store_dir(self, monkeypatch):
        monkeypatch.delenv("NFP_STORE_URI", raising=False)
        assert _store_location() == paths.STORE_DIR

    def test_uri_selects_remote_upath(self, monkeypatch):
        _set_s3_env(monkeypatch)
        loc = _store_location()
        assert paths.is_remote(loc)
        assert str(loc) == "s3://test-bucket/store"

    def test_remote_path_joins_preserve_uri(self, monkeypatch):
        _set_s3_env(monkeypatch)
        loc = _store_location()
        joined = loc / "source=ces" / "f.parquet"
        assert str(joined) == "s3://test-bucket/store/source=ces/f.parquet"

    def test_is_remote_false_for_plain_path(self):
        assert not paths.is_remote(Path("/tmp/store"))
        assert not paths.is_remote(paths.STORE_DIR)


def test_data_location_local_when_unset(monkeypatch):
    monkeypatch.delenv("NFP_DATA_URI", raising=False)
    from importlib import reload

    from nfp_lookups import paths

    reload(paths)
    assert paths.data_location() == paths.DATA_DIR
    assert paths.INDICATORS_DIR == paths.DATA_DIR / "indicators"


def test_data_location_remote_when_set(monkeypatch):
    monkeypatch.setenv("NFP_DATA_URI", "s3://alt-nfp")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://127.0.0.1:9000")
    from importlib import reload

    from nfp_lookups import paths

    reload(paths)
    loc = paths.data_location()
    assert paths.is_remote(loc)
    assert str(loc / "indicators").startswith("s3://alt-nfp/indicators")
    # Module state is restored by the autouse _isolate_paths_module fixture.


class TestStorageOptions:
    def test_local_path_returns_none(self):
        assert paths.storage_options_for(Path("/tmp/store")) is None
        assert paths.storage_options_for(paths.STORE_DIR) is None

    def test_remote_http_endpoint(self, monkeypatch):
        _set_s3_env(monkeypatch)
        opts = paths.storage_options_for(_store_location())
        assert opts == {
            "aws_region": "us-east-1",
            "aws_access_key_id": "test-key",
            "aws_secret_access_key": "test-secret",
            "aws_endpoint_url": "http://127.0.0.1:9000",
            "aws_allow_http": "true",
        }

    def test_remote_https_endpoint_omits_allow_http(self, monkeypatch):
        _set_s3_env(monkeypatch)
        monkeypatch.setenv("AWS_ENDPOINT_URL", "https://s3.example.com")
        opts = paths.storage_options_for(_store_location())
        assert "aws_allow_http" not in opts
        assert opts["aws_endpoint_url"] == "https://s3.example.com"

    def test_region_env_respected(self, monkeypatch):
        _set_s3_env(monkeypatch)
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        assert paths.storage_options_for(_store_location())["aws_region"] == "eu-west-1"
