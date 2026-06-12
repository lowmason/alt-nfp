"""Tests for nfp_lookups.paths — base-dir discovery and derived layout."""

from pathlib import Path

from nfp_lookups import paths
from nfp_lookups.paths import _find_base_dir


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

    def test_vintage_store_path_is_store_dir(self):
        assert paths.VINTAGE_STORE_PATH == paths.STORE_DIR

    def test_release_dates_artifacts(self):
        assert paths.RELEASES_DIR == paths.DOWNLOADS_DIR / "releases"
        assert paths.RELEASE_DATES_PATH == paths.INTERMEDIATE_DIR / "release_dates.parquet"
        assert paths.VINTAGE_DATES_PATH == paths.INTERMEDIATE_DIR / "vintage_dates.parquet"
