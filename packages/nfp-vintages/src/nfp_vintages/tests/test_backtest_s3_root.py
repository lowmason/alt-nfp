"""Regression tests — s3://-capable output root for run_a5/a4_backtest (plans/15 T11).

Two coverage goals:
  (a) output_root() returns a plain pathlib.Path for a local arg and a non-Path
      object whose str() starts with the s3:// URI for a remote arg.
  (b) _np_savez/_np_load go through path.open(), not np.savez(path), so they
      work on any PathLike (local or UPath); and the parquet write receives a
      plain str (not a UPath object — Polars would fail: 'Object does not have
      a .read() method').

No network I/O: all remote paths are fake PathLike stand-ins.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
from nfp_lookups.paths import BASE_DIR

# ---------------------------------------------------------------------------
# Script loader (scripts/ is not on pytest testpaths)
# ---------------------------------------------------------------------------


def _load_a5():
    """Import scripts/run_a5_backtest.py by path."""
    path = BASE_DIR / "scripts" / "run_a5_backtest.py"
    spec = importlib.util.spec_from_file_location("run_a5_backtest", path)
    assert spec and spec.loader, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fake remote path helpers (no real s3; mirrors TestRemotePathStringified)
# ---------------------------------------------------------------------------


class FakeRemoteFile:
    """os.PathLike whose str() is an s3:// URI but is NOT a pathlib.Path.

    .open(mode) delegates to a real BytesIO / tempfile so round-trips work.
    """

    def __init__(self, uri: str, real_file: Path):
        self._uri = uri
        self._real = real_file

    def __fspath__(self) -> str:
        return self._uri

    def __str__(self) -> str:
        return self._uri

    def exists(self) -> bool:
        return self._real.exists()

    def open(self, mode: str = "rb") -> Any:
        return self._real.open(mode)


# ---------------------------------------------------------------------------
# (a) output_root() type contract
# ---------------------------------------------------------------------------


class TestOutputRoot:
    def test_local_path_returns_pathlib_path(self, tmp_path):
        from nfp_lookups.paths import output_root

        result = output_root(str(tmp_path))
        assert isinstance(result, Path), f"expected Path, got {type(result)}"

    def test_s3_uri_returns_non_path_upath(self):
        from nfp_lookups.paths import output_root

        result = output_root("s3://alt-nfp/backtests/x")
        assert not isinstance(result, Path), (
            "output_root('s3://...') must return a UPath, not a pathlib.Path "
            "(pathlib.Path would mangle the double-slash)"
        )
        assert str(result).startswith("s3://alt-nfp/backtests/x"), (
            f"str() of result must preserve the s3:// URI; got {str(result)!r}"
        )

    def test_s3a_uri_returns_non_path_upath(self):
        from nfp_lookups.paths import output_root

        result = output_root("s3a://alt-nfp/backtests")
        assert not isinstance(result, Path)
        assert str(result).startswith("s3a://alt-nfp/backtests")


# ---------------------------------------------------------------------------
# (b) _np_savez/_np_load go through path.open(); parquet write passes str()
# ---------------------------------------------------------------------------


class TestNpHelpers:
    """Verify _np_savez and _np_load round-trip through path.open(), not np.savez(path).

    If they called np.savez(remote_path) directly, numpy would try to open the
    s3:// string as a local filename (appending .npz) and fail. The FakeRemoteFile
    whose .open() backs a real temp file proves the helpers respect the interface.
    """

    def _load_helpers(self):
        mod = _load_a5()
        return mod._np_savez, mod._np_load

    def test_round_trip_through_path_open(self, tmp_path):
        _np_savez, _np_load = self._load_helpers()

        real_file = tmp_path / "test.npz"
        fake = FakeRemoteFile("s3://alt-nfp/backtests/test.npz", real_file)

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10, 20])

        _np_savez(fake, a=a, b=b)
        assert real_file.exists(), "_np_savez must write to the backing real file via .open('wb')"

        result = _np_load(fake)
        np.testing.assert_array_equal(result["a"], a)
        np.testing.assert_array_equal(result["b"], b)

    def test_np_load_returns_plain_dict(self, tmp_path):
        _np_savez, _np_load = self._load_helpers()

        real_file = tmp_path / "test2.npz"
        fake = FakeRemoteFile("s3://alt-nfp/backtests/test2.npz", real_file)
        _np_savez(fake, x=np.array([7.0]))
        result = _np_load(fake)
        assert isinstance(result, dict), f"_np_load must return a plain dict, got {type(result)}"


class TestParquetWritePassesStr:
    """Verify the a5_results.parquet site passes str() to pl.write_parquet.

    Rather than driving cmd_score end-to-end (heavy; deep manifest deps),
    inspect the source to confirm str() is applied at the write site.
    This is a cross-check that the pattern is correct, not a runtime assertion.
    """

    def test_write_parquet_str_applied_in_source(self):
        source_path = BASE_DIR / "scripts" / "run_a5_backtest.py"
        source = source_path.read_text()
        # The Polars write must pass str(...) not a bare UPath object.
        assert 'write_parquet(str(root / "a5_results.parquet")' in source, (
            "run_a5_backtest.py must str()-wrap the path in df.write_parquet() "
            "to avoid the 'Object does not have a .read() method' UPath bug "
            "(plans/15 MinIO verify; commit 85fcc7b)."
        )

    def test_a4_write_parquet_str_applied_in_source(self):
        source_path = BASE_DIR / "scripts" / "run_a4_backtest.py"
        source = source_path.read_text()
        assert 'write_parquet(str(root / "a4_results.parquet")' in source, (
            "run_a4_backtest.py must str()-wrap the path in df.write_parquet() "
            "to avoid the UPath Polars bug."
        )
