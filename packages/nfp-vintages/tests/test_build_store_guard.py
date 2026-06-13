import pytest
from nfp_vintages.build_store import build_store
from upath import UPath


def test_refuses_canonical_remote_store_without_optin():
    with pytest.raises(RuntimeError, match="canonical"):
        build_store(store_path=UPath("s3://alt-nfp/store"))


def test_allows_scratch_rebuild_prefix(tmp_path):
    # Guard must not fire; the function proceeds and fails on missing input file.
    with pytest.raises(Exception) as exc:
        build_store(
            revisions_path=tmp_path / "nope.parquet",
            store_path=UPath("s3://alt-nfp/store-rebuild"),
        )
    assert not isinstance(exc.value, RuntimeError)


def test_allows_canonical_with_explicit_optin(tmp_path):
    # Guard must not fire when allow_canonical=True; fails on missing input file.
    with pytest.raises(Exception) as exc:
        build_store(
            revisions_path=tmp_path / "nope.parquet",
            store_path=UPath("s3://alt-nfp/store"),
            allow_canonical=True,
        )
    assert not isinstance(exc.value, RuntimeError)


def test_local_store_never_guarded(tmp_path):
    # Local paths are never guarded regardless of name; fails on missing input file.
    with pytest.raises(Exception) as exc:
        build_store(revisions_path=tmp_path / "nope.parquet", store_path=tmp_path / "store")
    assert not isinstance(exc.value, RuntimeError)
