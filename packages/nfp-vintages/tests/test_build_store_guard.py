import pytest
from nfp_vintages.build_store import build_store
from upath import UPath


def test_bare_cli_callback_passes_allow_canonical_false(monkeypatch):
    """Verify bare ``alt-nfp`` (build(None)) does not bypass the guard.

    Calling build(None) from the typer callback passes OptionInfo objects for
    un-resolved parameters, which are truthy.  The callback must pass
    allow_canonical=False explicitly so the guard fires against the canonical
    store.
    """
    import nfp_vintages.build_store as bs_mod

    captured = {}

    def capturing_build_store(**kwargs):
        captured.update(kwargs)

    # build() does `from nfp_vintages.build_store import build_store` on each
    # call — patch the module-level name so the import resolves to our stub.
    monkeypatch.setattr(bs_mod, 'build_store', capturing_build_store)

    from nfp_vintages.__main__ import build

    build(None, allow_canonical=False)

    assert captured.get('allow_canonical') is False, (
        f'build(None, allow_canonical=False) must forward allow_canonical=False '
        f'to build_store; got {captured!r}'
    )


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
