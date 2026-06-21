import pytest
from nfp_vintages.build_store import build_store
from upath import UPath


@pytest.fixture(autouse=True)
def _no_real_store(monkeypatch):
    """Fail closed: no test in this module may reach the real MinIO store.

    The guard under test stops canonical-store rebuilds, but the "allows" tests
    deliberately let ``build_store`` proceed past the guard — they must fail on
    missing *local* input, NEVER by reaching production S3. Blanking the creds
    and endpoint redirects any accidental S3 I/O away from the live MinIO so it
    cannot connect (defense-in-depth behind the missing-input-file failure).
    This exists because a red-phase run of this test once wiped the canonical
    store; see plans/8 'Store-write safety'.
    """
    for var in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ENDPOINT_URL",
        "AWS_SESSION_TOKEN",
        "NFP_STORE_URI",
    ):
        monkeypatch.delenv(var, raising=False)


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
