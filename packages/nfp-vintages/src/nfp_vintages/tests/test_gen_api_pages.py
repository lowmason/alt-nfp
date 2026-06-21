"""Unit test for scripts/gen_api_pages.py (the API-ref generator).

Exercises the pure walk/filter (iter_doc_targets) against a synthetic package
tree — no mkdocs build context, no network. Loaded by path because scripts/ is
not on testpaths (mirrors test_bootstrap_store.py).
"""
from __future__ import annotations

import importlib.util
import sys

from nfp_lookups.paths import BASE_DIR


def _load_gen():
    path = BASE_DIR / "scripts" / "gen_api_pages.py"
    spec = importlib.util.spec_from_file_location("gen_api_pages", path)
    assert spec and spec.loader, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # required on Py 3.12: @dataclass resolves annotations via sys.modules
    spec.loader.exec_module(mod)  # __name__ == "gen_api_pages" -> no I/O
    return mod


def _make_pkg(tmp_path):
    """A synthetic packages/ tree under one workspace package name."""
    src = tmp_path / "packages" / "nfp-lookups" / "src" / "nfp_lookups"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("")
    (src / "__main__.py").write_text("")
    (src / "public_mod.py").write_text("def f(): ...")
    (src / "_private.py").write_text("def g(): ...")
    internal = src / "_internal"
    internal.mkdir()
    (internal / "__init__.py").write_text("")
    (internal / "helper.py").write_text("def h(): ...")
    tests = src / "tests"
    tests.mkdir()
    (tests / "__init__.py").write_text("")
    (tests / "test_x.py").write_text("def test_x(): ...")
    return tmp_path / "packages"


def test_iter_doc_targets_public_only(tmp_path):
    gen = _load_gen()
    targets = gen.iter_doc_targets(_make_pkg(tmp_path))
    ids = {t.identifier for t in targets}
    assert "nfp_lookups.public_mod" in ids
    assert "nfp_lookups" in ids  # __init__ -> package index
    assert "nfp_lookups._private" not in ids
    assert not any(i.startswith("nfp_lookups._internal") for i in ids)
    assert not any("tests" in t.nav_parts for t in targets)
    assert not any(i.endswith("__main__") for i in ids)


def test_doc_path_for_init_is_index(tmp_path):
    gen = _load_gen()
    by_id = {t.identifier: t for t in gen.iter_doc_targets(_make_pkg(tmp_path))}
    assert by_id["nfp_lookups"].doc_path == "nfp_lookups/index.md"
    assert by_id["nfp_lookups.public_mod"].doc_path == "nfp_lookups/public_mod.md"
