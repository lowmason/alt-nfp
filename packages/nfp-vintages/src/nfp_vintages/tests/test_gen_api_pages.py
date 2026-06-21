"""Unit test for scripts/gen_api_pages.py (the API-ref generator).

Exercises the pure walk/filter (iter_doc_targets) against a synthetic package
tree — no mkdocs build context, no network. Loaded by path because scripts/ is
not on testpaths (mirrors test_bootstrap_store.py).
"""
from __future__ import annotations

import importlib.util
import sys
import types

from nfp_lookups.paths import BASE_DIR


def _load_gen():
    path = BASE_DIR / "scripts" / "gen_api_pages.py"
    spec = importlib.util.spec_from_file_location("gen_api_pages", path)
    assert spec and spec.loader, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    # Register before exec_module: on Python 3.12 a frozen @dataclass needs its
    # module in sys.modules during class creation. Restore the prior state after,
    # so the test leaves no stray "gen_api_pages" entry behind.
    prev = sys.modules.get(spec.name)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        if prev is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = prev
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


class _FakeFile:
    """Records what mkdocs_gen_files.open() is written, on context exit."""

    def __init__(self, store, path):
        self._store, self._path, self._buf = store, path, []

    def write(self, s):
        self._buf.append(s)

    def writelines(self, lines):
        self._buf.extend(lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._store[self._path] = "".join(self._buf)
        return False


class _FakeNav(dict):
    """Minimal stand-in for mkdocs_gen_files.Nav."""

    def build_literate_nav(self):
        return [f"* [{'.'.join(k)}]({v})\n" for k, v in self.items()]


def _fake_gen_files():
    fake = types.ModuleType("mkdocs_gen_files")
    fake.files = {}
    fake.Nav = _FakeNav
    fake.open = lambda path, mode: _FakeFile(fake.files, path)
    return fake


def test_generate_emits_reference_index_landing():
    """_generate() must write reference/index.md and lead SUMMARY with it.

    Regression guard for the /reference/ 404: without a landing page that URL has
    nothing to serve, and mkdocs --strict logs the dangling link only at INFO.
    """
    gen = _load_gen()
    fake = _fake_gen_files()
    prev = sys.modules.get("mkdocs_gen_files")
    sys.modules["mkdocs_gen_files"] = fake
    try:
        gen._generate()
    finally:
        if prev is None:
            sys.modules.pop("mkdocs_gen_files", None)
        else:
            sys.modules["mkdocs_gen_files"] = prev

    assert "reference/index.md" in fake.files
    assert "# API Reference" in fake.files["reference/index.md"]
    # The landing must be the first SUMMARY entry so section-index attaches it.
    assert fake.files["reference/SUMMARY.md"].startswith("* [Overview](index.md)\n")
    # The real packages still get pages (walk ran against the live tree).
    assert any(p.startswith("reference/nfp_lookups") for p in fake.files)
