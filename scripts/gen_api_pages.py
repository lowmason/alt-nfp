#!/usr/bin/env python3
"""Generate the per-package API reference (an mkdocs-gen-files script).

Run automatically by the mkdocs-gen-files plugin during ``mkdocs build`` via
``runpy.run_path``, which sets ``__name__`` to ``"<run_path>"``. Walks each
workspace package's public ``src/`` and emits one virtual reference page per
module plus a literate-nav ``SUMMARY.md``. Public-only: modules / sub-packages
whose name starts with ``_`` and any ``tests`` directory are skipped (``scripts/``
is not under ``src/`` and never appears).

The pure walk (``iter_doc_targets``) is import-safe so it can be unit-tested
without a mkdocs build context.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Dependency-chain order; each maps to packages/<dist-name>/src/<import-name>.
WORKSPACE_PACKAGES: tuple[str, ...] = (
    "nfp_lookups",
    "nfp_download",
    "nfp_ingest",
    "nfp_vintages",
    "nfp_model",
)


@dataclass(frozen=True)
class DocTarget:
    """One reference page to emit."""

    identifier: str             # dotted module, e.g. "nfp_ingest.vintage_store"
    doc_path: str               # under reference/, e.g. "nfp_ingest/vintage_store.md"
    nav_parts: tuple[str, ...]  # nav key, e.g. ("nfp_ingest", "vintage_store")


def _is_public(parts: tuple[str, ...]) -> bool:
    """True iff no path part is private (``_``-prefixed) or a ``tests`` dir."""
    return not any(p == "tests" or p.startswith("_") for p in parts)


def iter_doc_targets(packages_root: Path) -> list[DocTarget]:
    """Walk every workspace package under *packages_root* into public DocTargets.

    *packages_root* is the repo's ``packages/`` directory. For each package,
    walks ``packages/<dist>/src`` and yields one target per public module, with
    ``__init__.py`` mapped to the package's ``index.md`` and ``__main__.py``
    skipped.
    """
    targets: list[DocTarget] = []
    for import_name in WORKSPACE_PACKAGES:
        dist_name = import_name.replace("_", "-")
        src = packages_root / dist_name / "src"
        if not src.is_dir():
            continue
        for path in sorted(src.rglob("*.py")):
            parts = tuple(path.relative_to(src).with_suffix("").parts)
            if parts[-1] == "__main__":
                continue
            if parts[-1] == "__init__":
                parts = parts[:-1]
                if not parts:
                    continue
                doc_path = "/".join(parts) + "/index.md"
            else:
                doc_path = "/".join(parts) + ".md"
            if not _is_public(parts):
                continue
            targets.append(DocTarget(".".join(parts), doc_path, parts))
    return targets


def _generate() -> None:
    """Emit the reference pages + SUMMARY.md into the mkdocs build."""
    import mkdocs_gen_files

    root = Path(__file__).resolve().parent.parent  # repo root
    nav = mkdocs_gen_files.Nav()
    for target in iter_doc_targets(root / "packages"):
        nav[target.nav_parts] = target.doc_path
        with mkdocs_gen_files.open(f"reference/{target.doc_path}", "w") as fd:
            fd.write(f"::: {target.identifier}\n")
    with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as fd:
        fd.writelines(nav.build_literate_nav())


# mkdocs-gen-files runs this file via runpy.run_path -> __name__ == "<run_path>";
# a direct `python` run gives "__main__". Generate under both. Under importlib
# import (the unit test) __name__ is the module name, so the import is
# side-effect-free.
if __name__ in ("__main__", "<run_path>"):
    _generate()
