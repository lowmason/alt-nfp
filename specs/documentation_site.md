# Documentation Site — Design Brief (BRAINSTORM IN PROGRESS)

> **STATUS: draft / brainstorm paused 2026-06-20.** This is a resume-point, *not* a
> finalized spec. Decisions banked + findings + open questions are below. The real
> spec gets written once the remaining open questions are answered and a site design
> is approved (per `superpowers:brainstorming` → `writing-plans`). Paused so two
> unrelated checks (CLI optimization, local-data audit) happen first.

## Goal

A **very thorough** documentation site for the `alt-nfp` workspace using **MkDocs +
Material + mkdocstrings** (autodocstrings). Layered to serve three audiences at once.

## Findings (from context exploration)

- **Toolchain already installed** (root `pyproject.toml` `[dependency-groups] docs`):
  `mkdocs-material`, `mkdocstrings[python]`, **plus** `mkdocs-gen-files`,
  `mkdocs-literate-nav`, `mkdocs-section-index`. That exact set is the canonical
  mkdocstrings *"automatic API reference"* recipe (a `gen_files` script emits one page
  per module; nav auto-built from a generated `SUMMARY.md`). **No `mkdocs.yml` or
  `docs/` scaffold exists yet** (only a stray `docs/government_design.md`).
- **Docstrings are uniformly NumPy-style** — 39 files with `Parameters\n----------`,
  **zero** Google `Args:`. So mkdocstrings just needs `docstring_style: numpy`.
- **Docstring coverage = 87%** of public objects (256 objects, 222 documented). Weak
  spot: **`nfp_model` at 65%** (~14 undocumented public objects); ~34 missing repo-wide.
- **Rich source material to draw on:** 5 package `CLAUDE.md` maps, `specs/` (active
  design records) + `archive/` (implemented), `ARCHITECTURE.md`, `README.md`,
  `plans/`. The model has heavy math (AR(1), Fourier-GRW seasonal, QCEW Student-t
  anchor, structural birth/death) that a "methods record" audience will want rendered.
- **Public repo, proprietary gitignored `data/`** — docs must not leak data; examples
  can't use real vintages.

## Decisions banked

1. **Audience: comprehensive** — maintainers (how it works + how to extend) +
   newcomers (onboarding, conceptual walkthroughs) + a methods record (Bayesian
   methodology, vintage data model, parity-vs-ground-truth validation philosophy).
   Layered so each audience has a clear path.
2. **Docstring style: keep NumPy** — idiomatic for the scientific stack
   (numpy/scipy/jax/numpyro), already 87% consistent, zero conversion churn;
   mkdocstrings renders it via `docstring_style: numpy`.

## Open questions (resume here)

a. **Depth of hand-written conceptual / methodology docs** — how far beyond the
   auto API reference? Candidates: the Bayesian state-space model (math), the vintage
   data model + two-layer as-of censoring, the Track A/Track B framing, the
   build-here/validate-on-port + parity≠correctness philosophy, the A5 first-print
   evaluation. (Comprehensive audience implies "deep," but scope the set.)
b. **Fill the ~34 missing docstrings as part of this?** Especially `nfp_model` (65%)
   — the auto API ref is only as complete as the docstrings. Big-ish sub-task; decide
   in-scope vs follow-up.
c. **Hosting** — GitHub Pages via a CI workflow (`mkdocs gh-deploy`) vs local build
   only (`mkdocs serve`). Repo is public → GH Pages is natural.
d. **API-ref granularity** — public symbols only? Include the eval/`scripts/` layer
   and `nfp_model.tests`? Per-package sections vs one flat reference.
e. **Math rendering** — enable KaTeX/MathJax in Material for the model equations
   (almost certainly yes for the methods-record audience).

## Next brainstorm steps (when resumed)

1. Answer open questions a–e (one at a time).
2. Propose 2–3 site structures with trade-offs + a recommendation.
3. Present the design section-by-section; get approval.
4. Finalize this file into the real spec (strip the draft banner; remove open-question
   placeholders); self-review; user review gate.
5. `superpowers:writing-plans` → implementation plan → build.
