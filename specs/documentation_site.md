# Documentation Site — Design Spec

> **STATUS: finalized 2026-06-21.** Brainstorm complete; all open questions resolved.
> This is the design record that feeds `superpowers:writing-plans`. Implementation
> (scaffolding, config, page authoring, docstring fill) has **not** started.

## Goal

A comprehensive documentation site for the `alt-nfp` uv workspace, built with
**MkDocs + Material + mkdocstrings**, layered to serve three audiences from one
source tree:

- **Newcomers** — install, orient, run the CLI, produce a nowcast.
- **Maintainers** — the package architecture, dependency rules, storage/paths
  discipline, and how to extend the system.
- **Methods record** — the modeling methodology: the vintage data model, the
  additive (private + government) nowcast, and the Bayesian state-space math.

## Resolved decisions

| # | Decision | Choice |
|---|----------|--------|
| Audience | Who the docs serve | **Comprehensive** — newcomers + maintainers + methods record, layered |
| Docstrings | Style | **Keep NumPy** (`docstring_style: numpy`); already ~87% consistent, zero conversion churn |
| Concept depth | Hand-written methodology layer | **Full methods record** — dedicated pages for the modeling methodology (see below) |
| Docstring coverage | Fill the gaps? | **Fill all ~34 missing public docstrings** repo-wide to ~100% (nfp_model's ~14 first); doc-only, no logic change |
| Hosting | Where it deploys | **GitHub Pages, manual publish** — `workflow_dispatch` CI + local `mkdocs gh-deploy`; never auto-deploy on push |
| API ref | Scope & shape | **Per-package, public API only**; exclude `scripts/` and `tests/` |
| Math | Equation rendering | **KaTeX** via `pymdownx.arithmatex` (generic mode) |
| Structure | Information architecture | **Audience-first** top nav; Concepts section ordered as the data → model pipeline spine |

**Excluded from the published methods record (deliberate):** a "parity ≠ correctness /
validate-on-port philosophy" page and an "A5 first-print evaluation" page were considered
and cut. The methods record documents *how the model works* (durable modeling
methodology), not the research/validation philosophy or current evaluation findings —
which are also the most sensitive content to put on a public site. These remain captured
in `specs/` as internal design records and may be revisited as a follow-up.

## Site structure (information architecture)

Audience-first top-level navigation. The Concepts section is internally ordered as the
`data → [private + government] model` pipeline spine, so it also reads narratively for the
methods-record audience.

```
Home                              index.md — what alt-nfp is, the three reading paths
Get Started                       newcomer onboarding
   ├─ Installation                uv sync, the workspace, .env / NFP_* env URIs
   ├─ Quickstart                  run the CLI (status/update), produce a nowcast
   └─ The data story              store is proprietary + gitignored; examples are synthetic
Concepts & Methodology            the methods record
   ├─ Vintage data model & as-of censoring
   └─ Additive Nowcast Framework  Total = Private('05') + Government wedge(00−05)
        ├─ Overview               the additive identity; what's targeted & why; Track A/B ↔ this
        ├─ The Private State-Space Model    (Benchmark revision decomposition = H2 subsection)
        └─ The Government Wedge
Architecture & Internals          maintainer reference
   ├─ The package chain           lookups → download → ingest → vintages; model apart
   ├─ Boundaries & paths          no upward/private imports; nfp_lookups.paths discipline
   └─ Storage contract            MinIO/S3 env URIs; no ./data writes; rebuild-to-scratch/promote
CLI Reference                     alt-nfp commands (production workflow) + bootstrap_store.py
API Reference                     auto-generated (mkdocstrings); per-package, public only
```

### Methodology layer — page-by-page scope

**1. Vintage data model & as-of censoring.** What a real-time data *vintage* is; the
Hive-partitioned store keyed by `source` × `seasonally_adjusted`; and the **two-layer
as-of censoring** that prevents lookahead — Layer 1 panel-level rank-based selection (the
triangular CES rev-0/1/2 diagonal, benchmark-revision filtering) and Layer 2 model-level
`vintage_date` filtering (provider + cyclical publication lags). Draws on
`vintage_store.py`, `panel_adapter.py`, and the benchmark-splice growth convention.
Diagram: the two-layer censoring flow.

**2. Additive Nowcast Framework (nav section).**

- **2a. Overview.** The identity the model is built on: **Total NFP = Private (`'05'`)
  nowcast + Government wedge (`00 − 05`)**, combined additively. Why `'05'` is the primary
  modeling target; why `'00'` total is *decomposed* rather than modeled directly; how
  total-NFP consensus maps onto the framework. One line noting the internal code term
  "Track A / Track B" corresponds to the private / government components, so maintainers
  reading the source aren't lost. Diagram: the additive decomposition.

- **2b. The Private State-Space Model.** The math centerpiece, KaTeX-rendered: the latent
  NFP state and observation model — **AR(1)** core, **Fourier-GRW seasonal**, **QCEW
  Student-t anchor**, the **structural birth/death** term
  (`φ₀ + φ₁·X^birth + φ₂·BD^QCEW + φ₃·X^cycle + σ_bd·ξ`), the **cyclical indicators**
  (claims, NFCI, biz-apps, JOLTS) with their publication lags, and the **precision
  budget** apportioning information across sources. Drawn from `nfp_model/*`.
  - **H2 subsection — Benchmark revision decomposition.** What the annual CES benchmark
    re-anchoring to QCEW is, and how `benchmark.py` extracts the revision from this
    model's posterior and decomposes it into **continuing-units + birth/death**
    components. **Methodology-only**: the T-12…T-1 horizon backtest is mentioned as the
    *validation mechanism*, without publishing accuracy numbers.

- **2c. The Government Wedge.** Track B's forecast side: the thin **Bayesian change-space
  STS** forecasting the wedge `00 − 05` (not published `90`), with **announcement-priored
  interventions** (RIF magnitudes). Drawn from the government-wedge build.

## Auto-generated API reference

Standard mkdocstrings "automatic code reference" recipe, adapted for the 5-package
workspace.

- **`scripts/gen_api_pages.py`** (an `mkdocs-gen-files` script): walks each workspace
  package's public source under `packages/<pkg>/src/nfp_*/`, emits one virtual Markdown
  stub per module under a generated `reference/` tree (each stub is a single
  `::: nfp_x.module` autodoc directive), and writes the `SUMMARY.md` that
  `mkdocs-literate-nav` consumes to build the API nav. `mkdocs-section-index` gives each
  package a landing page.
- **Public-only filter** (the recipe's job): skip any module or member whose name starts
  with `_`; skip `tests/` directories entirely; skip `scripts/`. mkdocstrings honors
  `__all__` where defined, and the handler's `filters: ["!^_"]` drops private members.
- **Grouping:** one top-level section per package, ordered along the dependency chain
  (`nfp-lookups → nfp-download → nfp-ingest → nfp-vintages`, then `nfp-model`).
- **Test:** a unit test for `gen_api_pages.py` runs the walk against a small fixture tree
  and asserts (a) one stub per public module, (b) the `SUMMARY.md` nav structure, and
  (c) the public-only filter excludes `_private`, `tests/`, and `scripts/`.

## Docstring completion sub-phase

- Fill all **~34 missing public docstrings** repo-wide to ~100% public coverage, **NumPy
  style**, prioritizing `nfp_model` (~14, currently 65%) since the API ref's quality
  there is what the methods-record audience leans on.
- **Documentation-only** — no logic touched, no behavior change. This stays clear of the
  "don't touch model logic" firewall even where it edits `nfp_model` files.
- **Coverage gate:** add `interrogate` to the `docs` dependency group, configured in
  `pyproject.toml` `[tool.interrogate]` with `fail-under = 100` over public objects
  (ignore private, `tests/`, `scripts/`, `__init__`). Run in the docs workflow and
  locally so coverage can't regress.

## Toolchain & configuration

The `docs` dependency group is already installed (`mkdocs-material`,
`mkdocstrings[python]`, `mkdocs-gen-files`, `mkdocs-literate-nav`,
`mkdocs-section-index`). Add `interrogate` for the coverage gate. Representative
`mkdocs.yml` (root):

```yaml
site_name: alt-nfp
site_description: Bayesian state-space NFP nowcasting from real-time data vintages
repo_url: https://github.com/lowmason/alt-nfp
repo_name: lowmason/alt-nfp

theme:
  name: material
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.instant
    - navigation.top
    - toc.follow
    - content.code.copy
    - search.suggest
  palette:
    - media: "(prefers-color-scheme: light)"
      scheme: default
      toggle: { icon: material/weather-night, name: Switch to dark mode }
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      toggle: { icon: material/weather-sunny, name: Switch to light mode }

plugins:
  - search
  - gen-files:
      scripts: [scripts/gen_api_pages.py]
  - literate-nav:
      nav_file: SUMMARY.md
  - section-index
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: numpy
            show_root_heading: true
            show_source: true
            show_signature_annotations: true
            members_order: source
            filters: ["!^_"]

markdown_extensions:
  - admonition
  - pymdownx.details
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
  - pymdownx.arithmatex:
      generic: true
  - pymdownx.highlight
  - toc:
      permalink: true
  - tables

extra_javascript:
  - javascripts/katex.js
  - https://unpkg.com/katex@0/dist/katex.min.js
  - https://unpkg.com/katex@0/dist/contrib/auto-render.min.js
extra_css:
  - https://unpkg.com/katex@0/dist/katex.min.css

nav:
  - Home: index.md
  - Get Started:
      - Installation: get-started/installation.md
      - Quickstart: get-started/quickstart.md
      - The data story: get-started/data-story.md
  - Concepts & Methodology:
      - Vintage data model & as-of censoring: concepts/vintages-and-censoring.md
      - Additive Nowcast Framework:
          - Overview: concepts/additive/overview.md
          - The Private State-Space Model: concepts/additive/private-state-space.md
          - The Government Wedge: concepts/additive/government-wedge.md
  - Architecture & Internals:
      - The package chain: architecture/package-chain.md
      - Boundaries & paths: architecture/boundaries-and-paths.md
      - Storage contract: architecture/storage-contract.md
  - CLI Reference: cli-reference.md
  - API Reference: reference/   # built by literate-nav from generated SUMMARY.md
```

(`ruff` already `extend-exclude`s `docs`, so authored pages and any committed example
snippets are not linted.)

## Diagrams

Mermaid (rendered natively by Material via the `superfences` custom fence):

- The 5-package linear dependency chain (Architecture).
- The two-layer as-of censoring flow (Vintage data model page).
- The additive decomposition `Total = Private('05') + Government wedge(00−05)`
  (Additive Nowcast Framework overview).

## Hosting & CI

- **`.github/workflows/docs.yml`**, **`workflow_dispatch` trigger only** (manual — never
  on push). Steps: checkout, `uv sync --group docs`, `uv run mkdocs build --strict`
  (gate), then `uv run mkdocs gh-deploy --force` (pushes the built site to the
  `gh-pages` branch).
- Local authoring via `uv run mkdocs serve`; local publish via `uv run mkdocs gh-deploy`.
- Keeps a deliberate hand on the public-surface: the methods record is published only
  when the workflow is triggered, not on every commit.

## Build gate / verification

- **`mkdocs build --strict`** is the primary gate — it fails on broken internal links,
  missing autodoc targets, and unresolved references. Run in the workflow and locally.
- **`interrogate --fail-under 100`** guards public-docstring coverage.
- **`gen_api_pages.py` unit test** (above) guards the API-ref generator.

## Repo layout

- `mkdocs.yml` — repo root.
- `docs/` — authored Markdown (Home, Get Started, Concepts, Architecture, CLI Reference)
  plus `docs/javascripts/katex.js`. The existing stray `docs/government_design.md` is
  **folded into** the Government Wedge page and the stray file removed.
- `scripts/gen_api_pages.py` — the API-ref generator (not a CLI command).
- Generated `reference/` and the built `site/` are git-ignored.
- This design record stays in `specs/` (the repo's design-record home), separate from the
  site content in `docs/`, so a spec is never mistaken for a site page.

## Implementation phases (for writing-plans)

1. **Scaffold** — `mkdocs.yml`, `docs/index.md`, theme/math/Mermaid config; `mkdocs serve`
   renders a minimal home. *(Deliverable: site builds and serves.)*
2. **Auto API reference** — `gen_api_pages.py` + its unit test; per-package public-only
   reference renders under `--strict`.
3. **Docstring completion** — fill ~34 gaps (nfp_model first); add `interrogate` gate.
4. **Architecture & Get Started pages** — package chain, boundaries/paths, storage
   contract; installation, quickstart, data story; Mermaid dependency-chain diagram.
5. **Methodology layer** — the Vintage page (+ censoring diagram) and the Additive Nowcast
   Framework section (overview + private state-space [+ benchmark H2] + government wedge);
   KaTeX math; fold `government_design.md`.
6. **CLI Reference** — `alt-nfp` commands + `bootstrap_store.py`, drawn from
   `specs/cli_production_workflow.md`.
7. **Hosting/CI** — `docs.yml` (`workflow_dispatch`), strict-build gate; first manual
   `gh-deploy`.
