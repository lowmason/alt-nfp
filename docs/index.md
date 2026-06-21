# alt-nfp

Bayesian state-space **nonfarm-payroll (NFP) nowcasting** from real-time data
vintages. `alt-nfp` reconstructs what was *knowable on any past date* from
published BLS/QCEW data, then nowcasts the next NFP print with a Bayesian
state-space model — decomposed as **private employment + a government wedge**.

## Three ways to read these docs

- **New here?** Start with [Get Started](get-started/installation.md) — install,
  run the CLI, and produce a nowcast.
- **Maintaining or extending it?** See
  [Architecture & Internals](architecture/package-chain.md) — the package chain,
  boundaries, and storage contract.
- **Here for the method?** See
  [Concepts & Methodology](concepts/vintages-and-censoring.md) — the vintage data
  model and the additive Bayesian nowcast.

The full public API is in the [API Reference](reference/index.md).
