# Bayesian Workflow Reference

A condensed guide for Bayesian modeling, inference, and evaluation. Based on Gelman et al. (2020) "Bayesian Workflow" and Gabry et al. (2019) "Visualization in Bayesian Workflow."

---

## Core Principle

Bayesian workflow is iterative: model building → inference → checking → expansion. Expect to fit many models. Poor models and failures are informative steps, not dead ends.

---

## 1. Before Fitting a Model

### Model Construction

- Start from existing models (textbooks, papers, prior work) and adapt.
- Build modularly: treat each component (likelihood, link function, priors, hierarchy) as a placeholder that can be swapped later.
- Consider both ground-up (simple → complex) and top-down (complex → simplified) strategies. Ground-up is preferable when data are sparse or unbalanced.

### Parameterization

- Keep parameters interpretable and on natural scales.
- Separate out scale so unknowns are scale-free (e.g., model on `log(θ/θ_expected)` so 0 is meaningful).
- For hierarchical models, consider non-centered parameterizations to avoid funnel pathologies.

### Prior Selection

Think of a ladder from least to most informative:

1. Flat / improper prior
2. Super-vague but proper
3. Very weakly informative
4. Generic weakly informative
5. Specific informative

**Weakly informative priors** should generate plausible but not indistinguishable-from-observed data under the prior predictive distribution. They should have some mass on extreme-but-plausible datasets and no mass on impossible datasets.

Key rules of thumb:

- As models grow more complex, priors generally need to become tighter. The information budget is divided among more parameters.
- Even independent priors on coefficients interact through the likelihood; more predictors → stronger priors needed.
- Vague priors (e.g., N(0, 100)) often generate impossible data and should be avoided.
- Center priors on scientifically reasonable defaults (e.g., slope ≈ 1 if predictor is a noisy version of the outcome).

### Prior Predictive Checking

Simulate data from the prior predictive distribution `p(y) = ∫ p(y|θ)p(θ)dθ` and visualize. The simulated datasets should look like data that *could* plausibly be observed, not data that *match* the observed data. This catches bad priors before fitting.

---

## 2. Fitting a Model

### Algorithm Choice

- **Early exploration:** Use fast approximate methods (variational inference, Laplace approximation, INLA, empirical Bayes) to iterate quickly.
- **Final inference:** Use MCMC (typically HMC/NUTS via Stan, PyMC, NumPyro, etc.) for accurate posterior exploration.
- An approximate algorithm can be viewed as an exact algorithm for an approximate model.

### Fit Fast, Fail Fast

- Don't over-invest in computation for untested models.
- Run fewer iterations initially (e.g., 200 instead of 2000).
- Inappropriate models are often harder to fit — use this as a diagnostic signal.
- Fit on subsets of data or simplified versions first.

### Convergence

- Target R̂ < 1.01 for all parameters and quantities of interest.
- Monitor effective sample size (ESS).
- In early stages, don't require perfect convergence — large-scale features are sufficient for model comparison decisions.

---

## 3. Diagnosing Computational Problems

### The Folk Theorem of Statistical Computing

> When you have computational problems, often there's a problem with your model.

Before throwing more compute at convergence issues, check for: regions of parameter space not of substantive interest, nonsensical models, or code bugs.

### Debugging Strategy

Work from both ends simultaneously:

- **From the complex side:** Simplify the failing model until something works.
- **From the simple side:** Start simple and add features until the problem appears.
- Unit-test model components individually with simulated data.

### HMC-Specific Diagnostics

- **Divergent transitions** indicate regions of high curvature the sampler cannot explore. Visualize divergences in bivariate scatterplots or parallel coordinate plots.
  - Divergences concentrated in a small region → geometric pathology (e.g., funnel).
  - No obvious pattern → likely false positives (increase `adapt_delta`).
- **Funnel pathologies** in hierarchical models: when group-level variance → 0, the posterior forms a funnel. Fix with non-centered parameterization.
- **Multimodality types and fixes:**
  1. Near-zero mass in all but one mode → use good initial values or tighter priors.
  2. Label switching (symmetric modes) → add identifying constraints.
  3. Genuinely multi-modal → stacking, strong priors, or mixture approaches.
  4. Unstable tail → initialize near the mass.

### Other Fixes

- **Reparameterize** to improve posterior geometry.
- **Marginalize** troublesome parameters analytically when possible.
- **Add prior information** that was available but not yet included.
- **Add data** — models well-behaved for larger data can have issues in small-data regimes.

---

## 4. Fake-Data Simulation and Calibration

### Fake-Data Simulation

1. Choose reasonable parameter values.
2. Simulate a dataset matching the structure of real data.
3. Fit the model to the simulated data.
4. Verify that posterior inference recovers the known parameters.

This is the only point where you can directly verify that inference on latent variables is reliable. If it fails, simplify until it works and identify the breaking point.

### Simulation-Based Calibration (SBC)

A systematic extension: draw parameters from the prior, simulate data, fit, and check that aggregated posteriors recover the prior. Useful for validating both models and approximate inference algorithms.

---

## 5. Evaluating a Fitted Model

### Posterior Predictive Checking

Generate replicated datasets from the posterior predictive distribution `p(ỹ|y) = ∫ p(ỹ|θ)p(θ|y)dθ` and compare to observed data. Key visualizations:

- **Density overlays:** Compare distribution of replicated data to observed data.
- **Test statistics:** Compute summary statistics (mean, sd, skewness, min, max, quantiles) on replicated data and compare to observed. Choose statistics orthogonal to model parameters for informative checks.
- **Grouped checks:** Evaluate within levels of grouping variables to detect localized misfit.

Aim for *severe tests* — checks likely to fail if the model would give misleading answers to important questions.

### Cross-Validation

Posterior predictive checks use data twice (fitting + checking). Cross-validation (especially LOO-CV) partially avoids this.

- **LOO-PIT (Probability Integral Transform):** Should be uniform for well-calibrated models.
  - "Frown" shape → predictive distributions too wide.
  - "Smile" shape → predictive distributions too narrow.
- **PSIS-LOO:** Efficient approximation to LOO-CV via Pareto-smoothed importance sampling.
- **k̂ diagnostic:** Flags influential observations where full-data and LOO posteriors diverge substantially. Large k̂ values warrant investigation.

### Identifying Problematic Observations

- **Outliers:** Difficult to predict under the model.
- **High-leverage points:** Disproportionately influential on the posterior.
- Use pointwise ELPD (expected log predictive density) to compare how individual observations are handled across models.

### Prior Sensitivity

- Refit with alternative priors (sensitivity analysis).
- Compare prior and posterior standard deviations (shrinkage).
- Use importance sampling to approximate posteriors under new priors without refitting.

---

## 6. Modifying and Comparing Models

### Model Expansion

- Expand models based on posterior predictive check failures, domain knowledge, and new data.
- Center expansions on the current model: weakly informative extensions that reduce to the simpler model under appropriate parameter values.
- Bigger datasets demand bigger models. Big data is messy data — proxies, observational designs, heterogeneous coverage.

### Model Comparison

- Use LOO-CV (via PSIS-LOO) to compare predictive performance.
- When comparison is uncertain, use **stacking** to combine model inferences rather than selecting a single winner.
- Stacking is preferred over traditional Bayesian model averaging, which depends strongly on aspects with minimal predictive impact.
- **Projection predictive variable selection** is useful for finding parsimonious models with equivalent predictive performance.

### On Using Data Twice

Bayesian workflow uses observed data to guide model building, which raises overfitting concerns. Mitigations:

- Prior predictive checks don't match observed data closely — they check for plausibility, not fit.
- Posterior predictive checks flag problems for model *expansion*, not optimization.
- Cross-validation provides honest predictive assessment.
- A model whose assumptions survive severe tests is often more trustworthy than an untested preregistered model.

---

## 7. Software Engineering Practices

- **Version control:** Use Git from the start. Track models, data processing, and results.
- **Modularity:** Encapsulate repeated code into functions. Big tangled scripts are hard to debug and maintain.
- **Reproducibility:** Write self-contained scripts. Another person on another machine should be able to recreate results.
- **Readability:** Use descriptive parameter names, consistent formatting, and treat code as communication.

---

## Quick-Reference: Workflow Checklist

1. **EDA and model design:** Explore data, choose initial model structure, build modularly.
2. **Set priors:** Use weakly informative priors. Run prior predictive checks.
3. **Fake-data simulation:** Verify the model can recover known parameters.
4. **Fit the model:** Start fast and approximate, refine to full MCMC.
5. **Computational diagnostics:** Check R̂, ESS, divergences, energy diagnostics.
6. **Posterior predictive checks:** Compare replicated data to observed data.
7. **Cross-validation:** Assess predictive performance, identify influential points.
8. **Iterate:** Expand or modify model based on findings. Repeat from step 2.
9. **Compare models:** Use LOO-CV, stacking, pointwise diagnostics.
10. **Report:** Present results with uncertainty, document workflow decisions.

---

## Key References

- Gelman et al. (2020). Bayesian Workflow. arXiv:2011.01808.
- Gabry et al. (2019). Visualization in Bayesian Workflow. JRSS-A. arXiv:1709.01449.
- Gelman et al. (2013). Bayesian Data Analysis (3rd ed.). Chapman & Hall/CRC.
- Vehtari, Gelman, & Gabry (2017). Practical Bayesian model evaluation using LOO-CV and WAIC.
- Betancourt (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. arXiv:1701.02434.
