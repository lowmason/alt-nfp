# Visualization in Bayesian Workflow

**Authors:**
- Jonah Gabry† (Department of Statistics and ISERP, Columbia University, New York, USA)
- Daniel Simpson† (Department of Statistical Sciences, University of Toronto, Canada)
- Aki Vehtari (Department of Computer Science, Aalto University, Espoo, Finland)
- Michael Betancourt (ISERP, Columbia University, and Symplectomorphic, LLC, New York, USA)
- Andrew Gelman (Departments of Statistics and Political Science, Columbia University, New York, USA)

†Joint first author

**Source:** arXiv:1709.01449v5 [stat.ME] 9 Jun 2018

---

## Summary

Bayesian data analysis is about more than just computing a posterior distribution, and Bayesian visualization is about more than trace plots of Markov chains. Practical Bayesian data analysis, like all data analysis, is an iterative process of model building, inference, model checking and evaluation, and model expansion. Visualization is helpful in each of these stages of the Bayesian workflow and it is indispensable when drawing inferences from the types of modern, high-dimensional models that are used by applied researchers.

---

## 1. Introduction and Running Example

Visualization is a vital tool for data analysis, and its role is well established in both the exploratory and final presentation stages of a statistical workflow. The same visualization tools should be used at all points during an analysis.

The paper follows a single real example—estimating the global concentration of PM2.5 air pollution—through all phases of statistical workflow:

1. **Exploratory data analysis** to aid in setting up an initial model
2. **Computational model checks** using fake-data simulation and the prior predictive distribution
3. **Computational checks** to ensure the inference algorithm works reliably
4. **Posterior predictive checks** and other juxtapositions of data and predictions under the fitted model
5. **Model comparison** via tools such as cross-validation

The tools are implemented in the **bayesplot** R package (Gabry, 2017), which uses ggplot2 and is linked to Stan.

### Running Example: PM2.5 Air Pollution

The example estimates human exposure to air pollution from particulate matter measuring less than 2.5 microns in diameter (PM2.5). PM2.5 is linked to poor health outcomes and is estimated to be responsible for three million deaths worldwide each year (Shaddick et al., 2017).

**The statistical problem:** Only direct measurements from a sparse network of 2,980 ground monitors with heterogeneous spatial coverage are available. The monitoring network has especially poor coverage across Africa, central Asia, and Russia.

**Solution approach:** Supplement direct measurements with high-resolution satellite data that converts measurements of aerosol optical depth into estimates of PM2.5. The goal is to use ground monitor data to calibrate the approximate satellite measurements.

---

## 2. Exploratory Data Analysis Goes Beyond Just Plotting the Data

Exploratory data analysis is more than simply plotting the data. It is a method to build a network of increasingly complex models that can capture the features and heterogeneities present in the data (Gelman, 2004).

### Ground-Up vs. Top-Down Modeling

**Ground-up strategy:** Build a network of models knowing the limitations of the design. Particularly useful when data are sparse or unbalanced.

**Top-down strategy (common in ML):** Throw all available information into a complicated non-parametric procedure. Works well for representative data but can be prone to over-fitting or generalization error on sparse/unbalanced data.

### Three Models Developed

1. **Model 1:** Simple linear regression (assumes satellite data is a good predictor after affine adjustment)
2. **Model 2:** Multilevel model stratified by WHO super-region
3. **Model 3:** Multilevel model stratified by clustered super-region (6-component hierarchical clustering of ground monitor measurements)

The exploratory analysis revealed:
- A single linear regression would introduce ecological bias
- Some regions (Sub-Saharan Africa, certain clusters) don't have enough data to pin down linear trends
- Borrowing of strength through multilevel modeling may be appropriate

---

## 3. Fake Data Can Be Almost as Valuable as Real Data for Building Your Model

With proper priors, Bayesian models are generative models. We can visualize simulations from the prior marginal distribution of the data to assess the consistency of chosen priors with domain knowledge.

### Prior Predictive Distribution

The main advantage of assessing priors based on the prior marginal distribution for data is that it reflects the interplay between the prior distribution on parameters and the likelihood.

### Weakly Informative Joint Prior

A prior leads to a **weakly informative joint prior data generating process** if draws from the prior data generating distribution p(y) could represent any data set that could plausibly be observed:
- Should have some mass around extreme but plausible data sets
- Should have no mass on completely implausible data sets

### Example: PM2.5 Multilevel Model

Mathematical form:
$$y_{ij} \sim N(\beta_0 + \beta_{0j} + (\beta_1 + \beta_{1j})x_{ij}, \sigma^2)$$
$$\beta_{0j} \sim N(0, \tau_0^2), \quad \beta_{1j} \sim N(0, \tau_1^2)$$

**Vague priors (problematic):**
- βₖ ~ N(0, 100), τₖ² ~ Inv-Gamma(1, 100)
- Generate completely impossible data for this application

**Weakly informative priors (recommended):**
- β₀ ~ N(0, 1), β₁ ~ N(1, 1), τₖ ~ N₊(0, 1)
- Centered around models with intercept 0 and slope 1 (since satellite estimates are reasonably faithful)
- Still can generate extreme but plausible data

---

## 4. Graphical MCMC Diagnostics: Moving Beyond Trace Plots

Traditional MCMC diagnostic plots (trace plots, autocorrelation functions) can be helpful but are not always needed when chains mix well.

### Hamiltonian Monte Carlo Diagnostics

For HMC and its variants, we can get much more detailed information about Markov chain performance (Betancourt, 2017).

**Key insight:** HMC success requires that the geometry of the typical set (the set containing bulk of posterior probability mass) is fairly smooth. Non-smooth geometry causes the leap-frog integrator to rapidly diverge from the energy-conserving trajectory.

### Useful Diagnostic Plots

1. **Bivariate scatterplots** marking divergent transitions (`mcmc_scatter` in bayesplot)
2. **Parallel coordinate plots** (`mcmc_parcoord` in bayesplot)

### Interpreting Divergences

- **Concentration of divergences in small neighborhoods:** Indicates a region of high curvature that obstructs exploration
- **No obvious pattern to divergences:** Likely false positives
- These visualizations can differentiate between models with non-smooth typical sets and false positive heuristics

---

## 5. Posterior Predictive Checks Are Vital for Model Evaluation

**Core idea:** If a model is a good fit, we should be able to use it to generate data that resemble the observed data.

### Posterior Predictive Distribution

$$p(\tilde{y} | y) = \int p(\tilde{y} | \theta) p(\theta | y) d\theta$$

where y is current data, ỹ is new data to be predicted, and θ are model parameters.

### Key Visualizations

1. **Density overlays** (`ppc_dens_overlay`): Compare distributions of replicated datasets to observed data
2. **Test statistics** (`ppc_stat`): Check statistics orthogonal to model parameters (e.g., skewness for Gaussian models)
3. **Grouped checks** (`ppc_stat_grouped`): Check within levels of grouping variables

### Leave-One-Out Cross-Validation

LOO cross-validation partially avoids double use of data. The LOO probability integral transforms (LOO-PIT) should be asymptotically uniform for calibrated models.

**Interpreting LOO-PIT plots:**
- "Frown" shapes indicate univariate predictive distributions are too broad compared to data
- Suggests further modeling is necessary to accurately reflect uncertainty

---

## 6. Pointwise Plots for Predictive Model Comparison

Visual posterior predictive checks help identify unusual points:
- **Outliers:** Difficult to predict
- **High leverage points:** Influential on the posterior

### Tools

**LOO predictive distribution:** p(yᵢ | y₋ᵢ)

**ELPD comparison:** Compare expected log predictive densities for individual data points using PSIS-LOO (Pareto-smoothed importance sampling).

### PSIS-LOO k̂ Diagnostic

The k̂ diagnostic estimates how similar the full-data predictive distribution is to the LOO predictive distribution:
- **Large k̂ value:** Indicates a highly influential observation
- Useful for identifying observations that need special attention

### Example Finding

The only observation from Mongolia was identified as highly influential under Model 2 (k̂ ≈ 0.5 under Model 3, significantly lower), indicating Model 3 better resolves this data point.

---

## 7. Discussion

Visualization can be used as part of a strategy to:
- Compare models
- Identify ways in which a model fails to fit
- Check how well computational methods have resolved the model
- Understand the model well enough to set priors
- Iteratively improve the model

### On Using Data Twice

**Concern:** Using measured data to guide model building may cause poor generalization.

**Mitigations:**
1. **Prior predictive checks:** Don't cleave too closely to observed data; aim for a prior data generating process that can produce plausible (not indistinguishable) data sets
2. **Posterior predictive checks:** Check carefully for influential measurements; propose weakly informative extensions centered on the previous model (Simpson et al., 2017)

---

## Supplementary Material: The 8-Schools Problem

The 8-schools problem (Rubin, 1981; Gelman et al., 2013) demonstrates visualization of divergent trajectories.

### Diagnosing the Problem

- **Bivariate plot:** Divergences concentrate in a particular region (the neck of a funnel shape)
- **Parallel coordinates plot:** Divergent paths have small τ values, resulting in little variability in θⱼ's

### Solution: Reparameterization

Funnels in parameter space can be resolved through reparameterization:
- Move to a **non-centered parameterization** where the narrowest coordinate is made a priori independent of other coordinates
- This fattens out the funnel and removes the cluster of divergences

---

## Key bayesplot Functions

| Function | Purpose |
|----------|---------|
| `ppc_dens_overlay` | Density overlay of observed vs. replicated data |
| `ppc_stat` | Histogram of test statistics from posterior predictive |
| `ppc_stat_grouped` | Grouped posterior predictive statistics |
| `ppc_loo_pit` | LOO probability integral transform checks |
| `mcmc_scatter` | Bivariate scatterplot with divergence markers |
| `mcmc_parcoord` | Parallel coordinates plot with divergence markers |

---

## References

- Betancourt, M. (2017). A conceptual introduction to Hamiltonian Monte Carlo. arXiv:1701.02434.
- Betancourt, M. and Girolami, M. (2015). Hamiltonian Monte Carlo for hierarchical models. In Current Trends in Bayesian Methodology with Applications, pp. 79–101. Chapman & Hall.
- Gabry, J. (2017). bayesplot: Plotting for Bayesian models. R package version 1.3.0.
- Gelman, A. (2004). Exploratory data analysis for complex models. Journal of Computational and Graphical Statistics 13(4), 755–779.
- Gelman, A., Carlin, J.B., Stern, H.S., Dunson, D.B., Vehtari, A., and Rubin, D.B. (2013). Bayesian Data Analysis (Third ed.). Chapman & Hall/CRC.
- Gelman, A., Simpson, D., and Betancourt, M. (2017). The prior can generally only be understood in the context of the likelihood. arXiv:1708.07487.
- Shaddick, G. et al. (2017). Data integration model for air quality: a hierarchical approach to the global estimation of exposures to ambient air pollution. JRSS-C.
- Simpson, D. et al. (2017). Penalising model component complexity: A principled, practical approach to constructing priors. Statistical Science 32(1), 1–28.
- Vehtari, A., Gelman, A., and Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. Statistics and Computing 27(5), 1413–1432.

---

**Code and data:** https://github.com/jgabry/bayes-vis-paper
