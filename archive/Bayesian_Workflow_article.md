# Bayesian Workflow

**Authors:** Andrew Gelman, Aki Vehtari, Daniel Simpson, Charles C. Margossian, Bob Carpenter, Yuling Yao, Lauren Kennedy, Jonah Gabry, Paul-Christian Bürkner, Martin Modrák

**Date:** November 2, 2020

---

## Abstract

The Bayesian approach to data analysis provides a powerful way to handle uncertainty in all observations, model parameters, and model structure using probability theory. Probabilistic programming languages make it easier to specify and fit Bayesian models, but this still leaves us with many options regarding constructing, evaluating, and using these models, along with many remaining challenges in computation. Using Bayesian inference to solve real-world problems requires not only statistical skills, subject matter knowledge, and programming, but also awareness of the decisions made in the process of data analysis. All of these aspects can be understood as part of a tangled workflow of applied Bayesian statistics. Beyond inference, the workflow also includes iterative model building, model checking, validation and troubleshooting of computational problems, model understanding, and model comparison. We review all these aspects of workflow in the context of several examples, keeping in mind that in practice we will be fitting many models for any given problem, even if only a subset of them will ultimately be relevant for our conclusions.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Before Fitting a Model](#2-before-fitting-a-model)
3. [Fitting a Model](#3-fitting-a-model)
4. [Using Constructed Data to Find and Understand Problems](#4-using-constructed-data-to-find-and-understand-problems)
5. [Addressing Computational Problems](#5-addressing-computational-problems)
6. [Evaluating and Using a Fitted Model](#6-evaluating-and-using-a-fitted-model)
7. [Modifying a Model](#7-modifying-a-model)
8. [Understanding and Comparing Multiple Models](#8-understanding-and-comparing-multiple-models)
9. [Modeling as Software Development](#9-modeling-as-software-development)
10. [Example: Golf Putting](#10-example-of-workflow-involving-model-building-and-expansion-golf-putting)
11. [Example: Planetary Motion](#11-example-of-workflow-for-a-model-with-unexpected-multimodality-planetary-motion)
12. [Discussion](#12-discussion)

---

## 1. Introduction

### 1.1 From Bayesian Inference to Bayesian Workflow

Bayesian inference is the formulation and computation of conditional probability: p(θ|y) ∝ p(θ)p(y|θ). However, **Bayesian workflow** encompasses much more:

- Model building
- Inference
- Model checking and improvement
- Comparison of different models (not just for model choice or averaging, but to better understand the models)

In a typical Bayesian workflow, we fit a series of models—some poor choices in retrospect, some useful but flawed, and some ultimately worth reporting. The hopelessly wrong and seriously flawed models are unavoidable steps toward fitting useful models.

### 1.2 Why Do We Need a Bayesian Workflow?

We need workflow, rather than mere inference, for several reasons:

- **Computation can be challenging** — we often need to work through various steps including fitting simpler models, approximate computation, and exploration of the fitting process
- **We don't know the model ahead of time** — even when an acceptable model has been chosen, we generally want to expand it as we gather more data
- **Understanding requires comparison** — even with static data and a known model, understanding can best be achieved by comparing inferences from related models
- **Different models yield different conclusions** — presenting multiple models illustrates uncertainty in model choice

### 1.3 Workflow and Its Relation to Statistical Theory and Practice

The development of statistical methodology follows a progression:

> **Example → Case study → Workflow → Method → Theory**

"Workflow" is more general than an example but less precisely specified than a method. Many ideas started as hacks and eventually were formalized:

- Multilevel modeling formalizes empirical Bayes estimation
- Exploratory data analysis can be understood as predictive model checking
- Regularization methods (lasso, horseshoe) replaced ad hoc variable selection
- Gaussian processes serve as Bayesian replacements for kernel smoothing

### 1.4 Organizing the Many Aspects of Bayesian Workflow

The workflow is more tangled than typical textbook presentations due to:

1. **Computational difficulty** — requiring experimentation to compute or approximate the posterior
2. **Model complexity** — starting with models missing important features, gradually adding them
3. **Dynamic data** — data collection is often ongoing or can draw in related datasets
4. **Model comparison** — understanding models by comparing to alternatives

Statistics is all about uncertainty. Beyond data and parameters, we're often uncertain about fitting models correctly, setting up and expanding models, and interpreting results.

### 1.5 Aim and Structure of This Article

This article puts tacit knowledge of applied statistics into the open, targeting:

- Practitioners of applied Bayesian statistics (especially users of probabilistic programming languages like Stan)
- Developers of methods and software
- Researchers of Bayesian theory and methods

---

## 2. Before Fitting a Model

### 2.1 Choosing an Initial Model

The starting point is almost always to adapt what has been done before—using a model from a textbook, case study, or published paper applied to a similar problem. Templates can save time in model building and computing, and we should account for cognitive load for the person understanding results.

Sometimes we start simple and add features; other times we start big and strip down. Sometimes we consider multiple completely different approaches.

### 2.2 Modular Construction

A Bayesian model is built from modules that can be viewed as placeholders:

- Model data with normal distribution → replace with longer-tailed or mixture distribution
- Model latent regression as linear → replace with nonlinear splines or Gaussian processes
- Treat observations as exact → add measurement-error model
- Start with weak prior → make stronger when posterior includes unrealistic values

Thinking of components as placeholders takes pressure off model-building because you can always go back and generalize.

### 2.3 Scaling and Transforming Parameters

Parameters should be interpretable for practical and ethical reasons. This means:

- Natural scales
- Modeling as independent (if possible) or with interpretable dependence structure
- Separating out scale so unknown parameters are scale-free

For example, if a parameter is expected to be approximately 50, model on log(θ/50), so 0 corresponds to an interpretable value and differences on the log scale correspond to percentage changes.

### 2.4 Prior Predictive Checking

Prior predictive checks help understand the implications of a prior distribution in the context of a generative model. They provide a way to refine the model without using the data multiple times.

Key insight: Even independent priors on individual coefficients have different implications as the number of covariates increases. With more predictors, we need stronger priors (or enough data) to push the model away from extreme predictions.

A useful approach is to consider priors on outcomes and derive corresponding joint priors on parameters.

### 2.5 Generative and Partially Generative Models

Fully Bayesian data analysis requires a **generative model**—a joint probability distribution for all data and parameters. While Bayesian inference only needs the likelihood, Bayesian data analysis requires the generative model for:

- Predictive simulation
- Model checking

Models can be placed on a spectrum from least to most generative:

1. Non-generative methods (data summaries with no model)
2. Classical statistical models: p(y; θ) with no distribution for θ
3. Typical Bayesian models: p(y, θ|x) with unmodeled data x
4. Completely generative models: p(y, θ, x) with no "left out" data

---

## 3. Fitting a Model

Current state-of-the-art algorithms include:

- **Variational inference** — fast but possibly inaccurate approximation
- **Sequential Monte Carlo** — generalization of Metropolis
- **Hamiltonian Monte Carlo (HMC)** — uses gradient computation for efficient movement through continuous probability spaces

### 3.1 Initial Values, Adaptation, and Warmup

Markov chain simulation operates in multiple stages:

1. **Warmup phase** — moves simulations from unrepreresentative initial values toward the typical set
2. **Adaptation** — sets algorithm tuning parameters using warmup information
3. **Sampling phase** — ideally run until multiple chains have mixed

In model exploration, warmup has a third role: quickly flagging computationally problematic models.

### 3.2 How Long to Run an Iterative Algorithm

Recommended practice: Run until R̂ < 1.01 for all parameters and quantities of interest, and monitor the multivariate mixing statistic R*.

However, in early modeling stages, earlier stopping can make sense. Running many iterations for a newly-written model is similar to premature optimization in software engineering.

### 3.3 Approximate Algorithms and Approximate Models

There's a tradeoff between speed and accuracy:

- **Near the end of workflow** — require accurate exploration (usually MCMC)
- **At the beginning** — can make decisions based on large-scale features using simpler methods (empirical Bayes, Laplace approximation, INLA, variational inference)

An alternative view: an approximate algorithm is an exact algorithm for an approximate model.

### 3.4 Fit Fast, Fail Fast

An important goal is to fail fast when fitting bad models. Inappropriate and poorly fitting models can often be more difficult to fit—this is actually useful information.

The "fit fast, fail fast" principle: We want any attempt to fit an inappropriate model to fail quickly so we can move on to something more reasonable.

---

## 4. Using Constructed Data to Find and Understand Problems

### 4.1 Fake-Data Simulation

Working in a controlled setting where true parameters are known helps us understand:

- Data model and priors
- What can be learned from an experiment
- Validity of applied inference methods

The procedure:

1. Choose reasonable parameter values
2. Simulate fake dataset of same size, shape, and structure as original data
3. Fit the model to fake data
4. Check if inferences are close to assumed parameter values

If fake-data check fails, break down the model—go simpler until it works, then identify the problem.

Fake-data simulation is crucial because it's the only point where we can directly check inference on latent variables is reliable.

### 4.2 Simulation-Based Calibration (SBC)

A more comprehensive approach:

1. Draw model parameters from the prior
2. Simulate data conditional on these parameters
3. Fit the model to data
4. Compare posterior to simulated parameter values

By performing Bayesian inference across datasets simulated using parameters from the prior, we should recover the prior.

SBC is useful for evaluating how closely approximate algorithms match the theoretical posterior.

**Challenge:** SBC clashes with the tendency to specify priors wider than necessary. Weakly informative priors can cause simulated datasets to occasionally be extreme.

### 4.3 Experimentation Using Constructed Data

Fitting models to data simulated from different scenarios provides insight—not just about computational issues but about data and inference more generally.

Example: Simulating educational intervention studies with different treatment assignment mechanisms reveals how functional form assumptions interact with design balance.

---

## 5. Addressing Computational Problems

### 5.1 The Folk Theorem of Statistical Computing

> **When you have computational problems, often there's a problem with your model.**

Not always—sometimes models are legitimately difficult to fit—but many cases of poor convergence correspond to:

- Regions of parameter space not of substantive interest
- Nonsensical models
- Bugs in code

First instinct should not be to throw more computational resources at the problem, but to check for substantive pathology.

### 5.2 Starting at Simple and Complex Models and Meeting in the Middle

When a model is not performing well:

1. **From the complex side:** Gradually simplify the poorly-performing model until something works
2. **From the simple side:** Start with a simple, well-understood model and gradually add features until the problem appears

If the model has multiple components, perform unit tests—make sure each component can be fit separately using simulated data.

### 5.3 Getting a Handle on Models That Take a Long Time to Fit

Slow computation is often a sign of other problems. Suggestions for debugging:

- Simulate fake data and fit to that (removes lack of fit concerns)
- Start with smaller model and build up
- Run fewer iterations initially (200 instead of 2000)
- Use moderately informative priors
- Consider interactions of predictors
- Fit on a subset of data

Common theme: Think of any model choice as provisional.

### 5.4 Monitoring Intermediate Quantities

Save intermediate quantities and plot them along with MCMC output. Visualizations typically teach more than streams of numbers in the console.

When chains get stuck, plot predictions from the model given those parameter values to understand what's going wrong.

### 5.5 Stacking to Reweight Poorly Mixing Chains

When multiple chains are slow to mix but in a generally reasonable range, use stacking to combine simulations. Cross validation assigns weights to different chains, approximately discarding chains stuck in low-probability modes.

Non-uniform stacking weights, combined with traceplots and diagnostics, help understand where to focus effort.

### 5.6 Posterior Distributions with Multimodality and Other Difficult Geometry

Four types of problems:

1. **Effectively disjoint volumes with near-zero mass in all but one mode** — avoid minor modes with judicious initial values or prior information
2. **Trivially symmetric disjoint volumes** (e.g., label switching in mixtures) — restrict model to identify mode of interest
3. **Different disjoint volumes of high probability** — use stacking, strong mixture priors, or strong priors ruling out some modes
4. **Single volume with arithmetically unstable tail** — initialize near the mass; reparameterize if interested in rare events

### 5.7 Reparameterization

HMC works best when:

- Mass matrix is appropriately tuned
- Posterior geometry is relatively uninteresting (no sharp corners, cusps, irregularities)

For complex models, judiciously choosing parameterization can greatly improve behavior. Hierarchical models can have funnel pathologies when group-level variance approaches zero, but reparameterization often resolves these.

### 5.8 Marginalization

Challenging geometries often arise from interactions between parameters. By marginalizing out some parameters, we can efficiently draw MCMC samples from the marginal posterior, then recover the marginalized parameters by exact sampling from the conditional distribution.

### 5.9 Adding Prior Information

Problems in computation can often be fixed by including prior information that was available but not yet included.

Consider a ladder of abstraction:

1. Poor mixing of MCMC
2. Difficult geometry as mathematical explanation
3. Weakly informative data as statistical explanation
4. Substantive prior information as solution

Starting from the beginning: computational troubleshooting. Starting from the end: computational workflow.

### 5.10 Adding Data

Similar to adding prior information, one can constrain the model by adding new data sources. Models well-behaved for larger datasets can have computational issues in small data regimes.

---

## 6. Evaluating and Using a Fitted Model

### 6.1 Posterior Predictive Checking

Analogous to prior predictive checking, but parameter draws come from the posterior rather than the prior.

When comparing simulated datasets to actual data, if the data are unrepresentative of the posterior predictive distribution, this indicates model failure.

There's no general way to choose which checks to perform or when a failed check requires model adjustment. Try to find "severe tests"—checks likely to fail if the model would give misleading answers to important questions.

### 6.2 Cross Validation and Influence of Individual Data Points

Posterior predictive checking uses data for both fitting and evaluation, which can be overly optimistic. Cross validation leaves out part of the data, fits to the rest, and checks predictive performance on left-out data.

Three useful diagnostic approaches:

1. **Calibration checks** using cross validation predictive distribution
2. **Identifying difficult-to-predict observations** or groups
3. **Identifying influential observations** — how much information they provide

Leave-one-out cross validation (LOO-CV) can be efficiently approximated using importance sampling.

### 6.3 Influence of Prior Information

Understanding how posterior inferences depend on data and priors:

- **Generative perspective:** How parameters map to data (prior predictive simulation)
- **Inferential perspective:** Path from inputs (data, priors) to outputs (estimates, uncertainties)

Approaches to understanding prior influence:

- **Sensitivity analysis:** Refit with multiple priors
- **Shrinkage comparison:** Compare prior to posterior standard deviations
- **Importance sampling:** Approximate posterior of new model using old model's posterior
- **Static sensitivity analysis:** Study sensitivity without refitting

### 6.4 Summarizing Inference and Propagating Uncertainty

Bayesian inference is well-suited for problems with latent variables and unresolvable uncertainty. However, usual displays don't fully capture multiple levels of variation and uncertainty.

Follow general principles:

- Graph data and fitted models
- Use exploratory data analysis to uncover unexpected patterns
- Understand how the model relates to the data

---

## 7. Modifying a Model

Model building is language-like: putting together existing components (linear, logistic, exponential functions; additive and multiplicative models; various distributions; varying coefficients) to encompass new data and features.

### 7.1 Constructing a Model for the Data

In applications, set up a data model based on:

- Fit to data (from posterior predictive checks)
- Domain expertise

Often the most important aspect is not distributional form but how data link to underlying parameters of interest.

### 7.2 Incorporating Additional Data

A key part of workflow is expanding a model to use more data. This can be as simple as adding regression predictors, but when more parameters are added, it may be necessary to assume not all can have big effects simultaneously.

### 7.3 Working with Prior Distributions

Think of a ladder of possibilities:

1. (Improper) flat prior
2. Super-vague but proper prior
3. Very weakly informative prior
4. Generic weakly informative prior
5. Specific informative prior

Another view: a prior distribution is a constraint. The amount of prior information needed depends strongly on the parameter's role and depth in the hierarchy.

**Important:** As models become more complex, all priors often need to become tighter. The information budget must be divided among more parameters.

### 7.4 A Topology of Models

Models in any framework can be structured as a partial ordering. Our interest is not in assigning probabilities but in navigating among models.

The topology includes:

- Partial ordering of models
- Connections between parameters across different models
- Prior distributions adding continuous dimensions, bridging between models

---

## 8. Understanding and Comparing Multiple Models

### 8.1 Visualizing Models in Relation to Each Other

We fit multiple models for several reasons:

- Hard to fit and understand big models directly
- We make mistakes (typos, coding errors, conceptual errors)
- More data requires model expansion
- Well-specified models reveal opportunities for expansion
- Models start with placeholders that get refined
- Problems found in checking lead to expansion
- Simple models serve as comparisons
- Computational problems motivate changes

Given multiple models, we must be concerned with researcher degrees of freedom and overfitting.

### 8.2 Cross Validation and Model Averaging

Use cross validation to compare models fit to the same data. When there's non-negligible uncertainty in comparison, don't simply choose the best model—use stacking to combine inferences.

Stacking makes more sense than traditional Bayesian model averaging, which can depend strongly on aspects with minimal effect on predictions.

### 8.3 Comparing a Large Number of Models

When many candidate models exist as special cases of an expanded model, we're often interested in finding a smaller model with the same predictive performance.

**Projection predictive variable selection** is stable and reliable for finding smaller models with good predictive performance. It avoids overfitting by examining only projected submodels based on the expanded model's predictions.

---

## 9. Modeling as Software Development

Developing a statistical model in a probabilistic programming language is software development with several stages:

- Writing and debugging the model
- Preprocessing data
- Understanding, communicating, and using inferences

### 9.1 Version Control

Version control (e.g., Git) should be the first infrastructure for a project. Benefits:

- Revert to previously working versions
- Get differences between versions
- Share work and automatically merge
- Keep notes on findings and decisions
- Track reports, graphs, and data
- Package "release candidate" versions

Public repositories increase transparency about what was used for specific reports.

### 9.2 Testing as You Go

- **Top-down design:** From end-user goals to technical machinery
- **Bottom-up development:** From well-tested foundational functions to larger modules

The key is **modularity**. Big tangled functions are hard to document, read, debug, and maintain. Whenever code fragments are repeated, encapsulate them as functions.

### 9.3 Making It Essentially Reproducible

Goals:

- Another person on another machine could recreate the final report
- Produce essentially equivalent analyses and figures

Write scripts rather than entering commands on the command line. Scripts should be self-contained—not depending on global variables or other data not in the script.

### 9.4 Making It Readable and Maintainable

Treat programs like other forms of writing for an audience. Key principles:

- Consistency in naming and layout
- Avoid repetition (pull shared code into functions)
- Write readable code, not opaque code with comments

Example of good practice:

```stan
// Instead of:
real x17; // oxygen level, should be positive

// Write:
real<lower = 0> oxygen_level;
```

Keep code modular so fixing an error requires changing code in one place rather than many.

---

## 10. Example of Workflow Involving Model Building and Expansion: Golf Putting

This example demonstrates basic Bayesian modeling workflow using data on golf putting success rates as a function of distance from the hole.

### 10.1 First Model: Logistic Regression

Natural starting point:

$$y_j \sim \text{binomial}(n_j, \text{logit}^{-1}(a + bx_j))$$

### 10.2 Modeling from First Principles

A geometric model based on the physics: the ball must be hit within a threshold angle to go in the hole. Assuming the golfer's angle follows a normal distribution centered at 0 with standard deviation σ:

$$\Pr(\text{success}) = 2\Phi\left(\frac{\sin^{-1}((R-r)/x)}{\sigma}\right) - 1$$

This one-parameter model fits the data much better than logistic regression.

### 10.3 Testing the Fitted Model on New Data

New, more comprehensive data revealed:

- Higher success rates for short putts (possibly measurement issues or improved golfers)
- Systematic model failure at higher distances

### 10.4 A New Model Accounting for How Hard the Ball Is Hit

Adding a second parameter for distance control: the ball must be hit hard enough to reach the hole but not so hard it hops over.

This two-parameter model initially showed poor convergence—the binomial likelihood constrained too strongly at short distances where counts were highest.

### 10.5 Expanding the Model by Including a Fudge Factor

To allow for model misspecification, add an independent error term:

$$y_j/n_j \sim \text{normal}\left(p_j, \sqrt{p_j(1-p_j)/n_j + \sigma_y^2}\right)$$

This three-parameter model fits with no problems and provides excellent fit to the data.

### 10.6 General Lessons

- Simple one-parameter model fit initial data; new data required one more parameter
- Binomial likelihood can be too strong, making it difficult to fit all data
- Bigger data need bigger models
- Inferences from a sequence of models can be compared by graphing predictions and studying parameter estimate changes
- Even the final model is a work in progress

---

## 11. Example of Workflow for a Model with Unexpected Multimodality: Planetary Motion

This example illustrates starting with a complicated model, encountering problems, and figuring out what's going on.

### 11.1 Mechanistic Model of Motion

Using Newton's laws (Hamilton's formulation) to describe planetary motion:

$$\frac{dq}{dt} = \frac{p}{m}, \quad \frac{dp}{dt} = -\frac{k}{r^3}(q - q^*)$$

Goal: Infer the gravitational force parameter k, along with initial position and momentum.

First attempt fails dramatically—chains don't converge and take a long time.

### 11.2 Fitting a Simplified Model

Simplify to only estimate k with other parameters fixed. Observations:

- Run time varies widely between chains (2 to 2000 seconds)
- R̂ is large, indicating chains haven't mixed
- Traceplots show chains stuck at local modes

The local modes arise from the periodic structure: as k changes and the orbit rotates, some observed and simulated positions happen to become relatively close, inducing wiggles in the likelihood tail.

### 11.3 Bad Markov Chain, Slow Markov Chain?

Chains with lowest log posterior and highest k were also slowest—an instance of the folk theorem.

An easy deterministic problem can become difficult in Bayesian analysis because inference requires solving across a range of parameter values.

### 11.4 Building Up the Model

Gradually build back to the original model, using insights from the simplified model. Most inference problems can be traced to interaction between likelihood and cyclical observations.

### 11.5 General Lessons

- Examining simplified models helps understand challenges
- Multimodal posteriors prevent cohesive exploration; understand how modes arise
- Starting points matter in non-asymptotic regimes
- Don't mindlessly discard misbehaving chains—analyze where poor behavior comes from
- Adjusting initial estimates should be based on understanding, not convenience

---

## 12. Discussion

### 12.1 Different Perspectives on Statistical Modeling and Prediction

Three perspectives:

1. **Traditional statistical:** Model chosen ahead of time; goal is accurate posterior summary
2. **Machine learning:** Goal is prediction; computation stops when cross validation accuracy plateaus
3. **Model exploration:** Much effort spent trying models with terrible fit, poor performance, slow convergence

These imply different inferential goals and different tolerance for approximations.

### 12.2 Justification of Iterative Model Building

Model navigation is the next transformative step in data science, following:

1. Data summarization (up to 1900)
2. Modeling (Gauss and Laplace to present)
3. Computation (current)

From a **human perspective:** Limited cognitive capabilities make gradual learning easier. Building rich models takes effort; efficient to start simple and stop when sufficient.

From a **computational perspective:** Given finite resources, algorithms with asymptotic guarantees can fail in finite time. Deconstructing into simpler versions helps understand computational challenges.

### 12.3 Model Selection and Overfitting

Potential issue: Model improvement conditioned on discrepancy between model and data means data are used more than once.

However, Bayesian workflow avoids worst overfitting problems because we don't search for optimally-fitting models or make hard selection under uncertainty. Instead, we use problems to reassess choices and include additional information.

Often a model whose assumptions withstood severe tests is more trustworthy than a preregistered model that hasn't been tested at all.

### 12.4 Bigger Datasets Demand Bigger Models

"Big data" doesn't alleviate the need for careful modeling. Big data is messy data—observational rather than experimental, using proxies rather than direct measurements.

To make relevant inferences requires:

- Extrapolating from sample to population
- Extrapolating from control to treatment group
- Extrapolating from measurements to latent variables

Each step requires statistical assumptions and adjustment.

### 12.5 Prediction, Generalization, and Poststratification

The three core tasks of statistics:

1. Generalizing from sample to population
2. Generalizing from control to treatment group
3. Generalizing from observed data to underlying constructs

Bayesian workflow doesn't stop with inference for the fitted model. We're also interested in inferences for new real-world situations.

### 12.6 Going Forward

All workflows have holes. To ultimately check the model and push workflow forward, we often need to collect more data, expanding the model along the way.

The plan:

1. Abstract principles from current best practice (this article)
2. Apply workflow to applied problems as case studies
3. Implement as much as possible in software tools

The ultimate goal: Enable data analysts to use statistical modeling more effectively and build confidence in inferences and decisions.

---

## References

*Note: The original paper contains an extensive reference list. Key references include:*

- Carpenter et al. (2017) — Stan probabilistic programming language
- Gabry et al. (2019) — Visualization in Bayesian workflow
- Gelman et al. (2013) — Bayesian Data Analysis
- Vehtari et al. (2017) — Practical Bayesian model evaluation using leave-one-out cross-validation
- Talts et al. (2020) — Simulation-based calibration

---

*This markdown document is a conversion of "Bayesian Workflow" by Gelman et al. (2020). The original paper contains additional figures, detailed mathematical notation, and comprehensive references.*
