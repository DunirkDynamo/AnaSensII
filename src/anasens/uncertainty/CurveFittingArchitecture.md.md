# Monte Carlo Inference Library

A modular Python library for model fitting via maximum likelihood, confidence interval estimation, and prediction interval construction.

The library is organized so that:

- **Models** define deterministic relationships.
- **Error models** define residual probability distributions.
- **Inference** performs MLE and covariance estimation.
- **Confidence interval methods** estimate uncertainty in the fitted model.
- **Prediction strategies** compute prediction intervals using either analytic or Monte Carlo methods.
- **Plotting** produces publication-quality figures.
- **Demo scripts** illustrate complete workflows.

---

## Project Structure

```text
mc_inference/
│
├── README.md
├── requirements.txt
├── demo.py
│
├── models/
│   ├── __init__.py
│   └── linear.py
│
├── error_models/
│   ├── __init__.py
│   ├── base.py
│   ├── gaussian.py
│   ├── laplace.py
│   ├── student_t.py
│   ├── logistic.py
│   └── skew_normal.py
│
├── inference/
│   ├── __init__.py
│   ├── mle.py
│   └── covariance.py
│
├── intervals/
│   ├── __init__.py
│   ├── base.py
│   ├── ci_engine.py
│   ├── delta.py
│   ├── monte_carlo.py
│   └── profile.py
│
├── prediction/
│   ├── __init__.py
│   ├── base.py
│   ├── gaussian_analytic.py
│   ├── monte_carlo.py
│   └── factory.py
│
├── plotting/
│   ├── __init__.py
│   └── publication.py
│
└── utils/
    ├── __init__.py
    ├── numerical_derivatives.py
    └── statistics.py
```

---

# Module Responsibilities

## `models/`

Contains deterministic model classes.

Examples:

- Linear regression
- Polynomial models
- Exponential models
- User-defined models (future)

Each model evaluates

\[
y=f(x,\theta)
\]

---

## `error_models/`

Defines additive residual distributions

\[
y=f(x,\theta)+\varepsilon
\]

Each error model implements

- `sample(x, rng)`
- `logpdf(residual, x)`

These automatically define both

- the likelihood for MLE
- the residual sampling used for prediction intervals

Current models:

- Gaussian
- Laplace
- Student-t
- Logistic
- Cauchy
- Skew-normal

---

## `inference/`

Performs statistical inference.

### `mle.py`

Responsible only for

- constructing the log-likelihood from the model and error model
- optimizing model parameters

### `covariance.py`

Computes

- numerical Hessians
- Fisher-information covariance estimates

---

## `intervals/`

Computes confidence intervals for the fitted model.

Each method implements a common interface.

Current methods:

- Delta Method
- Monte Carlo
- Profile Likelihood

`ci_engine.py`

Provides a unified API that dispatches to the selected confidence interval strategy.

---

## `prediction/`

Constructs prediction intervals.

Prediction strategies include

### GaussianAnalyticStrategy

Uses

- model variance
- error variance

to construct

\[
\sigma_{\text{pred}}^2(x)
=
\sigma_{\text{model}}^2(x)
+
\sigma_{\text{error}}^2(x)
\]

and returns prediction intervals analytically.

---

### MonteCarloStrategy

Samples

- model parameters
- residual error

to estimate prediction intervals numerically.

---

### `factory.py`

Automatically selects the optimal strategy.

For example

- Gaussian error → analytic prediction
- all other distributions → Monte Carlo prediction

---

## `plotting/`

Publication-quality plotting.

Produces figures containing

- observed data
- fitted model
- confidence band
- prediction band

Future additions might include

- residual plots
- QQ plots
- likelihood profiles

---

## `utils/`

Shared numerical routines.

Examples

- finite-difference derivatives
- gradient estimation
- Hessians
- statistical helper functions

No statistical inference occurs here.

---

## `demo.py`

Complete end-to-end example.

Workflow:

1. Generate synthetic data.
2. Select a model.
3. Select an error model.
4. Perform maximum likelihood estimation.
5. Estimate parameter covariance.
6. Compute confidence intervals.
7. Compute prediction intervals.
8. Produce publication-quality plots.