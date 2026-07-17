# Confidence Intervals for Maximum Likelihood Curve Fitting

## Overview

This package provides three methods for constructing confidence intervals and confidence bands around a model fitted by Maximum Likelihood Estimation (MLE):

1. **Delta Method**
2. **Profile Likelihood**
3. **Monte Carlo Propagation**

Although the underlying statistical techniques differ considerably, all three methods return the same object:

```python
ConfidenceBand
```

This provides a unified interface for evaluating confidence intervals, plotting confidence bands, and querying the fitted model.

---

# Mathematical Background

Suppose we fit a model

```math
y=f(x;\boldsymbol{\theta})
```

using Maximum Likelihood Estimation.

The fitting procedure returns

```python
theta_hat
cov
```

where

* `theta_hat` is the maximum likelihood estimate of the parameter vector,
* `cov` is the estimated covariance matrix of the MLE.

Throughout this package we assume the standard asymptotic result

```math
\hat{\boldsymbol{\theta}}
\sim
N(\boldsymbol{\theta},\Sigma),
```

where

```math
\Sigma
=
\operatorname{Cov}(\hat{\boldsymbol{\theta}}).
```

The covariance matrix should already represent the covariance of the estimator itself.

**Do not divide it by the sample size.**

---

## Invariance of the Maximum Likelihood Estimator

A fundamental property of MLEs is that

```math
\widehat{f(\boldsymbol{\theta})}
=
f(\hat{\boldsymbol{\theta}}).
```

Consequently,

```math
f(x;\hat{\boldsymbol{\theta}})
```

is itself an estimator.

The purpose of this package is to quantify the uncertainty of this estimator.

---

# Package Architecture

```
                 Data
                  │
                  ▼
      Maximum Likelihood Fit
                  │
                  ▼
      θ̂ , Cov(θ̂), log-likelihood
                  │
        ┌─────────┼─────────┐
        │         │         │
        ▼         ▼         ▼
     Delta     Profile     Monte Carlo
     Method    Likelihood  Propagation
        │         │         │
        └─────────┼─────────┘
                  ▼
          ConfidenceBand
                  │
        ┌─────────┼─────────┐
        │         │         │
        ▼         ▼         ▼
   interval(x)  mean(x)  evaluate(x_grid)
```

All three confidence interval methods produce exactly the same public interface.

Only the internal computation differs.

---

# ConfidenceBand

The `ConfidenceBand` class is the common output type for all confidence interval engines.

Internally it supports two different representations.

## Analytic Confidence Bands

Used by:

* Delta Method

The confidence band is represented by two functions:

* the fitted model

```python
mean_func(x)
```

* the variance

```python
variance_func(x)
```

Confidence intervals are computed analytically whenever requested.

No interpolation is performed.

---

## Interpolated Confidence Bands

Used by:

* Profile Likelihood
* Monte Carlo

These methods first compute exact confidence intervals at many values of `x`.

The resulting confidence limits are then interpolated.

Interpolation is therefore **only a convenience for rapid evaluation**.

The underlying confidence limits were computed exactly using the chosen statistical method.

---

## Common Interface

Every `ConfidenceBand` supports

```python
band.interval(x)
```

Returns

```python
(lower, upper)
```

---

```python
band.mean(x)
```

Returns

```python
mean
```

---

```python
band.evaluate(x_grid)
```

Returns

```python
{
    "x": ...,
    "mean": ...,
    "lower": ...,
    "upper": ...
}
```

---

# Delta Method

## Overview

The Delta Method is the fastest confidence interval method implemented in the package.

It propagates uncertainty analytically using derivatives of the fitted model.

The implementation supports

* First-order Delta Method
* Higher-order Delta Method

The higher-order implementation incorporates Hessian and third-derivative tensor corrections.

---

## Required Inputs

```python
func(x, theta)
grad_func(x, theta)
theta_hat
cov
```

For higher-order propagation:

```python
hess_func(x, theta)
third_func(x, theta)
```

---

## First-Order Delta Method

```python
delta = DeltaMethodEngine(
    func=func,
    theta_hat=theta,
    cov=cov,
    grad=grad_func,
)
```

Construct an analytic confidence band

```python
band = delta.band(
    alpha=0.05,
    order=1,
)
```

Evaluate a confidence interval

```python
band.interval(2.5)
```

Evaluate on a plotting grid

```python
result = band.evaluate(x_grid)
```

---

## Higher-Order Delta Method

```python
delta = DeltaMethodEngine(
    func=func,
    theta_hat=theta,
    cov=cov,
    grad=grad_func,
    hess=hess_func,
    third=third_func,
)
```

Construct the confidence band

```python
band = delta.band(
    alpha=0.05,
    order=2,
)
```

The returned object is still a `ConfidenceBand`.

Unlike the other methods, confidence limits are evaluated analytically rather than by interpolation.

---

## When to Use

Recommended when

* computational speed is important,
* derivatives are available,
* the model is only mildly nonlinear,
* an analytic confidence band is desired.

---

# Profile Likelihood

## Overview

Profile Likelihood uses the likelihood surface directly rather than propagating covariance.

Confidence limits are obtained using Wilks' theorem.

For confidence level

```math
1-\alpha,
```

the allowed parameter region satisfies

```math
2
\left[
\ell(\hat{\theta})
-
\ell(\theta)
\right]
\le
\chi^2_{1,1-\alpha}.
```

For each value of `x`, the model is minimized and maximized over this likelihood region.

---

## Required Inputs

```python
func(x, theta)
loglik(theta)
theta_hat
bounds
```

---

## Exact Interval at One x

```python
profile = ProfileLikelihoodEngine(
    func=func,
    loglik=loglik,
    theta_hat=theta,
    bounds=bounds,
)

lower, upper = profile.interval(
    x=2.5,
    alpha=0.05,
)
```

Each call performs an exact constrained optimization.

---

## Confidence Band

Construct a confidence band

```python
band = profile.band(
    x_data=x_data,
    alpha=0.05,
    n_points=100,
)
```

The engine

1. computes exact Wilks confidence intervals at each grid point,
2. constructs an interpolated `ConfidenceBand`.

Subsequent calls such as

```python
band.interval(2.5)
```

require no further optimization.

---

## When to Use

Recommended when

* the likelihood function is available,
* nonlinear models are present,
* asymmetric confidence intervals are expected,
* a likelihood-based inference is preferred.

---

# Monte Carlo Propagation

## Overview

Monte Carlo propagation samples parameter vectors from

```math
\theta
\sim
N(\hat{\theta},\hat{\Sigma})
```

Each parameter sample is propagated through the model

```math
f(x;\theta).
```

The resulting empirical distribution is used to estimate confidence intervals.

---

## Required Inputs

```python
func(x, theta)
theta_hat
cov
```

---

## Exact Interval at One x

```python
mc = MonteCarloConfidence(
    func=func,
    theta_hat=theta,
    cov=cov,
)

lower, upper = mc.interval(
    x=2.5,
    alpha=0.05,
    n_samples=10000,
)
```

This performs a fresh Monte Carlo simulation for the requested value of `x`.

---

## Confidence Band

Construct a confidence band

```python
band = mc.band(
    x_data=x_data,
    alpha=0.05,
    n_points=100,
    n_samples=10000,
)
```

The engine

1. performs an independent Monte Carlo simulation at each grid point,
2. stores the resulting confidence limits,
3. returns an interpolated `ConfidenceBand`.

Subsequent evaluations require no further simulation.

---

## When to Use

Recommended when

* derivatives are unavailable,
* the model is strongly nonlinear,
* asymmetric confidence intervals are expected,
* computational cost is acceptable.

---

# Choosing a Method

| Method             | Speed | Robustness | Derivatives Required | Likelihood Required | Representation |
| ------------------ | :---: | :--------: | :------------------: | :-----------------: | -------------- |
| Delta Method       | ★★★★★ |    ★★☆☆☆   |          Yes         |          No         | Analytic       |
| Profile Likelihood | ★★★☆☆ |    ★★★★☆   |          No          |         Yes         | Interpolated   |
| Monte Carlo        | ★★☆☆☆ |    ★★★★★   |          No          |          No         | Interpolated   |

---

# Confidence Level

All methods accept

```python
alpha
```

Examples

```python
alpha = 0.10   # 90%
alpha = 0.05   # 95%
alpha = 0.01   # 99%
```

The package requires

```python
0 < alpha < 1
```

---

# Typical Workflow

```python
# Fit model
result = mle.fit(...)

# Extract results
theta = result.theta_hat
cov = result.cov

# Select confidence interval engine
engine = MonteCarloConfidence(
    func=func,
    theta_hat=theta,
    cov=cov,
)

# Construct confidence band
band = engine.band(
    x_data=x_data,
    alpha=0.05,
    n_points=100,
)

# Evaluate confidence interval
lower, upper = band.interval(2.5)

# Plot
grid = np.linspace(min(x_data), max(x_data), 300)

out = band.evaluate(grid)

plt.plot(out["x"], out["mean"])
plt.fill_between(
    out["x"],
    out["lower"],
    out["upper"],
    alpha=0.3,
)
```

---

# Notes

* The **Delta Method** produces an analytic confidence band. No interpolation is used.

* **Profile Likelihood** computes exact Wilks-theorem confidence limits at a finite number of points before interpolation.

* **Monte Carlo** computes exact Monte Carlo confidence limits at a finite number of points before interpolation.

* Interpolation is performed solely to accelerate repeated evaluation and plotting. It is **not** part of the underlying statistical method.

* This package computes **confidence intervals for the fitted mean function**. Prediction intervals, which additionally account for measurement noise, are implemented separately.
