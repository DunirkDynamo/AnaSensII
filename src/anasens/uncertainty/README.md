# Monte Carlo Inference

A modular Python library for maximum likelihood estimation, confidence interval construction, and prediction interval estimation for arbitrary deterministic models.

The library is designed around a simple statistical model:

\[
y = f(x;\theta) + \varepsilon,
\]

where

- \(f(x;\theta)\) is a deterministic model,
- \(\theta\) are unknown model parameters,
- \(\varepsilon\) is a user-selected residual (error) distribution.

Unlike ordinary least squares libraries that assume Gaussian errors, this package allows the residual distribution to be chosen independently of the deterministic model. The selected residual distribution automatically defines both the likelihood used during maximum likelihood estimation and the prediction interval construction.

---

# Features

- Maximum likelihood estimation (MLE)
- Multiple residual error distributions
    - Gaussian
    - Laplace (LAD)
    - Student's t
    - Logistic
    - Cauchy
    - Skew-normal
- Numerical covariance estimation from the observed Fisher information
- Confidence intervals for the fitted model
    - Delta Method
    - Monte Carlo propagation
    - Profile likelihood
- Prediction intervals
    - Closed-form analytic solution for Gaussian residuals
    - Monte Carlo simulation for arbitrary residual distributions
- Publication-quality plotting

---

# Philosophy

The package separates the statistical problem into independent components.

## Model

The deterministic relationship

\[
f(x;\theta)
\]

describes the expected value of the observations.

Examples include

- linear
- polynomial
- exponential
- logistic
- user-defined models

The model contains **no statistical assumptions**.

---

## Error Model

The error model specifies the probability distribution of the residual

\[
r = y - f(x;\theta).
\]

Each error model provides two operations:

- Evaluate the log probability density of a residual.
- Generate random residual samples.

The same error model is used consistently throughout the inference process:

- maximum likelihood estimation
- covariance estimation
- Monte Carlo confidence intervals
- prediction intervals

This guarantees statistical consistency.

---

## Inference

Model parameters are estimated by maximum likelihood.

The likelihood is automatically constructed from the selected error model.

For observations

\[
(x_i,y_i),
\]

the log-likelihood is

\[
\log L(\theta)
=
\sum_i
\log
p_\varepsilon
\left(
y_i-f(x_i;\theta)
\right),
\]

where

\[
p_\varepsilon
\]

is the residual probability density.

---

## Confidence Intervals

The package supports multiple approaches for estimating uncertainty in the fitted model.

### Delta Method

Uses first-order (and later higher-order) propagation of parameter uncertainty.

Fast and appropriate for mildly nonlinear models.

---

### Monte Carlo

Draws parameter vectors from the asymptotic sampling distribution

\[
\theta
\sim
\mathcal N(\hat\theta,\Sigma)
\]

and propagates the uncertainty through the model numerically.

Useful for nonlinear models.

---

### Profile Likelihood

Constructs confidence intervals directly from the likelihood surface.

Provides a likelihood-based alternative that often performs well for nonlinear models.

---

## Prediction Intervals

Prediction intervals incorporate both

- parameter uncertainty
- observational noise

For Gaussian residual models, prediction intervals are computed analytically.

For all other residual distributions, prediction intervals are estimated using Monte Carlo simulation.

---

# Package Structure

```text
mc_inference/
│
├── README.md
├── requirements.txt
├── demo.py
│
├── models/
├── error_models/
├── inference/
├── intervals/
├── prediction/
├── plotting/
└── utils/
```

Each package has a single responsibility.

---

# Example Workflow

```python
from models.linear import LinearModel
from error_models.gaussian import GaussianError
from inference.mle import MLE

model = LinearModel()

error = GaussianError(sigma=0.5)

mle = MLE(model, error, x, y)

theta_hat = mle.fit(initial_guess)
```

Estimate the parameter covariance.

```python
cov = covariance_from_loglik(
    mle.loglik,
    theta_hat
)
```

Construct confidence intervals.

```python
ci = MonteCarloCI(
    model,
    theta_hat,
    cov
)
```

Construct prediction intervals.

```python
prediction = PredictionEngine(
    model=model,
    theta_hat=theta_hat,
    covariance=cov,
    error_model=error
)
```

Generate publication-quality plots.

```python
plot_results(...)
```

---

# Dependencies

- NumPy
- SciPy
- Matplotlib

---

# Future Development

Planned additions include

- symbolic likelihood construction using SymPy
- automatic differentiation
- higher-order delta method
- bootstrap confidence intervals
- generalized linear models
- Bayesian inference
- inverse regression (Feldman–Cousins-style confidence belts)
- HDF5 project storage
- support for user-defined symbolic residual distributions

---

# License

MIT License.