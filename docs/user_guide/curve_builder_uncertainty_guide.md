# Curve Builder Uncertainty Analysis Guide

## Purpose and Scope

This document explains the new uncertainty functionality integrated into AnaSens Curve Builder, with emphasis on statistical formulation, mathematical details, and practical usage in both GUI and programmatic workflows.

The integration now supports, directly in the Curve Builder tab:

- Confidence bands for the fitted mean curve using the Delta Method.
- Prediction bands for future observations using Gaussian analytic propagation.
- Combined plotting of:
  - observed data,
  - fitted model curve,
  - model confidence band,
  - observation prediction band.

The internal uncertainty toolkit also contains more general methods (MLE, Monte Carlo confidence/prediction, profile likelihood), which are described here for academic and future extension workflows.

## Statistical Model

We assume observations follow

$$
y_i = f(x_i; \theta) + \varepsilon_i,
$$

where:

- $f(x;\theta)$ is a deterministic model family,
- $\theta \in \mathbb{R}^p$ is the unknown parameter vector,
- $\varepsilon_i$ are residual errors.

For the current Curve Builder GUI integration, the operational residual model is Gaussian:

$$
\varepsilon_i \sim \mathcal{N}(0,\sigma^2).
$$

## Distinguishing Confidence and Prediction Bands

This distinction is central:

1. Confidence band (for fitted mean): uncertainty in $f(x;\hat\theta)$ only.
2. Prediction band (for new observations): uncertainty in future $y_{new}(x)$, centered on the fitted predictive location $f(x;\hat\theta)$ and including both parameter uncertainty and observation noise.

Consequently, prediction bands are always wider (or equal in degenerate limits) than confidence bands.

## Parameter Estimation and Covariance

### Curve-fitting estimate

In the current Curve Builder fit flow, model parameters are estimated via nonlinear least squares through SciPy fitting utilities, returning:

- $\hat\theta$ (point estimate),
- estimated covariance matrix $\widehat{\Sigma}$ from the optimizer when available.

### Degrees of freedom and residual scale

Let $n$ be the number of observations and $p$ be number of fitted parameters.

$$
\nu = \max(n-p,1)
$$

and residual standard deviation estimate:

$$
\hat\sigma = \sqrt{\frac{\sum_{i=1}^n (y_i - f(x_i;\hat\theta))^2}{\nu}}.
$$

### Covariance fallback strategy

If optimizer covariance is unavailable or invalid, the integration computes a numerical Jacobian $J$ and uses a pseudo-inverse fallback:

$$
\widehat{\Sigma} \approx (J^T J)^+.
$$

Here $(\cdot)^+$ denotes the Moore-Penrose pseudo-inverse.

## Confidence Band Mathematics (Delta Method)

### First-order Delta variance

At any $x$, define gradient

$$
g(x) = \nabla_\theta f(x;\theta)\big|_{\theta=\hat\theta}.
$$

Then first-order propagated variance of fitted mean is

$$
\operatorname{Var}[f(x;\hat\theta)] \approx g(x)^T \widehat{\Sigma} g(x).
$$

In implementation, derivatives are computed numerically by central finite differences:

$$
\frac{\partial f}{\partial \theta_j}(x;\hat\theta)
\approx
\frac{f(x;\hat\theta + h e_j) - f(x;\hat\theta - h e_j)}{2h},
$$

with small step size $h$.

### Pointwise confidence interval

Using normal quantile $z_{1-\alpha/2}$:

$$
\hat f(x) \pm z_{1-\alpha/2}\sqrt{g(x)^T\widehat{\Sigma}g(x)}.
$$

The current Curve Builder band computation uses $\alpha=0.05$ by default (95% confidence).

## Prediction Band Mathematics (Gaussian Analytic)

For a new future observation at $x$:

$$
y_{new}(x) = f(x;\hat\theta) + \varepsilon_{new}, \quad \varepsilon_{new}\sim\mathcal{N}(0,\hat\sigma^2).
$$

Approximate predictive variance:

$$
\operatorname{Var}[y_{new}(x)]
\approx
\underbrace{g(x)^T\widehat{\Sigma}g(x)}_{\text{model uncertainty}}
+
\underbrace{\hat\sigma^2}_{\text{observation noise}}.
$$

Prediction interval uses Student-$t$ multiplier with $\nu$ degrees of freedom when supplied:

$$
\hat f(x) \pm t_{1-\alpha/2,\nu}
\sqrt{g(x)^T\widehat{\Sigma}g(x) + \hat\sigma^2}.
$$

If dof is omitted, normal quantiles are used.

## What Is Implemented in the Curve Builder Tab Today

The GUI integration currently computes:

- Confidence band method: Delta Method (order 1).
- Prediction band method: Gaussian analytic prediction.
- Display coverage: default 95% ($\alpha=0.05$).
- Shared plotting on a dense $x$ grid (default 400 points).

Statistics panel reports:

- confidence method label,
- prediction method label,
- residual $\hat\sigma$,
- degrees of freedom.

## GUI Workflow

### Step-by-step

1. Load dataset.
2. Select x and y columns.
3. Optionally add exact-match filters.
4. Confirm plot.
5. Select fit type:
   - polynomial,
   - power law,
   - custom expression.
6. Apply fit.
7. The tab renders:
   - data points,
   - fitted curve,
   - confidence band,
   - prediction band.

### Interpretation in the plot

- Confidence band reflects uncertainty in the mean calibration curve.
- Prediction band reflects uncertainty for a new single observation at each $x$.

For calibration transfer or inverse estimation, confidence width is relevant to mean-function precision, while prediction width is relevant to expected variability of newly measured signals.

## Exported Artifacts and Their Relation to Bands

When exporting a fitted model, the tab writes:

- plot image,
- pickled model object,
- text details.

The exported model stores information needed for confidence and prediction edge evaluation (including covariance, dof, residual scale), enabling downstream uncertainty-aware inversion workflows.

## Programmatic Usage in AnaSens

## A. Direct Curve Builder band service

```python
import numpy as np
from anasens.curve_builder.fitters import FitService
from anasens.curve_builder.band_service import BandService

x = np.asarray([...], dtype=float)
y = np.asarray([...], dtype=float)

fit_service = FitService()
fitting_result = fit_service.fit_polynomial(x, y, order=2)

band_service = BandService()
bands = band_service.compute_gaussian_bands(
    fitting_result=fitting_result,
    x_values=x,
    y_values=y,
    alpha=0.05,
    grid_size=400,
)

conf = bands.confidence_band
pred = bands.prediction_band
```

## B. Using internal uncertainty engines directly

```python
import numpy as np
from anasens.uncertainty.error_models import GaussianError
from anasens.uncertainty.inference.mle import MLE
from anasens.uncertainty.intervals import DeltaMethod
from anasens.uncertainty.prediction import GaussianAnalyticPrediction

# model signature: model(x, theta)
def model(x, theta):
    a, b = theta
    return a * np.asarray(x) + b

x = np.asarray([...], dtype=float)
y = np.asarray([...], dtype=float)

error = GaussianError(sigma=0.2)
mle = MLE(model=model, error_model=error, x=x, y=y)
fit = mle.fit(initial_guess=[1.0, 0.0], method="BFGS")

def grad(x0, theta):
    return np.asarray([x0, 1.0], dtype=float)

x_grid = np.linspace(x.min(), x.max(), 300)

conf_band = DeltaMethod(
    model=model,
    theta_hat=fit.theta_hat,
    cov=fit.covariance,
    grad=grad,
).band(x_grid=x_grid, alpha=0.05)

pred_band = GaussianAnalyticPrediction(
    model=model,
    theta_hat=fit.theta_hat,
    cov=fit.covariance,
    error_model=error,
    grad=grad,
    dof=max(len(x) - len(fit.theta_hat), 1),
).band(x_grid=x_grid, alpha=0.05)
```

## Available Advanced Methods in the Internal Toolkit

Even if not yet wired into the Curve Builder GUI path, the internal package includes:

1. Monte Carlo confidence bands:
   - sample $\theta^* \sim \mathcal{N}(\hat\theta,\widehat\Sigma)$,
   - propagate $f(x;\theta^*)$,
   - use quantiles for interval limits.
2. Profile-likelihood confidence intervals:
   - constrained maximization with $f(x;\theta)=q$,
   - Wilks-threshold acceptance.
3. Monte Carlo prediction for non-Gaussian residuals:
   - sample parameters,
   - sample residuals from selected error model,
   - quantile predictive intervals.

These methods are useful for strongly nonlinear models and non-Gaussian error structures.

## Assumptions and Validity Conditions

The current GUI default bands rely on:

1. Local asymptotic normality of parameter estimator.
2. Adequate model specification (structural fit quality).
3. Numerical stability of covariance estimate.
4. Approximate additive homoscedastic Gaussian residual behavior for analytic prediction path.

When these assumptions are weak, prefer Monte Carlo or profile-likelihood alternatives.

## Numerical and Practical Considerations

1. Gradient step size:
   - finite-difference noise can inflate/deflate propagated variance.
2. Ill-conditioned covariance:
   - singular or near-singular matrices indicate weak parameter identifiability.
3. Extrapolation:
   - bands are most defensible inside calibration domain.
4. Data sufficiency:
   - low dof can produce unstable residual scale and broad intervals.

## Recommended Reporting Language for Publications

When reporting results generated by this workflow, explicitly state:

1. Model family and fitted equation.
2. Estimation method (nonlinear least squares in current tab path, or MLE if used directly).
3. Confidence-band method (Delta, order 1).
4. Prediction-band method (Gaussian analytic, $t$ multiplier, dof reported).
5. Coverage level (for example, 95%).
6. Residual noise assumptions and any diagnostics performed.

## Minimal Checklist Before Trusting Bands

1. Verify residuals are centered and roughly structureless versus $x$.
2. Check fitted parameters are identifiable and covariance finite.
3. Confirm confidence band is narrower than prediction band.
4. Avoid interpreting extrapolated intervals as equally reliable.

## Extension Roadmap for Academic Use

Potential high-value next integrations:

1. GUI selector for confidence method:
   - Delta,
   - Monte Carlo,
   - Profile likelihood.
2. GUI selector for prediction method:
   - Gaussian analytic,
   - Monte Carlo residual simulation.
3. Error-model selector and estimation controls.
4. Residual diagnostics panel (QQ, scale-location, heteroscedastic checks).
5. Joint/simultaneous bands in addition to pointwise intervals.

## Internal Module Map

The integrated functionality is organized in:

- Curve Builder integration layer:
  - band orchestration service,
  - GUI wiring,
  - plot overlays.
- Internal uncertainty toolkit:
  - bands representations,
  - inference engines,
  - interval methods,
  - prediction methods,
  - error models.

This separation allows stable GUI workflows while preserving advanced statistical extensibility for research use.

## Versioning Note

This document reflects the first integrated uncertainty release where Curve Builder uses Delta confidence and Gaussian analytic prediction by default.

As additional methods are exposed in the UI, update this document with method-selection semantics, diagnostics guidance, and computational complexity benchmarks.

---

# Appendix A. Maximum Likelihood Estimation in Full Detail

This appendix derives the estimation problem used by the internal uncertainty toolkit and explains how the fitted parameters and covariance matrix arise.

## A.1 Data model

Assume independent observations $(x_i, y_i)$ for $i=1,\dots,n$ and the additive error model

$$
y_i = f(x_i;\theta) + \varepsilon_i,
$$

where the residuals are independent and identically distributed according to a density $p_\varepsilon$.


Define the residual at observation $i$ as


$$
r_i(\theta) = y_i - f(x_i;\theta).
$$

The likelihood is then


$$
L(\theta)
=
\prod_{i=1}^n p_\varepsilon\big(r_i(\theta)\mid x_i\big).
$$

Taking logarithms gives the log-likelihood


$$
\ell(\theta)
=
\log L(\theta)
=
\sum_{i=1}^n \log p_\varepsilon\big(r_i(\theta)\mid x_i\big).
$$

The maximum likelihood estimator is


$$
\hat\theta = \arg\max_{\theta} \ell(\theta).
$$

The implementation in [`src/anasens/uncertainty/inference/mle.py`](src/anasens/uncertainty/inference/mle.py) minimizes the negative log-likelihood instead:


$$
\hat\theta = \arg\min_{\theta} -\ell(\theta).
$$

## A.2 Gaussian errors and least squares

If the residual model is Gaussian,

$$
\varepsilon_i \sim \mathcal{N}(0,\sigma^2),
$$

then


$$
p_\varepsilon(r_i)
=
\frac{1}{\sqrt{2\pi}\sigma}
\exp\left(-\frac{r_i^2}{2\sigma^2}\right).
$$

Therefore


$$
\ell(\theta)
=
\sum_{i=1}^n
\left[
-\frac{1}{2}\log(2\pi)
-\log\sigma
-\frac{r_i(\theta)^2}{2\sigma^2}
\right].
$$

Collect the terms that do not depend on $\theta$ into a constant $C$:


$$
\ell(\theta)
=
C - \frac{1}{2\sigma^2}\sum_{i=1}^n r_i(\theta)^2.
$$

Since $C$ and $\sigma$ are fixed, maximizing $\ell(\theta)$ is equivalent to minimizing the sum of squared residuals:


$$
\hat\theta
=
\arg\min_{\theta} \sum_{i=1}^n \big(y_i - f(x_i;\theta)\big)^2.
$$

This is why the current Curve Builder fitting flow, which uses nonlinear least squares, is compatible with the Gaussian uncertainty machinery.

## A.3 Score function

Differentiate the log-likelihood with respect to $\theta$:


$$
\nabla_\theta \ell(\theta)
=
\sum_{i=1}^n
\nabla_\theta \log p_\varepsilon\big(r_i(\theta)\mid x_i\big).
$$

By the chain rule,


$$
\nabla_\theta \log p_\varepsilon(r_i(\theta)\mid x_i)
=
\frac{\partial}{\partial r}
\log p_\varepsilon(r\mid x_i)\Big|_{r=r_i(\theta)}
\nabla_\theta r_i(\theta).
$$

Because


$$
r_i(\theta) = y_i - f(x_i;\theta),
$$

we have


$$
\nabla_\theta r_i(\theta) = -\nabla_\theta f(x_i;\theta).
$$

Thus


$$
\nabla_\theta \ell(\theta)
=
-\sum_{i=1}^n
\left[
\frac{\partial}{\partial r}
\log p_\varepsilon(r\mid x_i)\Big|_{r=r_i(\theta)}
\right]
\nabla_\theta f(x_i;\theta).
$$

At the maximum likelihood estimate, the score vanishes under standard regularity conditions:


$$
\nabla_\theta \ell(\hat\theta) = 0.
$$

## A.4 Observed Fisher information and covariance

The Hessian of the negative log-likelihood at the MLE is the observed information matrix:


$$
\widehat{I}(\hat\theta)
=
\nabla_\theta^2\big[-\ell(\theta)\big]\Big|_{\theta=\hat\theta}.
$$

Under standard asymptotic theory,


$$
\operatorname{Cov}(\hat\theta)
\approx
\widehat{I}(\hat\theta)^{-1}.
$$

This approximation is the basis for the internal covariance estimation helper and for downstream Delta Method and Monte Carlo propagation.

If an optimizer directly supplies an inverse-Hessian approximation, the internal toolkit uses that quantity as a covariance estimate when numerically valid.

---

# Appendix B. Delta Method Derivation Without Skipping Steps

The Delta Method is the approximation that turns parameter uncertainty into uncertainty in the fitted curve.

## B.1 Scalar output at one x

Fix an input value $x$ and define


$$
q(\theta) = f(x;\theta).
$$

We want the uncertainty in $q(\hat\theta)$ induced by uncertainty in $\hat\theta$ itself.

Let

$$
\delta = \hat\theta - \theta_0,
$$

where $\theta_0$ is the true parameter value. Under asymptotic normality,


$$
\delta \approx \mathcal{N}(0,\Sigma).
$$

## B.2 Taylor expansion

Expand $q(\hat\theta)$ around $\theta_0$:


$$
q(\hat\theta)
=
q(\theta_0 + \delta).
$$

First-order Taylor expansion gives


$$
q(\theta_0 + \delta)
\approx
q(\theta_0) + \nabla q(\theta_0)^T\delta.
$$

If we denote the gradient by


$$
g = \nabla q(\theta_0),
$$

then


$$
q(\hat\theta) \approx q(\theta_0) + g^T\delta.
$$

## B.3 Mean of the approximation

Taking expectations,


$$
\mathbb{E}[q(\hat\theta)]
\approx
\mathbb{E}[q(\theta_0) + g^T\delta]
=
q(\theta_0) + g^T\mathbb{E}[\delta].
$$

Because $\mathbb{E}[\delta]=0$,


$$
\mathbb{E}[q(\hat\theta)] \approx q(\theta_0).
$$

## B.4 Variance of the approximation

Now compute the variance:


$$
\operatorname{Var}[q(\hat\theta)]
\approx
\operatorname{Var}[g^T\delta].
$$

For a constant vector $g$ and random vector $\delta$ with covariance $\Sigma$,


$$
\operatorname{Var}[g^T\delta]
=
g^T\Sigma g.
$$

Hence


$$
\operatorname{Var}[f(x;\hat\theta)]
\approx
g(x)^T\Sigma g(x).
$$

Replacing the unknown covariance $\Sigma$ with the estimated covariance $\widehat\Sigma$ yields the implemented formula.

## B.5 Confidence interval at one x

Approximating the distribution of $q(\hat\theta)$ as normal,


$$
q(\hat\theta) \approx \mathcal{N}\left(q(\hat\theta),\, g^T\widehat\Sigma g\right),
$$

we obtain the pointwise interval


$$
f(x;\hat\theta) \pm z_{1-\alpha/2}\sqrt{g(x)^T\widehat\Sigma g(x)}.
$$

The current internal implementation uses this form in the Delta Method engine.

## B.6 Vector-valued band over an x-grid

Let $x_1,\dots,x_m$ be a grid. For each $x_j$ compute


$$
\hat f_j = f(x_j;\hat\theta),
$$

and


$$
\widehat{\operatorname{Var}}(\hat f_j) = g(x_j)^T\widehat\Sigma g(x_j).
$$

Then the band arrays are


$$
	ext{lower}_j = \hat f_j - z_{1-\alpha/2}\sqrt{\widehat{\operatorname{Var}}(\hat f_j)},
$$

$$
	ext{upper}_j = \hat f_j + z_{1-\alpha/2}\sqrt{\widehat{\operatorname{Var}}(\hat f_j)}.
$$

This is exactly the object represented by `ConfidenceBand`.

## B.7 Second-order Delta Method

The internal toolkit also exposes a second-order option. The derivation begins with a second-order Taylor expansion:


$$
q(\theta_0+\delta)
\approx
q(\theta_0) + g^T\delta + \frac{1}{2}\delta^T H \delta,
$$

where $H$ is the Hessian of $q$ with respect to $\theta$.

The variance then includes additional terms involving contractions of $H$ and higher derivatives. These terms are implemented in the second-order path of the Delta Method engine. They are more expensive and are only recommended when the model is sufficiently smooth and the first-order approximation is visibly inadequate.

---

# Appendix C. Prediction Band Derivation

Prediction bands differ from confidence bands because they include the random noise in a future measurement.

## C.1 Future observation model

For a new observation at input $x$,

$$
y_{new}(x) = f(x;\hat\theta) + \eta(x),
$$

where $\eta(x)$ is the future residual.

If the residual model is Gaussian,

$$
\eta(x) \sim \mathcal{N}(0,\hat\sigma^2).
$$

## C.2 Conditional variance decomposition

Decompose the predictive uncertainty into two parts:

$$
\operatorname{Var}[y_{new}(x)]
=
\operatorname{Var}[f(x;\hat\theta)] + \operatorname{Var}[\eta(x)]
$$

under the working assumption that the model uncertainty and observation noise are independent.

Substituting the Delta Method approximation gives


$$
\operatorname{Var}[y_{new}(x)]
\approx
g(x)^T\widehat\Sigma g(x) + \hat\sigma^2.
$$

## C.3 Normal approximation and interval

If the predictive distribution is approximated by a normal law centered at the fitted location,


$$
y_{new}(x)
\approx
\mathcal{N}\left(f(x;\hat\theta),\, g(x)^T\widehat\Sigma g(x) + \hat\sigma^2\right),
$$

then the pointwise interval is


$$
f(x;\hat\theta)
\pm
c_{1-\alpha/2}
\sqrt{g(x)^T\widehat\Sigma g(x) + \hat\sigma^2}.
$$

In the implemented Gaussian analytic path, the critical value $c_{1-\alpha/2}$ is either:

1. $z_{1-\alpha/2}$ if no degrees of freedom are supplied, or
2. $t_{1-\alpha/2,\nu}$ if degrees of freedom are provided.

## C.4 Why prediction bands are wider

Since

$$
g(x)^T\widehat\Sigma g(x) + \hat\sigma^2
\ge
g(x)^T\widehat\Sigma g(x),
$$

and the critical value is nonnegative, the prediction band is at least as wide as the confidence band at each x.

This is not a plotting artifact. It is a direct consequence of including a second variance term.

## C.5 Monte Carlo prediction band derivation

For arbitrary additive residual distributions, the predictive distribution is simulated rather than approximated analytically.

The simulation sequence is:

1. Draw parameter samples

$$
	heta^{(s)} \sim \mathcal{N}(\hat\theta,\widehat\Sigma), \quad s=1,\dots,S.
$$

2. Compute fitted means

$$
\mu^{(s)} = f(x;\theta^{(s)}).
$$

3. Draw residuals

$$
\varepsilon^{(s)} \sim p_\varepsilon.
$$

4. Form predictive draws

$$
y^{(s)}_{new} = \mu^{(s)} + \varepsilon^{(s)}.
$$

5. Use empirical quantiles:

$$
	ext{lower}(x) = Q_{\alpha/2}\left(\{y^{(s)}_{new}\}_{s=1}^S\right),
$$

$$
	ext{upper}(x) = Q_{1-\alpha/2}\left(\{y^{(s)}_{new}\}_{s=1}^S\right).
$$

This simulation targets the full predictive distribution and therefore supports non-Gaussian residual models.

---

# Appendix D. Monte Carlo Confidence Bands

The internal toolkit also provides Monte Carlo confidence bands for the fitted mean.

## D.1 Sampling model

Approximate the parameter estimator distribution by

$$
	heta^* \sim \mathcal{N}(\hat\theta,\widehat\Sigma).
$$

For each draw,

$$
q^{(s)}(x) = f(x;\theta^{(s)}).
$$

## D.2 Empirical interval construction

The confidence band is built from empirical quantiles of the sampled values:

$$
	ext{lower}(x) = Q_{\alpha/2}\left(\{q^{(s)}(x)\}_{s=1}^S\right),
$$

$$
	ext{upper}(x) = Q_{1-\alpha/2}\left(\{q^{(s)}(x)\}_{s=1}^S\right).
$$

The band does not add observation noise, so it remains a confidence band rather than a prediction band.

## D.3 When Monte Carlo is useful

Monte Carlo bands are useful when:

1. The model is highly nonlinear.
2. The Delta approximation is visibly asymmetric or unstable.
3. A derivative-free uncertainty estimate is preferred.

---

# Appendix E. Profile Likelihood Confidence Intervals

Profile likelihood uses the likelihood surface directly.

## E.1 Definition

For a fixed x and candidate curve value q, define the constrained optimization problem

$$
\ell_p(q; x) = \max_{\theta} \ell(\theta)
$$

subject to

$$
f(x;\theta) = q.
$$

## E.2 Likelihood-ratio test statistic

Wilks' theorem motivates the statistic

$$
\Lambda(q; x) = 2\left[\ell(\hat\theta) - \ell_p(q; x)\right].
$$

For a one-dimensional parameter of interest, the asymptotic cutoff is

$$
\Lambda(q; x) \le \chi^2_{1,1-\alpha}.
$$

Equivalently,

$$
\ell_p(q; x) \ge \ell(\hat\theta) - \frac{1}{2}\chi^2_{1,1-\alpha}.
$$

The internal implementation scans a user-supplied $q$ grid, accepts values meeting the threshold, and returns the minimum and maximum accepted values.

## E.3 Why profile likelihood is not used for prediction intervals

Prediction intervals concern new random observations, not the fitted mean itself. Profile likelihood is naturally a statement about the parameter-constrained mean response and therefore belongs to confidence-band construction.

---

# Appendix F. Practical Interpretation Notes

## F.1 Band ordering on the plot

The curve-builder renderer draws elements in this order:

1. prediction band,
2. confidence band,
3. fitted curve,
4. data points.

This is deliberate. The wider prediction band sits behind the narrower confidence band, which sits behind the fitted curve, while the data remain visible on top.

## F.2 Visual meaning

If the prediction band is much wider than the confidence band, most of the width is due to observation noise rather than parameter uncertainty.

If both are narrow, the model is precise and the residual variance is low.

If the confidence band is narrow but prediction band is wide, the mean curve is well identified but future measurements remain noisy.

## F.3 Recommended reporting statement

When writing this up in a manuscript, a suitable description is:

"Parameters were estimated by nonlinear fitting. Pointwise 95% confidence bands for the fitted mean were obtained by first-order Delta Method propagation of the estimated parameter covariance. Pointwise 95% prediction bands for future observations were obtained by combining parameter uncertainty with estimated residual variance under an additive Gaussian error model."

## F.4 Future extension note

If you later enable Monte Carlo or profile-likelihood options in the GUI, document them separately and specify whether the reported interval is:

1. pointwise or simultaneous,
2. confidence or prediction,
3. analytical or simulation-based,
4. model-based or data-driven.

Those distinctions are statistically material and should not be collapsed in technical writing.
