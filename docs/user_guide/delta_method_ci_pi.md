# Delta-Method Confidence and Prediction Intervals

This note explains how the Curve Builder computes confidence intervals (CI) and prediction intervals (PI) when the delta method is selected.

The goal is to be explicit about what is computed, which quantities are used, and what is not being approximated by Monte Carlo in the delta-method path.

## 1. What the delta method is used for

The fitted curve is written as

$$
f(x; \hat\theta)
$$

where $\hat\theta$ is the fitted parameter vector and $\operatorname{Cov}(\hat\theta)$ is the estimated parameter covariance matrix.

The delta method propagates uncertainty in $\hat\theta$ through the model by linearizing the model around the fitted parameters.

For a single input $x$, let

$$
g(x) = \nabla_\theta f(x; \hat\theta)
$$

be the gradient of the model with respect to the parameters.

Then the first-order delta-method variance of the fitted mean is

$$
\operatorname{Var}\big[f(x; \hat\theta)\big] \approx g(x)^T\,\operatorname{Cov}(\hat\theta)\,g(x).
$$

This is the core quantity used for the confidence band.

## 2. Confidence interval for the fitted mean

The confidence interval describes uncertainty in the fitted mean curve only:

$$
f(x; \hat\theta)
$$

It does not include observation noise.

Using the delta-method standard error

$$
\operatorname{SE}_{\mathrm{CI}}(x) = \sqrt{g(x)^T\,\operatorname{Cov}(\hat\theta)\,g(x)},
$$

the implementation uses a normal or $t$ critical value depending on the internal band machinery and degrees of freedom available. The band half-width is

$$
h_{\mathrm{CI}}(x) = c_{\alpha}\,\operatorname{SE}_{\mathrm{CI}}(x).
$$

So the interval is

$$
\Big[f(x; \hat\theta) - h_{\mathrm{CI}}(x),\; f(x; \hat\theta) + h_{\mathrm{CI}}(x)\Big].
$$

### Interpretation

- Wider where the fitted model is more sensitive to parameter changes.
- Narrower where the gradient is small or the parameter covariance is small.
- This band reflects uncertainty in the mean response, not future measurement noise.

## Delta-Extended

The higher-order delta method extends the first-order linear approximation by
including curvature information from the model itself.

In Curve Builder, this corresponds to the delta method with order 2 for the
confidence band. The implementation uses:

- the gradient $g(x)$,
- the Hessian of $f(x;\theta)$ with respect to $\theta$,
- and a third-derivative tensor.

If we write the parameter error as

$$
\delta = \hat\theta - \theta,
$$

then the second-order Taylor expansion around $\hat\theta$ is

$$
f(x; \theta) \approx f(x; \hat\theta)
+ g(x)^T(\theta - \hat\theta)
+ \frac{1}{2}(\theta - \hat\theta)^T H(x)(\theta - \hat\theta),
$$

where $H(x)$ is the Hessian of $f(x;\theta)$ with respect to $\theta$.

Using that expansion, the second-order delta-method variance is approximated by

$$
\operatorname{Var}[f(x; \hat\theta)]
\approx
g(x)^T\,\Sigma\,g(x)
+ \text{higher-order corrections},
$$

where $\Sigma = \operatorname{Cov}(\hat\theta)$.

In the Curve Builder implementation, those higher-order corrections are computed
from the Hessian and the third-derivative tensor, giving the order-2 variance
approximation

$$
\operatorname{Var}_{\mathrm{ext}}[f(x; \hat\theta)]
=
g(x)^T\,\Sigma\,g(x)
+ C_{H}(x)
+ C_{T}(x),
$$

where $C_{H}(x)$ is the Hessian-based correction and $C_{T}(x)$ is the
third-derivative correction.

The first-order term is still the same variance contribution used above:

$$
g(x)^T\,\operatorname{Cov}(\hat\theta)\,g(x).
$$

The higher-order method adds correction terms that account for local curvature
in the fitted surface. Conceptually, this matters when the model is not close to
linear in the parameters over the uncertainty region.

The practical effect is:

- the band can widen or shift relative to first-order delta,
- the approximation is better when the curvature of $f(x;\theta)$ is important,
- and the extra terms use deterministic derivatives, not sampling.

The corresponding higher-order confidence interval uses the same critical value
construction as the first-order version, but with the extended standard error

$$
\operatorname{SE}_{\mathrm{ext}}(x) = \sqrt{\operatorname{Var}_{\mathrm{ext}}[f(x; \hat\theta)]}.
$$

So its band half-width is

$$
h_{\mathrm{ext}}(x) = c_{\alpha}\,\operatorname{SE}_{\mathrm{ext}}(x).
$$

The higher-order delta method is therefore still analytical. It is not Monte
Carlo. The difference is that it uses more of the local Taylor expansion than
the first-order version.

For the Curve Builder prediction interval, the supported delta path remains the
first-order analytic propagation of model variance plus observation noise.
The higher-order extension is used for confidence bands, not as a separate
prediction-band algorithm.

## 3. Prediction interval for a new observation

The prediction interval describes uncertainty for a future observation:

$$
y_{\mathrm{new}}(x) = f(x; \hat\theta) + \varepsilon.
$$

For the Curve Builder delta path, the Gaussian error model is the supported analytic prediction case. The error term contributes an additional variance term

$$
\sigma^2.
$$

So the total delta-method prediction variance is

$$
\operatorname{Var}\big[y_{\mathrm{new}}(x)\big] \approx g(x)^T\,\operatorname{Cov}(\hat\theta)\,g(x) + \sigma^2.
$$

The corresponding standard error is

$$
\operatorname{SE}_{\mathrm{PI}}(x) = \sqrt{g(x)^T\,\operatorname{Cov}(\hat\theta)\,g(x) + \sigma^2}.
$$

The prediction interval half-width is

$$
h_{\mathrm{PI}}(x) = c_{\alpha}\,\operatorname{SE}_{\mathrm{PI}}(x).
$$

So the interval is

$$
\Big[f(x; \hat\theta) - h_{\mathrm{PI}}(x),\; f(x; \hat\theta) + h_{\mathrm{PI}}(x)\Big].
$$

### Interpretation

- This band is always at least as wide as the confidence band when $\sigma^2 > 0$.
- It adds observation noise on top of parameter uncertainty.
- In Curve Builder, this is the band used when the delta prediction interval is requested.

## 4. What is not happening

The delta-method prediction interval is not Monte Carlo sampling.

It is not drawing repeated synthetic observations to estimate quantiles.

Instead, it computes the variance analytically from the gradient and covariance, then applies a critical value.

That is why the delta-method prediction band should appear smooth when evaluated on a sufficiently dense grid.

## 5. Practical summary

For each $x$:

1. Evaluate the fitted mean $f(x;\hat\theta)$.
2. Compute the gradient $g(x)$ with respect to the parameters.
3. Propagate parameter uncertainty with $g(x)^T\,\operatorname{Cov}(\hat\theta)\,g(x)$.
4. For the prediction interval, add $\sigma^2$ for observation noise.
5. Multiply the resulting standard error by the interval critical value.

## 6. Source locations

The relevant implementation lives in:

- [src/anasens/curve_builder/band_service.py](../../src/anasens/curve_builder/band_service.py)
- [src/anasens/uncertainty/intervals/delta.py](../../src/anasens/uncertainty/intervals/delta.py)
- [src/anasens/uncertainty/prediction/gaussian_analytic.py](../../src/anasens/uncertainty/prediction/gaussian_analytic.py)

The plot renderer then displays the resulting bands without changing the underlying statistics.