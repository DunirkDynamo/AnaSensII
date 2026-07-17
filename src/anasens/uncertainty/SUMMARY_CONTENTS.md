# Confidence Bands and Prediction Intervals

This package separates two related but different uncertainty problems:

1. **Confidence bands for the fitted model**
2. **Prediction intervals for future observations**

These should not be treated as the same object.

---

# 1. Confidence Bands for the Fitted Model

A confidence band quantifies uncertainty in the fitted model mean:

```math
f(x; \hat{\theta})
```

It describes uncertainty in the estimated curve itself.

It does **not** include future measurement noise.

The package supports three methods for model confidence bands:

* Delta Method
* Monte Carlo parameter propagation
* Profile Likelihood

---

## 1.1 Delta Method

The Delta Method propagates uncertainty from the fitted parameters into the model output.

```math
\operatorname{Var}[f(x;\hat{\theta})]
\approx
J(x)\operatorname{Cov}(\hat{\theta})J(x)^T
```

where `J(x)` is the gradient of the model with respect to the parameters.

### Pros

* Very fast
* Analytical
* Good for mildly nonlinear models
* Works well when derivatives are available
* Supports higher-order corrections if Hessian and third derivatives are supplied

### Cons

* Local approximation
* Can fail for strongly nonlinear models
* Requires analytical derivatives
* Usually gives symmetric intervals

---

## 1.2 Monte Carlo Confidence Bands

Monte Carlo confidence bands sample model parameters from the asymptotic joint Normal distribution:

```math
\theta^* \sim N(\hat{\theta}, \operatorname{Cov}(\hat{\theta}))
```

Then each sampled parameter vector is propagated through the model:

```math
f(x;\theta^*)
```

Equal-tail quantiles of these values give the confidence limits.

### Pros

* Handles nonlinear models better than first-order Delta Method
* Does not require derivatives
* Can produce asymmetric intervals
* Conceptually simple

### Cons

* Slower than Delta Method
* Depends on Monte Carlo sample size
* Still relies on asymptotic Normality of the fitted parameters

---

## 1.3 Profile Likelihood Confidence Bands

Profile likelihood uses the likelihood surface directly.

For a fixed `x`, candidate model values `q` are tested by maximizing the likelihood subject to:

```math
f(x;\theta)=q
```

Wilks' theorem is then used to determine which `q` values are allowed.

### Pros

* Uses likelihood geometry directly
* Good for nonlinear models
* Can produce asymmetric intervals
* Does not require parameter covariance propagation

### Cons

* Computationally expensive
* Requires the log-likelihood
* Requires constrained optimization
* More complex numerically
* Not used for prediction intervals in this package

---

# 2. Prediction Intervals

A prediction interval quantifies uncertainty in a future observation:

```math
y_{\text{new}}(x)
```

It includes both:

1. model uncertainty
2. observation noise

The package supports two prediction interval methods:

* Gaussian analytic prediction
* Monte Carlo prediction

Profile likelihood is intentionally not used for prediction intervals.

---

## 2.1 Gaussian Analytic Prediction

This method is used only when the error model is Gaussian:

```math
\epsilon \sim N(0,\sigma^2)
```

The total prediction variance is:

```math
\operatorname{Var}[y_{\text{new}}(x)]
=
\operatorname{Var}[f(x;\hat{\theta})]
+
\sigma^2
```

The model variance is computed using the Delta Method.

The interval is then:

```math
f(x;\hat{\theta})
\pm
c
\sqrt{
\operatorname{Var}[f(x;\hat{\theta})]
+
\sigma^2
}
```

where `c` is a Normal or Student-t critical value.

### Pros

* Fast
* Analytical
* Exact under Gaussian assumptions and linear propagation
* No Monte Carlo required

### Cons

* Valid only for Gaussian additive noise
* Depends on model variance from Delta Method
* Not appropriate for non-Gaussian error models

---

## 2.2 Monte Carlo Prediction

Monte Carlo prediction works for any additive error model.

For each `x`:

1. sample parameters:

```math
\theta^* \sim N(\hat{\theta}, \operatorname{Cov}(\hat{\theta}))
```

2. compute the model location:

```math
\mu^*=f(x;\theta^*)
```

3. sample observation noise:

```math
\epsilon^* \sim p_\epsilon(\epsilon \mid x)
```

4. form a simulated future observation:

```math
y^*=\mu^*+\epsilon^*
```

5. take equal-tail quantiles of the simulated `y*` values.

### Pros

* Works for arbitrary additive error models
* Handles non-Gaussian prediction intervals
* Can produce asymmetric prediction intervals
* Directly simulates the predictive distribution

### Cons

* Slower than the Gaussian analytic method
* Requires many samples for stable tails
* Still depends on asymptotic Normality of parameter estimates

---

# 3. Method Summary

## Confidence Bands for the Model

| Method             | Uses Observation Noise? | Requires Derivatives? | Requires Log-Likelihood? | Handles Nonlinearity |  Speed |
| ------------------ | ----------------------: | --------------------: | -----------------------: | -------------------: | -----: |
| Delta Method       |                      No |                   Yes |                       No |             Moderate |   Fast |
| Monte Carlo        |                      No |                    No |                       No |                 Good | Medium |
| Profile Likelihood |                      No |                    No |                      Yes |                 Good |   Slow |

---

## Prediction Intervals

| Method            | Error Models             | Includes Observation Noise? | Uses Profile Likelihood? |       Speed |
| ----------------- | ------------------------ | --------------------------: | -----------------------: | ----------: |
| Gaussian Analytic | Gaussian only            |                         Yes |                       No |        Fast |
| Monte Carlo       | Any additive error model |                         Yes |                       No | Medium/Slow |

---

# 4. Design Rule

The package follows this rule:

```text
Confidence bands quantify uncertainty in the fitted model mean.
Prediction bands quantify uncertainty in future observations.
```

Therefore:

```text
Confidence bands:
    Delta Method
    Monte Carlo
    Profile Likelihood

Prediction bands:
    Gaussian analytic Delta Method, only for Gaussian errors
    Monte Carlo, for all additive error models
```

Profile likelihood is not used for prediction intervals because it quantifies uncertainty in the fitted model mean, not future observation noise.
