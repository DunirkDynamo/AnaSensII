"""Specialized polynomial prediction interval methods.

These methods are adapted from the existing PI mini-package implementation
without changing their mathematical formulations.
"""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from scipy.optimize import brentq, minimize
from scipy.stats import chi2, t


class GeneralizedInverseProfileCalibration:
    """Generalized inverse profile-likelihood calibration for polynomials."""

    def __init__(self, x_cal, y_cal, degree=1):
        self.x_cal = np.asarray(x_cal, dtype=float)
        self.y_cal = np.asarray(y_cal, dtype=float)
        self.n = len(x_cal)
        self.degree = int(degree)

        # Pre-fit polynomial OLS to get stable initialization vectors.
        self.poly_init = np.polyfit(self.x_cal, self.y_cal, self.degree)
        # Reverse to ascending coefficients [beta_0, beta_1, ...].
        self.beta_init = self.poly_init[::-1]

    def _evaluate_poly(self, x, beta):
        """Evaluate an ascending polynomial sum(beta_j * x^j)."""
        x_numeric = np.asarray(x, dtype=float)
        return sum(b * (x_numeric**j) for j, b in enumerate(beta))

    def _joint_neg_log_lik(self, params, x_0, y_0):
        # params layout: [beta_0, beta_1, ..., beta_d, sigma_sq]
        beta = params[:-1]
        sigma_sq = params[-1]

        if sigma_sq <= 0:
            return np.inf

        cal_resid = self.y_cal - self._evaluate_poly(self.x_cal, beta)
        pred_resid = y_0 - self._evaluate_poly(x_0, beta)

        rss = np.sum(cal_resid**2) + pred_resid**2
        total_n = self.n + 1

        return (total_n / 2.0) * np.log(2 * np.pi * sigma_sq) + (rss / (2.0 * sigma_sq))

    def _profile_log_lik(self, x_0, y_0):
        init_sigma_sq = np.var(self.y_cal - self._evaluate_poly(self.x_cal, self.beta_init))
        init_params = list(self.beta_init) + [init_sigma_sq]

        res = minimize(self._joint_neg_log_lik, init_params, args=(x_0, y_0), method="Nelder-Mead")
        if not res.success:
            raise ValueError(f"Profile likelihood optimization failed: {res.message}")
        return -res.fun

    def estimate(self, y_0, confidence_level=0.95):
        # Global MLE optimization (floating both parameters and x_0).
        obj = lambda p: self._joint_neg_log_lik(p[:-1], p[-1], y_0)

        init_sigma_sq = np.var(self.y_cal - self._evaluate_poly(self.x_cal, self.beta_init))

        # Approximate x_0 by roots of initial OLS fit minus y_0.
        roots = np.roots(self.poly_init - np.array([0] * self.degree + [y_0]))
        real_roots = roots[np.isreal(roots)].real
        if len(real_roots) > 0:
            x_0_init = real_roots[np.argmin(np.abs(real_roots - np.mean(self.x_cal)))]
        else:
            x_0_init = np.mean(self.x_cal)

        init_all = list(self.beta_init) + [init_sigma_sq, x_0_init]

        global_res = minimize(obj, init_all, method="Nelder-Mead")
        if not global_res.success:
            raise ValueError(f"Global profile optimization failed: {global_res.message}")

        x_0_mle = global_res.x[-1]
        log_l_max = -global_res.fun

        crit_val = chi2.ppf(confidence_level, df=1) / 2.0
        target_f = lambda x: (log_l_max - self._profile_log_lik(x, y_0)) - crit_val

        # Dynamically determine step scales based on OLS error approximation.
        beta_1_approx = self.beta_init[1] if len(self.beta_init) > 1 else 1.0
        se_approx = np.sqrt(init_sigma_sq) / (abs(beta_1_approx) + 1e-6)

        lower_bound = brentq(target_f, x_0_mle - 20 * se_approx, x_0_mle)
        upper_bound = brentq(target_f, x_0_mle, x_0_mle + 20 * se_approx)

        return {
            "mle_concentration": float(x_0_mle),
            "ci_lower": float(lower_bound),
            "ci_upper": float(upper_bound),
        }


def fit_poly_quantile_regression(x, y, degree, q, method="interior"):
    """Fit an arbitrary-degree polynomial quantile model using statsmodels."""
    x_numeric = np.asarray(x, dtype=float)
    y_numeric = np.asarray(y, dtype=float)
    design = np.vander(x_numeric, int(degree) + 1, increasing=True)
    model = sm.QuantReg(y_numeric, design)
    result = model.fit(q=float(q), method=method)
    return np.asarray(result.params, dtype=float)


def evaluate_poly_fit(x, beta):
    """Evaluate ascending polynomial coefficients on x."""
    x_numeric = np.asarray(x, dtype=float)
    return sum(b * (x_numeric**j) for j, b in enumerate(beta))


def exact_polynomial_gaussian_prediction_interval(x_cal, y_cal, x_grid, degree, confidence=0.95):
    """Exact analytical OLS polynomial PI with Gaussian errors."""
    x_cal = np.asarray(x_cal, dtype=float)
    y_cal = np.asarray(y_cal, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    degree = int(degree)

    n = len(x_cal)
    p = degree + 1

    design = np.vander(x_cal, p, increasing=True)
    beta_ols = np.linalg.lstsq(design, y_cal, rcond=None)[0]

    y_fit = evaluate_poly_fit(x_grid, beta_ols)
    residuals = y_cal - evaluate_poly_fit(x_cal, beta_ols)
    s_err = np.sqrt(np.sum(residuals**2) / max(n - p, 1))

    grid_design = np.vander(x_grid, p, increasing=True)
    inv_xtx = np.linalg.pinv(design.T @ design)
    leverage = np.sum((grid_design @ inv_xtx) * grid_design, axis=1)

    alpha = 1.0 - float(confidence)
    t_val = t.ppf(1.0 - alpha / 2.0, df=max(n - p, 1))
    pi_half_width = t_val * s_err * np.sqrt(1.0 + leverage)

    return y_fit, y_fit - pi_half_width, y_fit + pi_half_width


def quantile_polynomial_prediction_interval(x_cal, y_cal, x_grid, degree, confidence=0.95, method="interior"):
    """Polynomial quantile-regression PI."""
    alpha = 1.0 - float(confidence)
    lower_q = alpha / 2.0
    upper_q = 1.0 - alpha / 2.0

    beta_lower = fit_poly_quantile_regression(x_cal, y_cal, degree, lower_q, method=method)
    beta_median = fit_poly_quantile_regression(x_cal, y_cal, degree, 0.50, method=method)
    beta_upper = fit_poly_quantile_regression(x_cal, y_cal, degree, upper_q, method=method)

    y_lower = evaluate_poly_fit(x_grid, beta_lower)
    y_median = evaluate_poly_fit(x_grid, beta_median)
    y_upper = evaluate_poly_fit(x_grid, beta_upper)

    return y_median, y_lower, y_upper


def profile_likelihood_modified_map(x_cal, y_cal, degree, confidence=0.95, n_y_points=35):
    """Map profile-likelihood inverse-calibration intersections in (x, y)."""
    x_cal = np.asarray(x_cal, dtype=float)
    y_cal = np.asarray(y_cal, dtype=float)

    y_values_to_test = np.linspace(float(np.min(y_cal)), float(np.max(y_cal)), int(n_y_points))
    engine = GeneralizedInverseProfileCalibration(x_cal, y_cal, degree=degree)

    x_mle = []
    x_lower = []
    x_upper = []
    y_valid = []

    for y_val in y_values_to_test:
        try:
            result = engine.estimate(y_val, confidence_level=confidence)
            x_mle.append(result["mle_concentration"])
            x_lower.append(result["ci_lower"])
            x_upper.append(result["ci_upper"])
            y_valid.append(float(y_val))
        except ValueError:
            continue

    if not y_valid:
        raise ValueError("Profile-likelihood modified mapping did not converge for any y-grid points.")

    return {
        "y_grid": np.asarray(y_valid, dtype=float),
        "x_mle": np.asarray(x_mle, dtype=float),
        "x_lower": np.asarray(x_lower, dtype=float),
        "x_upper": np.asarray(x_upper, dtype=float),
    }
