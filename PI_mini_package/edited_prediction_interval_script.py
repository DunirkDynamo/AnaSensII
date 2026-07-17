import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brentq
from scipy.stats import chi2, t, norm

# =====================================================================
# 1. CORE UTILITIES & HELPER FUNCTIONS
# =====================================================================

def design_matrix(x: np.ndarray, degree: int) -> np.ndarray:
    """Generates a Vandermonde design matrix for polynomial regression.

    This utility constructs a design matrix where each row represents a data
    point raised to progressive powers: [1, x, x^2, ..., x^d].

    Args:
        x (np.ndarray): 
            A 1D array of independent variable observations of shape (n_samples,).
        degree (int): 
            The degree of the polynomial model. Must be a non-negative integer.

    Returns:
        np.ndarray: 
            The Vandermonde design matrix of shape (n_samples, degree + 1).
    """
    return np.vander(np.asarray(x), degree + 1, increasing=True)


def evaluate_poly(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Evaluates a polynomial function given inputs and coefficient weights.

    Computes the matrix multiplication of the polynomial design matrix and 
    the parameter coefficients vector.

    Args:
        x (np.ndarray): 
            A 1D array of coordinate points of shape (n_points,).
        beta (np.ndarray): 
            A 1D array of polynomial coefficients of shape (degree + 1,), 
            ordered from lowest power (intercept) to highest power.

    Returns:
        np.ndarray: 
            The evaluated polynomial responses of shape (n_points,).
    """
    X = design_matrix(x, len(beta) - 1)
    return X @ beta


# =====================================================================
# 2. FORWARD PREDICTION INTERVAL MODELS (Predicting Y given X)
# =====================================================================

class PolynomialOLS:
    """Ordinary Least Squares (OLS) Polynomial Regression with analytical PIs.

    Attributes:
        degree (int): 
            The polynomial degree chosen for the model.
        beta (np.ndarray or None): 
            Fitted polynomial coefficients of shape (degree + 1,).
        s_err (float or None): 
            Residual standard error of the fitted model (residual variance square root).
        inv_xtx (np.ndarray or None): 
            The inverse of the covariance design matrix (X^T * X)^-1.
        X_design (np.ndarray or None): 
            The training Vandermonde design matrix.
        y_train (np.ndarray or None): 
            The training dependent variable vector.
        n (int): 
            Number of training samples.
        p (int): 
            Number of model parameters (degree + 1).
    """

    def __init__(self, degree: int = 1):
        """Initializes the PolynomialOLS estimator.

        Args:
            degree (int, optional): 
                Degree of the polynomial. Defaults to 1 (linear).
        """
        self.degree: int = degree
        self.beta: np.ndarray = None
        self.s_err: float = None
        self.inv_xtx: np.ndarray = None
        self.X_design: np.ndarray = None
        self.y_train: np.ndarray = None
        self.n: int = 0
        self.p: int = degree + 1

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PolynomialOLS":
        """Fits the polynomial OLS model to the calibration data.

        Calculates model parameters and analytical covariance metrics.

        Args:
            x (np.ndarray): 
                Independent variable array of shape (n_samples,).
            y (np.ndarray): 
                Dependent variable array of shape (n_samples,).

        Returns:
            PolynomialOLS: 
                The fitted estimator instance.
        """
        self.y_train = np.asarray(y)
        self.n = len(x)
        self.X_design = design_matrix(x, self.degree)
        
        # OLS closed-form parameter estimation via least squares solver
        self.beta, _, _, _ = np.linalg.lstsq(self.X_design, self.y_train, rcond=None)
        
        # Calculate residual standard error (with degrees-of-freedom correction)
        residuals = self.y_train - self.X_design @ self.beta
        self.s_err = np.sqrt(np.sum(residuals**2) / (self.n - self.p))
        
        # Calculate parametric covariance projection helper
        self.inv_xtx = np.linalg.inv(self.X_design.T @ self.X_design)
        return self

    def predict(self, x: np.ndarray, confidence: float = 0.95) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates point predictions and analytical prediction intervals (PI).

        Args:
            x (np.ndarray): 
                Target points for predictions, shape (n_points,).
            confidence (float, optional): 
                The target coverage probability of the interval. Defaults to 0.95.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                - y_fit (np.ndarray): Mean predicted responses of shape (n_points,).
                - y_lower (np.ndarray): Lower prediction bounds of shape (n_points,).
                - y_upper (np.ndarray): Upper prediction bounds of shape (n_points,).
        """
        x_arr = np.asarray(x)
        y_fit = evaluate_poly(x_arr, self.beta)
        
        X_grid = design_matrix(x_arr, self.degree)
        # Compute local leverage for each target point
        leverage = np.sum((X_grid @ self.inv_xtx) * X_grid, axis=1)
        
        alpha = 1 - confidence
        # Use Student's t-distribution critical values for exact finite-sample coverage
        t_val = t.ppf(1 - alpha / 2, df=self.n - self.p)
        pi_half_width = t_val * self.s_err * np.sqrt(1 + leverage)
        
        return y_fit, y_fit - pi_half_width, y_fit + pi_half_width


class PolynomialDeltaMethod:
    """Forward prediction intervals using the Delta Method with t-distribution scaling."""

    def __init__(self, degree: int = 1):
        self.degree: int = degree
        self.beta: np.ndarray = None
        self.cov_beta: np.ndarray = None
        self.s_err: float = None
        self.n: int = 0
        self.p: int = degree + 1

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PolynomialDeltaMethod":
        self.n = len(x)
        X_design = design_matrix(x, self.degree)
        self.beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
        self.s_err = np.sqrt(np.sum((y - X_design @ self.beta)**2) / (self.n - self.p))
        
        inv_xtx = np.linalg.inv(X_design.T @ X_design)
        self.cov_beta = (self.s_err ** 2) * inv_xtx
        return self

    def predict(self, x: np.ndarray, confidence: float = 0.95) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_arr = np.asarray(x)
        y_fit = evaluate_poly(x_arr, self.beta)
        X_grid = design_matrix(x_arr, self.degree) 
        
        pred_var = np.zeros(len(x_arr))
        for i, jacobian_row in enumerate(X_grid):
            var_fit = jacobian_row.T @ self.cov_beta @ jacobian_row
            pred_var[i] = var_fit + (self.s_err ** 2)
            
        alpha = 1 - confidence
        # FIX: Replacing asymptotic normal z-score with Student's t-score
        t_val = t.ppf(1 - alpha / 2, df=self.n - self.p)
        pi_half_width = t_val * np.sqrt(pred_var)
        
        return y_fit, y_fit - pi_half_width, y_fit + pi_half_width


class PolynomialMonteCarlo:
    """Polynomial prediction intervals using asymptotic Monte Carlo.

    Model-parameter uncertainty is represented by drawing polynomial
    coefficients from their asymptotic multivariate normal distribution:

        beta_sim ~ N(beta_hat, cov_beta_hat)

    Prediction noise is then drawn independently from the selected fitted
    error model.

    For ``error_mode="normal"``:

        epsilon_sim ~ N(0, s_err**2)

    For ``error_mode="bootstrap"``:

        epsilon_sim

    is sampled with replacement from the centered fitted residuals. This
    bootstraps only the measurement-noise distribution; parameter uncertainty
    is still represented by the asymptotic Gaussian parameter distribution.

    Attributes:
        degree:
            Degree of the fitted polynomial.
        n_simulations:
            Number of Monte Carlo realizations used for each prediction.
        error_mode:
            Measurement-noise simulation mode. Supported values are
            ``"normal"`` and ``"bootstrap"``.
        beta:
            Fitted polynomial coefficients in ascending-power order.
        cov_beta:
            Estimated covariance matrix of the fitted coefficients.
        s_err:
            Estimated residual standard deviation.
        residuals:
            Centered, leverage-adjusted residuals used by bootstrap noise
            simulation.
        raw_residuals:
            Unadjusted fitted residuals.
        leverage:
            Diagonal elements of the training-data hat matrix.
        n:
            Number of fitted observations.
        p:
            Number of fitted polynomial coefficients.
    """

    def __init__(
        self,
        degree: int = 1,
        n_simulations: int = 10000,
        error_mode: str = "normal",
    ):
        """Initialize the Monte Carlo polynomial model.

        Args:
            degree:
                Polynomial degree. Must be non-negative.
            n_simulations:
                Number of Monte Carlo realizations. Must be positive.
            error_mode:
                Measurement-noise simulation method:

                - ``"normal"`` draws Gaussian errors using the fitted
                  residual standard deviation.
                - ``"bootstrap"`` samples centered residuals with replacement.

        Raises:
            ValueError:
                If an argument is outside its allowed range.
        """
        if not isinstance(degree, int) or degree < 0:
            raise ValueError("degree must be a non-negative integer.")

        if not isinstance(n_simulations, int) or n_simulations <= 0:
            raise ValueError("n_simulations must be a positive integer.")

        valid_error_modes = {"normal", "bootstrap"}
        if error_mode not in valid_error_modes:
            raise ValueError(
                f"Unknown error_mode {error_mode!r}. "
                f"Expected one of {sorted(valid_error_modes)}."
            )

        self.degree: int = degree
        self.n_simulations: int = n_simulations
        self.error_mode: str = error_mode

        self.beta: np.ndarray | None = None
        self.cov_beta: np.ndarray | None = None
        self.s_err: float | None = None
        self.raw_residuals: np.ndarray | None = None
        self.residuals: np.ndarray | None = None
        self.leverage: np.ndarray | None = None

        self.n: int = 0
        self.p: int = degree + 1

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> "PolynomialMonteCarlo":
        """Fit the polynomial model and estimate parameter covariance.

        The coefficients are estimated by ordinary least squares. For the
        Gaussian regression model, this is also the maximum-likelihood
        estimate of the polynomial coefficients.

        The residual variance is estimated using the degrees-of-freedom
        corrected estimator

            s_err**2 = RSS / (n - p),

        and the coefficient covariance is estimated as

            cov_beta = s_err**2 (X.T X)^(-1).

        For residual-bootstrap prediction noise, the fitted residuals are
        corrected for their observation-specific variance reduction. The
        diagonal leverage values are

            h_ii = diag(X (X.T X)^(-1) X.T),

        and the adjusted residuals are

            r_i_adjusted = r_i / sqrt(1 - h_ii).

        The adjusted residuals are then centered so their empirical mean is
        zero. This correction is handled entirely in the backend; the public
        class interface is unchanged.

        Args:
            x:
                One-dimensional predictor values with shape ``(n_samples,)``.
            y:
                One-dimensional response values with shape ``(n_samples,)``.

        Returns:
            The fitted estimator.

        Raises:
            ValueError:
                If the inputs are invalid or there are insufficient degrees
                of freedom.
            np.linalg.LinAlgError:
                If the design matrix is rank deficient.
        """
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        if x_arr.ndim != 1 or y_arr.ndim != 1:
            raise ValueError("x and y must both be one-dimensional arrays.")

        if x_arr.size != y_arr.size:
            raise ValueError("x and y must contain the same number of values.")

        if x_arr.size == 0:
            raise ValueError("x and y cannot be empty.")

        if not np.all(np.isfinite(x_arr)):
            raise ValueError("x contains non-finite values.")

        if not np.all(np.isfinite(y_arr)):
            raise ValueError("y contains non-finite values.")

        self.n = x_arr.size
        self.p = self.degree + 1

        degrees_of_freedom = self.n - self.p
        if degrees_of_freedom <= 0:
            raise ValueError(
                "The number of observations must exceed the number of "
                "fitted polynomial coefficients."
            )

        X_design = design_matrix(x_arr, self.degree)

        matrix_rank = np.linalg.matrix_rank(X_design)
        if matrix_rank < self.p:
            raise np.linalg.LinAlgError(
                "The polynomial design matrix is rank deficient."
            )

        self.beta, _, _, _ = np.linalg.lstsq(
            X_design,
            y_arr,
            rcond=None,
        )

        self.raw_residuals = y_arr - X_design @ self.beta

        residual_sum_of_squares = float(
            self.raw_residuals @ self.raw_residuals
        )
        residual_variance = residual_sum_of_squares / degrees_of_freedom
        self.s_err = float(np.sqrt(residual_variance))

        inv_xtx = np.linalg.inv(X_design.T @ X_design)
        self.cov_beta = residual_variance * inv_xtx

        # Calculate only the diagonal of the hat matrix
        #
        #     H = X (X.T X)^(-1) X.T,
        #
        # without constructing the full n-by-n matrix. For least squares,
        # Var(r_i) = sigma^2 (1 - h_ii), so fitted residuals are systematically
        # narrower than the underlying measurement-error distribution.
        self.leverage = np.sum(
            (X_design @ inv_xtx) * X_design,
            axis=1,
        )

        # Protect against tiny floating-point excursions outside [0, 1].
        self.leverage = np.clip(self.leverage, 0.0, 1.0)

        one_minus_leverage = np.clip(
            1.0 - self.leverage,
            1e-12,
            None,
        )

        leverage_adjusted_residuals = (
            self.raw_residuals / np.sqrt(one_minus_leverage)
        )

        # Centering restores the zero-mean additive-error assumption after
        # correcting the residual variance. These are the residuals sampled
        # when error_mode="bootstrap".
        self.residuals = (
            leverage_adjusted_residuals
            - np.mean(leverage_adjusted_residuals)
        )

        return self

    def predict(
        self,
        x: np.ndarray,
        confidence: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate predictions and pointwise Monte Carlo prediction intervals.

        For each Monte Carlo realization, this method:

        1. Draws a coefficient vector from

               beta_sim ~ N(beta_hat, cov_beta_hat).

        2. Evaluates the polynomial at the requested prediction points.

        3. Draws an independent measurement error using the selected
           error model.

        4. Adds the measurement error to the simulated fitted mean.

        The returned limits are pointwise empirical quantiles of the
        simulated future responses.

        Args:
            x:
                Prediction points with shape ``(n_points,)``. A scalar is also
                accepted and converted to a one-element array.
            confidence:
                Central prediction-interval coverage probability.

        Returns:
            A tuple containing:

            - fitted mean evaluated using ``beta_hat``;
            - lower pointwise prediction bound;
            - upper pointwise prediction bound.

        Raises:
            RuntimeError:
                If the model has not been fitted.
            ValueError:
                If ``x`` or ``confidence`` is invalid.
        """
        if (
            self.beta is None
            or self.cov_beta is None
            or self.s_err is None
            or self.residuals is None
        ):
            raise RuntimeError("The model must be fitted before prediction.")

        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must lie strictly between 0 and 1.")

        x_arr = np.atleast_1d(np.asarray(x, dtype=float))

        if x_arr.ndim != 1:
            raise ValueError("x must be a scalar or one-dimensional array.")

        if not np.all(np.isfinite(x_arr)):
            raise ValueError("x contains non-finite values.")

        n_points = x_arr.size
        X_grid = design_matrix(x_arr, self.degree)

        # Parameter-estimation uncertainty:
        # beta_hat is assumed asymptotically multivariate normal.
        simulated_betas = np.random.multivariate_normal(
            mean=self.beta,
            cov=self.cov_beta,
            size=self.n_simulations,
        )

        # Each row is one simulated fitted mean curve.
        simulated_means = simulated_betas @ X_grid.T

        # Future measurement noise is drawn from the fitted error model.
        if self.error_mode == "normal":
            simulated_errors = np.random.normal(
                loc=0.0,
                scale=self.s_err,
                size=(self.n_simulations, n_points),
            )

        elif self.error_mode == "bootstrap":
            simulated_errors = np.random.choice(
                self.residuals,
                size=(self.n_simulations, n_points),
                replace=True,
            )

        else:
            # Defensive check in case error_mode is modified after construction.
            raise ValueError(
                f"Unknown error_mode {self.error_mode!r}. "
                "Expected 'normal' or 'bootstrap'."
            )

        simulated_responses = simulated_means + simulated_errors

        alpha = 1.0 - confidence
        lower_quantile = 100.0 * alpha / 2.0
        upper_quantile = 100.0 * (1.0 - alpha / 2.0)

        y_fit = X_grid @ self.beta
        y_lower = np.percentile(
            simulated_responses,
            lower_quantile,
            axis=0,
        )
        y_upper = np.percentile(
            simulated_responses,
            upper_quantile,
            axis=0,
        )

        return y_fit, y_lower, y_upper

class PolynomialQuantileReg:
    """Quantile Regression for distribution-free prediction intervals using a manual solver.

    Bypasses dependencies on statsmodels by implementing and optimizing a manual
    pinball (quantile) loss function using scipy's Nelder-Mead optimization.

    Attributes:
        degree (int): 
            The polynomial degree chosen for the model.
        beta_lower (np.ndarray or None): 
            Lower quantile coefficients of shape (degree + 1,).
        beta_median (np.ndarray or None): 
            Median quantile coefficients of shape (degree + 1,).
        beta_upper (np.ndarray or None): 
            Upper quantile coefficients of shape (degree + 1,).
    """

    def __init__(self, degree: int = 1):
        """Initializes the PolynomialQuantileReg estimator.

        Args:
            degree (int, optional): 
                Degree of the polynomial. Defaults to 1.
        """
        self.degree: int = degree
        self.beta_lower: np.ndarray = None
        self.beta_median: np.ndarray = None
        self.beta_upper: np.ndarray = None

    @staticmethod
    def _pinball_loss(beta: np.ndarray, X: np.ndarray, y: np.ndarray, q: float) -> float:
        """Computes the Pinball (Quantile) Loss function.

        This function penalizes overestimations and underestimations asymmetrically
        depending on the target quantile value.

        Args:
            beta (np.ndarray): 
                Model parameters to evaluate.
            X (np.ndarray): 
                Design matrix of shape (n_samples, parameters).
            y (np.ndarray): 
                Real target variable of shape (n_samples,).
            q (float): 
                Target quantile to estimate (0.0 < q < 1.0).

        Returns:
            float: 
                Total cumulative pinball loss.
        """
        residuals = y - X @ beta
        return np.sum(np.maximum(q * residuals, (q - 1) * residuals))

    def fit(self, x: np.ndarray, y: np.ndarray, confidence: float = 0.95) -> "PolynomialQuantileReg":
        """Fits the lower, median, and upper quantile functions to training data.

        Uses the OLS solution as a robust warm-start initialization vector for
        numerical optimization.

        Args:
            x (np.ndarray): 
                Training independent variable of shape (n_samples,).
            y (np.ndarray): 
                Training dependent variable of shape (n_samples,).
            confidence (float, optional): 
                The target coverage probability. Defaults to 0.95.

        Returns:
            PolynomialQuantileReg: 
                The fitted estimator instance.
        """
        X = design_matrix(x, self.degree)
        alpha = 1 - confidence
        q_lower = alpha / 2
        q_median = 0.50
        q_upper = 1 - alpha / 2

        # Compute OLS solution to serve as a reliable warm start for optimization
        beta_ols, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        # Optimize parameters individually using Nelder-Mead (highly robust for non-smooth loss functions)
        res_lower = minimize(self._pinball_loss, beta_ols, args=(X, y, q_lower), method='Nelder-Mead', options={'maxiter': 5000})
        res_median = minimize(self._pinball_loss, beta_ols, args=(X, y, q_median), method='Nelder-Mead', options={'maxiter': 5000})
        res_upper = minimize(self._pinball_loss, beta_ols, args=(X, y, q_upper), method='Nelder-Mead', options={'maxiter': 5000})

        self.beta_lower = res_lower.x
        self.beta_median = res_median.x
        self.beta_upper = res_upper.x
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates point predictions based on fitted quantiles.

        Args:
            x (np.ndarray): 
                Target evaluation points, shape (n_points,).

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                - y_median (np.ndarray): Median predicted responses of shape (n_points,).
                - y_lower (np.ndarray): Lower quantile prediction bounds of shape (n_points,).
                - y_upper (np.ndarray): Upper quantile prediction bounds of shape (n_points,).
        """
        x_arr = np.asarray(x)
        y_lower = evaluate_poly(x_arr, self.beta_lower)
        y_median = evaluate_poly(x_arr, self.beta_median)
        y_upper = evaluate_poly(x_arr, self.beta_upper)
        return y_median, y_lower, y_upper


# =====================================================================
# 3. INVERSE CALIBRATION MODELS (Estimating X given Y)
# =====================================================================

import numpy as np
from scipy.optimize import minimize, brentq
from scipy.stats import chi2

def evaluate_poly(x, beta):
    """Evaluates a polynomial with coefficients in ascending order: 
    beta[0] + beta[1]*x + beta[2]*x^2 + ..."""
    return sum(b * (x ** i) for i, b in enumerate(beta))

class ProfileLikelihoodCalibrator:
    """Inverts a polynomial function using a profile likelihood ratio formulation.
    Calculates exact, asymmetric calibration confidence intervals by scanning the
    log-likelihood surface directly.
    """

    def __init__(self, degree: int = 1):
        self.degree: int = degree
        self.x_cal: np.ndarray = None
        self.y_cal: np.ndarray = None
        self.n: int = 0
        self.beta_init: np.ndarray = None
        self.poly_init: np.ndarray = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ProfileLikelihoodCalibrator":
        self.x_cal = np.asarray(x)
        self.y_cal = np.asarray(y)
        self.n = len(x)
        self.poly_init = np.polyfit(self.x_cal, self.y_cal, self.degree)
        self.beta_init = self.poly_init[::-1] # Ascending order [intercept, slope, ...]
        return self

    def _joint_neg_log_lik(self, params: list, x_0: float, y_0: float) -> float:
        beta = params[:-1]
        sigma_sq = params[-1]
        
        if sigma_sq <= 1e-15:
            return 1e15
        
        cal_resid = self.y_cal - evaluate_poly(self.x_cal, beta)
        pred_resid = y_0 - evaluate_poly(x_0, beta)
        rss = np.sum(cal_resid**2) + pred_resid**2
        return ((self.n + 1) / 2.0) * np.log(2 * np.pi * sigma_sq) + (rss / (2.0 * sigma_sq))

    def _profile_log_lik(self, x_0: float, y_0: float) -> float:
        init_sigma_sq = np.var(self.y_cal - evaluate_poly(self.x_cal, self.beta_init))
        init_params = list(self.beta_init) + [init_sigma_sq]
        
        # Nelder-Mead is used here for stability over L-BFGS-B
        res = minimize(self._joint_neg_log_lik, init_params, args=(x_0, y_0), method='Nelder-Mead')
        return -res.fun 

    def calibrate(self, y_0: float, confidence: float = 0.95) -> tuple[float, float, float]:
        init_sigma_sq = np.var(self.y_cal - evaluate_poly(self.x_cal, self.beta_init))
        
        # Find starting point for x_0 using polynomial roots
        coefs = self.poly_init.copy()
        coefs[-1] -= y_0
        roots = np.roots(coefs)
        
        real_roots = roots[np.isreal(roots)].real
        if len(real_roots) > 0:
            x_0_init = real_roots[np.argmin(np.abs(real_roots - np.mean(self.x_cal)))]
        else:
            x_0_init = np.mean(self.x_cal)
            
        # To avoid the parameter ordering bug, we structure the objective wrapper to expect:
        # p[:-2] -> beta coefficients
        # p[-2]  -> sigma_sq
        # p[-1]  -> x_0
        obj = lambda p: self._joint_neg_log_lik(list(p[:-2]) + [p[-2]], p[-1], y_0)
        
        init_all = list(self.beta_init) + [init_sigma_sq, x_0_init]
        
        global_res = minimize(obj, init_all, method='Nelder-Mead')
        
        # Correctly extract parameters based on our mapping layout
        x_0_mle = global_res.x[-1]
        log_l_max = -global_res.fun
        
        crit_val = chi2.ppf(confidence, df=1) / 2.0  
        target_f = lambda x: (log_l_max - self._profile_log_lik(x, y_0)) - crit_val
        
        beta_1_approx = self.beta_init[1] if len(self.beta_init) > 1 else 1.0
        se_approx = np.sqrt(init_sigma_sq) / (abs(beta_1_approx) + 1e-6)
        
        # Using 15 instead of 20 to maintain search bracket stability
        lower_bound = brentq(target_f, x_0_mle - 15 * se_approx, x_0_mle)
        upper_bound = brentq(target_f, x_0_mle, x_0_mle + 15 * se_approx)
        
        return x_0_mle, lower_bound, upper_bound
# =====================================================================
# 4. PIPELINE EXECUTION, STATISTICAL REPORTING & PLOTTING
# =====================================================================

if __name__ == "__main__":
    POLY_DEGREE = 2  # Parabolic curve modeling

    # 1. Generate Synthetic Calibration Data
    np.random.seed(42)
    nsamples = 1000
    x_cal = np.linspace(0.1, 2.5, nsamples)
    y_cal = 5.0 + 5 * x_cal - 1.18 * (x_cal**2) + np.random.normal(0, 0.05, size=nsamples)

    x_grid = np.linspace(min(x_cal) - 0.2, max(x_cal) + 0.2, 100)
    y_grid_test = np.linspace(min(y_cal) + 0.1, max(y_cal) - 0.1, 35)

    # 2. Fit Forward Models
    ols = PolynomialOLS(degree=POLY_DEGREE).fit(x_cal, y_cal)
    delta_m = PolynomialDeltaMethod(degree=POLY_DEGREE).fit(x_cal, y_cal)
    mc_normal = PolynomialMonteCarlo(degree=POLY_DEGREE, n_simulations=10000, error_mode='normal').fit(x_cal, y_cal)
    mc_boot = PolynomialMonteCarlo(degree=POLY_DEGREE, n_simulations=10000, error_mode='bootstrap').fit(x_cal, y_cal)
    qreg = PolynomialQuantileReg(degree=POLY_DEGREE).fit(x_cal, y_cal)

    # 3. Fit Inverse Calibrators
    profiler = ProfileLikelihoodCalibrator(degree=POLY_DEGREE).fit(x_cal, y_cal)

    # 4. Compute Forward Prediction Grids
    y_ols_fit, y_ols_lo, y_ols_up = ols.predict(x_grid)
    y_delta_fit, y_delta_lo, y_delta_up = delta_m.predict(x_grid)
    y_mc_norm_fit, y_mc_norm_lo, y_mc_norm_up = mc_normal.predict(x_grid)
    y_mc_boot_fit, y_mc_boot_lo, y_mc_boot_up = mc_boot.predict(x_grid)
    y_qreg_fit, y_qreg_lo, y_qreg_up = qreg.predict(x_grid)

    # =================================================================
    # STATISTICAL REPORTING: Generate 5 Evenly Spaced Concentration Tables
    # =================================================================
    # Select 5 concentrations evenly spaced across the calibration curve range
    evaluation_xs = np.linspace(min(x_cal), max(x_cal), 5)
    
    # Define our models to iterate through
    models = {
        "Analytical OLS Exact": ols,
        "Asymptotic Delta Method": delta_m,
        "MC (Parametric Normal)": mc_normal,
        "MC (Residual Bootstrap)": mc_boot,
        "Quantile Regression (Manual)": qreg
    }

    print("=" * 70)
    print("      FORWARD PREDICTION INTERVALS (PI) AT REPRESENTATIVE POINTS")
    print("=" * 70)

    for idx, eval_x in enumerate(evaluation_xs, start=1):
        print(f"\nTable {idx}: Calibration Point at x = {eval_x:.4f}")
        print("-" * 75)
        print(f"{'PI Method':<30} | {'Lower PI':<12} | {'Fit (Mean/Med)':<14} | {'Upper PI':<12}")
        print("-" * 75)
        
        for name, model in models.items():
            if name == "Quantile Regression (Manual)":
                fit_val, lo_val, up_val = model.predict(np.array([eval_x]))
            else:
                fit_val, lo_val, up_val = model.predict(np.array([eval_x]), confidence=0.95)
            
            print(f"{name:<30} | {lo_val[0]:<12.5f} | {fit_val[0]:<14.5f} | {up_val[0]:<12.5f}")
        print("-" * 75)

    # =================================================================
    # PLOTTING ROUTINES: Global Comparison Plots
    # =================================================================
    plt.figure(figsize=(12, 5))
    
    # Plot A: Forward Comparison Envelope
    plt.subplot(1, 2, 1)
    plt.scatter(x_cal, y_cal, color='gray', s=10, alpha=0.4, label='Data')
    plt.plot(x_grid, y_ols_up, color='darkorange', linestyle='--', label='Analytical OLS')
    plt.plot(x_grid, y_ols_lo, color='darkorange', linestyle='--')
    plt.plot(x_grid, y_delta_up, color='crimson', linestyle=':', label='Delta Method')
    plt.plot(x_grid, y_delta_lo, color='crimson', linestyle=':')
    plt.plot(x_grid, y_mc_norm_up, color='purple', linestyle='-.', label='MC (Normal Noise)')
    plt.plot(x_grid, y_mc_norm_lo, color='purple', linestyle='-.')
    plt.plot(x_grid, y_mc_boot_up, color='deepskyblue', linestyle='-', label='MC (Residual Bootstrap)')
    plt.plot(x_grid, y_mc_boot_lo, color='deepskyblue', linestyle='-')
    plt.plot(x_grid, y_qreg_up, color='forestgreen', linestyle='-', alpha=0.5, label='Quantile Reg (Manual)')
    plt.plot(x_grid, y_qreg_lo, color='forestgreen', linestyle='-', alpha=0.5)
    plt.title("Forward PIs Comparison")
    plt.xlabel("Concentration (x)")
    plt.ylabel("Response (y)")
    plt.legend(loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.2)

    # Plot B: Inverse Profile Likelihood
    plt.subplot(1, 2, 2)
    plt.scatter(x_cal, y_cal, color='gray', s=10, alpha=0.4, label='Data')
    x_mle, x_prof_lo, x_prof_up, y_prof_valid = [], [], [], []
    for y_v in y_grid_test:
        try:
            m, l, u = profiler.calibrate(y_v)
            x_mle.append(m)
            x_prof_lo.append(l)
            x_prof_up.append(u)
            y_prof_valid.append(y_v)
        except ValueError:
            continue
    plt.plot(x_mle, y_prof_valid, color='navy', linewidth=1.5, label='Profile MLE')
    plt.plot(x_prof_lo, y_prof_valid, color='navy', linestyle='--', label='Profile Likelihood 95%')
    plt.plot(x_prof_up, y_prof_valid, color='navy', linestyle='--')
    plt.fill_betweenx(y_prof_valid, x_prof_lo, x_prof_up, color='navy', alpha=0.1)
    plt.title("Inverse Profile Likelihood Calibration")
    plt.xlabel("Concentration (x)")
    plt.ylabel("Response (y)")
    plt.legend(loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    # =================================================================
    # PLOTTING ROUTINES: Dedicated Subplots per Forward Method
    # =================================================================
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.flatten()

    # 1. Subplot for Analytical OLS
    axes[0].scatter(x_cal, y_cal, color='gray', s=10, alpha=0.4, label='Data')
    axes[0].plot(x_grid, y_ols_fit, color='darkorange', linewidth=2, label='Fit')
    axes[0].fill_between(x_grid, y_ols_lo, y_ols_up, color='darkorange', alpha=0.8, label='95% PI')
    axes[0].set_title("Analytical OLS Exact")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.2)

    # 2. Subplot for Delta Method
    axes[1].scatter(x_cal, y_cal, color='gray', s=10, alpha=0.4, label='Data')
    axes[1].plot(x_grid, y_delta_fit, color='crimson', linewidth=2, label='Fit')
    axes[1].fill_between(x_grid, y_delta_lo, y_delta_up, color='crimson', alpha=0.8, label='95% PI')
    axes[1].set_title("Asymptotic Delta Method")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.2)

    # 3. Subplot for MC (Normal Noise)
    axes[2].scatter(x_cal, y_cal, color='gray', s=10, alpha=0.4, label='Data')
    axes[2].plot(x_grid, y_mc_norm_fit, color='purple', linewidth=2, label='Fit')
    axes[2].fill_between(x_grid, y_mc_norm_lo, y_mc_norm_up, color='purple', alpha=0.8, label='95% PI')
    axes[2].set_title("MC (Parametric Normal)")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].legend(loc='upper left')
    axes[2].grid(True, alpha=0.2)

    # 4. Subplot for MC (Residual Bootstrap)
    axes[3].scatter(x_cal, y_cal, color='gray', s=10, alpha=0.4, label='Data')
    axes[3].plot(x_grid, y_mc_boot_fit, color='deepskyblue', linewidth=2, label='Fit')
    axes[3].fill_between(x_grid, y_mc_boot_lo, y_mc_boot_up, color='deepskyblue', alpha=0.8, label='95% PI')
    axes[3].set_title("MC (Residual Bootstrap)")
    axes[3].set_xlabel("x")
    axes[3].set_ylabel("y")
    axes[3].legend(loc='upper left')
    axes[3].grid(True, alpha=0.2)

    # 5. Subplot for Quantile Regression
    axes[4].scatter(x_cal, y_cal, color='gray', s=10, alpha=0.4, label='Data')
    axes[4].plot(x_grid, y_qreg_fit, color='forestgreen', linewidth=2, label='Fit (Median)')
    axes[4].fill_between(x_grid, y_qreg_lo, y_qreg_up, color='forestgreen', alpha=0.8, label='95% PI')
    axes[4].set_title("Quantile Regression (Manual Solver)")
    axes[4].set_xlabel("x")
    axes[4].set_ylabel("y")
    axes[4].legend(loc='upper left')
    axes[4].grid(True, alpha=0.2)

    # Hide the empty 6th subplot
    axes[5].axis('off')

    plt.tight_layout()
    plt.show()


# =====================================================================
# 5. BASELINE-REFERENCED COMPARATIVE SCATTERPLOTS (Relative to OLS Exact)
# =====================================================================

# Color palette mapping for consistent visual identity
method_colors = {
    "Analytical OLS Exact": "#FF8C00",         # Dark Orange
    "Asymptotic Delta Method": "#DC143C",      # Crimson
    "MC (Parametric Normal)": "#800080",       # Purple
    "MC (Residual Bootstrap)": "#00BFFF",      # Deep Sky Blue
    "Quantile Regression (Manual)": "#228B22"  # Forest Green
}

# Collect the specific bounds at our 5 evaluation points
plot_data = {name: {"x": [], "lower": [], "upper": []} for name in models.keys()}

for eval_x in evaluation_xs:
    for name, model in models.items():
        if name == "Quantile Regression (Manual)":
            _, lo_val, up_val = model.predict(np.array([eval_x]))
        else:
            _, lo_val, up_val = model.predict(np.array([eval_x]), confidence=0.95)
        
        plot_data[name]["x"].append(eval_x)
        plot_data[name]["lower"].append(lo_val[0])
        plot_data[name]["upper"].append(up_val[0])

# Convert lists to arrays for vectorized math
for name in plot_data:
    plot_data[name]["x"] = np.array(plot_data[name]["x"])
    plot_data[name]["lower"] = np.array(plot_data[name]["lower"])
    plot_data[name]["upper"] = np.array(plot_data[name]["upper"])

# Extract the baseline (Analytical OLS Exact) values
baseline_lowers = plot_data["Analytical OLS Exact"]["lower"]
baseline_uppers = plot_data["Analytical OLS Exact"]["upper"]

# Initialize the baseline-referenced plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

# Plot 1: Deviation of Lower Prediction Bounds from OLS Exact
ax1.set_title("Lower PI Boundary: Deviation from Analytical OLS", fontsize=12, fontweight='bold', pad=15)
ax1.axhline(0, color='#FF8C00', linestyle='-', linewidth=1.5, alpha=0.7, label='Analytical OLS (Baseline)') 
for name, data in plot_data.items():
    # Subtract OLS Exact baseline
    deviation_lower = data["lower"] - baseline_lowers
    
    # We can skip plotting the OLS baseline as scattered points if we want a cleaner plot,
    # but keeping it shows its exact placement at 0.
    ax1.scatter(
        data["x"], deviation_lower, 
        color=method_colors[name], 
        s=90, marker='o', edgecolors='black', linewidths=0.8, alpha=0.85,
        label=name if name != "Analytical OLS Exact" else None # Already in legend via axhline
    )
ax1.set_xlabel("Concentration (x)", fontsize=10)
ax1.set_ylabel("y_lower - y_lower(OLS Exact)", fontsize=10)
ax1.grid(True, linestyle=':', alpha=0.6)

# Plot 2: Deviation of Upper Prediction Bounds from OLS Exact
ax2.set_title("Upper PI Boundary: Deviation from Analytical OLS", fontsize=12, fontweight='bold', pad=15)
ax2.axhline(0, color='#FF8C00', linestyle='-', linewidth=1.5, alpha=0.7, label='Analytical OLS (Baseline)')
for name, data in plot_data.items():
    # Subtract OLS Exact baseline
    deviation_upper = data["upper"] - baseline_uppers
    
    ax2.scatter(
        data["x"], deviation_upper, 
        color=method_colors[name], 
        s=90, marker='s', edgecolors='black', linewidths=0.8, alpha=0.85,
        label=name if name != "Analytical OLS Exact" else None
    )
ax2.set_xlabel("Concentration (x)", fontsize=10)
ax2.set_ylabel("y_upper - y_upper(OLS Exact)", fontsize=10)
ax2.grid(True, linestyle=':', alpha=0.6)

# Position a unified legend gracefully on the right
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.18, 0.5), borderaxespad=0., frameon=True)

plt.tight_layout()
plt.show()