import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize


@dataclass(frozen=True)
class FitResult:
    """
    Result object returned by maximum likelihood fitting.

    Attributes
    ----------
    theta_hat : numpy.ndarray
        Maximum likelihood estimate of the model parameters.

    covariance : numpy.ndarray
        Estimated covariance matrix of ``theta_hat``.

    hessian : numpy.ndarray or None
        Hessian of the negative log-likelihood evaluated at ``theta_hat``,
        if available.

    loglik : float
        Maximized log-likelihood value.

    success : bool
        Whether the optimizer reported successful convergence.

    message : str
        Optimizer termination message.

    optimizer_result : scipy.optimize.OptimizeResult
        Full raw result returned by ``scipy.optimize.minimize``.
    """

    theta_hat: np.ndarray
    covariance: np.ndarray
    hessian: np.ndarray | None
    loglik: float
    success: bool
    message: str
    optimizer_result: object


class MLE:
    """
    Maximum Likelihood Estimator.

    This class performs maximum likelihood estimation for an arbitrary
    deterministic model and additive residual error model.

    The observation model is assumed to be:

        y = f(x; theta) + epsilon

    where epsilon is described by the selected error model.

    The log-likelihood is constructed automatically as:

        log L(theta)
        =
        sum_i log p_epsilon(y_i - f(x_i; theta) | x_i)
    """

    def __init__(self, model, error_model, x, y):
        self.model = model
        self.error_model = error_model
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)

    def loglik(self, theta):
        """
        Evaluate the log-likelihood.

        Parameters
        ----------
        theta : array-like
            Model parameter vector.

        Returns
        -------
        float
            Log-likelihood evaluated at ``theta``.
        """
        theta = np.asarray(theta, dtype=float)

        prediction = self.model(self.x, theta)
        residuals = self.y - prediction

        log_likelihood = 0.0

        for residual, x_i in zip(residuals, self.x):
            log_likelihood += self.error_model.logpdf(residual, x_i)

        return float(log_likelihood)

    def negative_loglik(self, theta):
        """
        Evaluate the negative log-likelihood.

        Parameters
        ----------
        theta : array-like
            Model parameter vector.

        Returns
        -------
        float
            Negative log-likelihood evaluated at ``theta``.
        """
        return -self.loglik(theta)

    def fit(self, initial_guess, method="BFGS", **kwargs):
        """
        Perform maximum likelihood estimation.

        Parameters
        ----------
        initial_guess : array-like
            Initial parameter estimate supplied to the optimizer.

        method : str, optional
            Optimization method passed to ``scipy.optimize.minimize``.

            Default is ``"BFGS"`` because BFGS returns an inverse-Hessian
            approximation, which can be used as an approximate covariance
            matrix.

        **kwargs
            Additional keyword arguments passed directly to
            ``scipy.optimize.minimize``.

        Returns
        -------
        FitResult
            Structured fit result containing parameter estimates,
            covariance estimate, Hessian information if available, and
            optimizer diagnostics.

        Raises
        ------
        RuntimeError
            If the optimizer does not provide enough curvature information
            to estimate the covariance matrix.
        """
        initial_guess = np.asarray(initial_guess, dtype=float)

        optimizer_result = minimize(
            self.negative_loglik,
            x0=initial_guess,
            method=method,
            **kwargs,
        )

        theta_hat = np.asarray(optimizer_result.x, dtype=float)

        covariance, hessian = self._extract_covariance_and_hessian(
            optimizer_result
        )

        return FitResult(
            theta_hat=theta_hat,
            covariance=covariance,
            hessian=hessian,
            loglik=self.loglik(theta_hat),
            success=bool(optimizer_result.success),
            message=str(optimizer_result.message),
            optimizer_result=optimizer_result,
        )

    def _extract_covariance_and_hessian(self, optimizer_result):
        """
        Extract covariance information from an optimizer result.

        Parameters
        ----------
        optimizer_result : scipy.optimize.OptimizeResult
            Raw result returned by ``scipy.optimize.minimize``.

        Returns
        -------
        tuple
            Pair ``(covariance, hessian)``.

            ``covariance`` is the estimated covariance matrix of the MLE.

            ``hessian`` is the Hessian of the negative log-likelihood if it
            can be recovered directly or by inverting the covariance estimate.
            Otherwise it is ``None``.

        Notes
        -----
        For maximum likelihood estimation, the asymptotic covariance matrix is

            Cov(theta_hat) ~= H^{-1}

        where H is the Hessian of the negative log-likelihood at the optimum.

        Some optimizers provide ``hess_inv`` directly. In that case, it is
        treated as the covariance estimate.
        """

        if hasattr(optimizer_result, "hess_inv"):
            hess_inv = optimizer_result.hess_inv

            if hasattr(hess_inv, "todense"):
                covariance = np.asarray(hess_inv.todense(), dtype=float)
            else:
                covariance = np.asarray(hess_inv, dtype=float)

            covariance = self._validate_covariance(covariance)

            try:
                hessian = np.linalg.inv(covariance)
            except np.linalg.LinAlgError:
                hessian = None

            return covariance, hessian

        if hasattr(optimizer_result, "hess"):
            hessian = np.asarray(optimizer_result.hess, dtype=float)
            hessian = self._validate_hessian(hessian)

            try:
                covariance = np.linalg.inv(hessian)
            except np.linalg.LinAlgError as exc:
                raise np.linalg.LinAlgError(
                    "Optimizer supplied a Hessian, but it could not be "
                    "inverted to estimate the covariance matrix."
                ) from exc

            return covariance, hessian

        raise RuntimeError(
            "Optimizer result does not contain Hessian or inverse-Hessian "
            "information. Use an optimizer that provides curvature information, "
            "such as BFGS, or attach covariance estimation before building "
            "intervals."
        )

    def _validate_covariance(self, covariance):
        """
        Validate covariance matrix shape.

        Parameters
        ----------
        covariance : array-like
            Candidate covariance matrix.

        Returns
        -------
        numpy.ndarray
            Validated covariance matrix.
        """
        covariance = np.asarray(covariance, dtype=float)

        if covariance.ndim != 2:
            raise ValueError("Covariance matrix must be two-dimensional.")

        if covariance.shape[0] != covariance.shape[1]:
            raise ValueError("Covariance matrix must be square.")

        return covariance

    def _validate_hessian(self, hessian):
        """
        Validate Hessian matrix shape.

        Parameters
        ----------
        hessian : array-like
            Candidate Hessian matrix.

        Returns
        -------
        numpy.ndarray
            Validated Hessian matrix.
        """
        hessian = np.asarray(hessian, dtype=float)

        if hessian.ndim != 2:
            raise ValueError("Hessian matrix must be two-dimensional.")

        if hessian.shape[0] != hessian.shape[1]:
            raise ValueError("Hessian matrix must be square.")

        return hessian
