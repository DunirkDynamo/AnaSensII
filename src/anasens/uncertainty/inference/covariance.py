import numpy as np


def numerical_hessian(func, theta, step_size=1e-5):
    """
    Estimate the Hessian matrix of a scalar function numerically.

    Parameters
    ----------
    func : callable
        Scalar-valued function of a parameter vector.

        Must accept

            func(theta)

        and return a scalar.

    theta : array-like
        Parameter vector at which to evaluate the Hessian.

    step_size : float, optional
        Finite-difference step size.

        Default is 1e-5.

    Returns
    -------
    numpy.ndarray
        Hessian matrix evaluated at theta.

    Notes
    -----
    This function uses a centered finite-difference approximation:

        H_ij ≈ [f(θ+h_i+h_j) - f(θ+h_i-h_j)
                - f(θ-h_i+h_j) + f(θ-h_i-h_j)] / (4h²)

    The input function should be reasonably smooth near theta.
    """
    theta = np.asarray(theta, dtype=float)
    n_params = theta.size

    hessian = np.zeros((n_params, n_params), dtype=float)

    for i in range(n_params):
        for j in range(n_params):
            step_i = np.zeros(n_params)
            step_j = np.zeros(n_params)

            step_i[i] = step_size
            step_j[j] = step_size

            f_pp = func(theta + step_i + step_j)
            f_pm = func(theta + step_i - step_j)
            f_mp = func(theta - step_i + step_j)
            f_mm = func(theta - step_i - step_j)

            hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (
                4.0 * step_size**2
            )

    return hessian


def covariance_from_loglik(loglik, theta_hat, step_size=1e-5):
    """
    Estimate parameter covariance from log-likelihood curvature.

    Parameters
    ----------
    loglik : callable
        Log-likelihood function.

        Must accept

            loglik(theta)

        and return a scalar log-likelihood value.

    theta_hat : array-like
        Maximum likelihood estimate of the parameter vector.

    step_size : float, optional
        Finite-difference step size used for Hessian estimation.

        Default is 1e-5.

    Returns
    -------
    numpy.ndarray
        Estimated covariance matrix of the MLE.

    Notes
    -----
    Under standard regularity conditions, the MLE is asymptotically normal:

        theta_hat ~ N(theta, I(theta)^(-1))

    The observed Fisher information is approximated by the Hessian of the
    negative log-likelihood at the MLE:

        I_hat = ∇²[-log L(theta_hat)]

    The covariance estimate is then:

        Cov(theta_hat) ≈ I_hat^(-1)

    This is an asymptotic approximation. It is most reliable when the
    likelihood is smooth and locally well-approximated by a quadratic.
    """
    theta_hat = np.asarray(theta_hat, dtype=float)

    observed_information = numerical_hessian(
        lambda theta: -loglik(theta),
        theta_hat,
        step_size=step_size,
    )

    try:
        covariance = np.linalg.inv(observed_information)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "Observed information matrix is singular and cannot be inverted. "
            "The model may be poorly identified, overparameterized, or the "
            "finite-difference step size may be inappropriate."
        ) from exc

    return covariance