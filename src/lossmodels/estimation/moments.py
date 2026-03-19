import numpy as np

from ..frequency import Poisson
from ..severity import Exponential, Gamma, Lognormal, Weibull


def _validate_positive_data(data, name: str = "data") -> np.ndarray:
    """
    Validate that input data are nonempty and strictly positive.
    """
    data = np.asarray(data, dtype=float)

    if data.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if np.any(data <= 0):
        raise ValueError(f"{name} must contain only positive values.")

    return data


def _validate_count_data(data, name: str = "data") -> np.ndarray:
    """
    Validate that input data are nonempty, nonnegative, and integer-valued.
    """
    data = np.asarray(data)

    if data.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if np.any(data < 0):
        raise ValueError(f"{name} must contain only nonnegative values.")
    if not np.all(np.equal(data, np.floor(data))):
        raise ValueError(f"{name} must contain only integer-valued counts.")

    return data.astype(int)


def fit_poisson_moments(data) -> Poisson:
    """
    Fit a Poisson frequency model by the method of moments.

    For Poisson(lambda):
        E[N] = lambda

    So:
        lambda_hat = sample mean
    """
    data = _validate_count_data(data)

    lam_hat = float(np.mean(data))
    if lam_hat <= 0:
        raise ValueError("Estimated lambda must be positive.")

    return Poisson(lam=lam_hat)


def fit_exponential_moments(data) -> Exponential:
    """
    Fit an Exponential severity model by the method of moments.

    For Exponential(rate):
        E[X] = 1 / rate

    So:
        rate_hat = 1 / sample mean
    """
    data = _validate_positive_data(data)

    mean_x = float(np.mean(data))
    if mean_x <= 0:
        raise ValueError("Sample mean must be positive.")

    rate_hat = 1.0 / mean_x
    return Exponential(rate=rate_hat)


def fit_gamma_moments(data) -> Gamma:
    """
    Fit a Gamma severity model by the method of moments.

    For Gamma(alpha, theta):
        E[X] = alpha * theta
        Var(X) = alpha * theta^2

    Solving:
        alpha_hat = mean^2 / var
        theta_hat = var / mean
    """
    data = _validate_positive_data(data)

    mean_x = float(np.mean(data))
    var_x = float(np.var(data, ddof=0))

    if mean_x <= 0:
        raise ValueError("Sample mean must be positive.")
    if var_x <= 0:
        raise ValueError("Sample variance must be positive.")

    alpha_hat = mean_x**2 / var_x
    theta_hat = var_x / mean_x

    return Gamma(alpha=alpha_hat, theta=theta_hat)


def fit_lognormal_moments(data) -> Lognormal:
    """
    Fit a Lognormal severity model by the method of moments.

    For Lognormal(mu, sigma):
        E[X] = exp(mu + sigma^2 / 2)
        Var(X) = (exp(sigma^2) - 1) * exp(2mu + sigma^2)

    Solving:
        sigma^2 = log(1 + var / mean^2)
        mu = log(mean) - sigma^2 / 2
    """
    data = _validate_positive_data(data)

    mean_x = float(np.mean(data))
    var_x = float(np.var(data, ddof=0))

    if mean_x <= 0:
        raise ValueError("Sample mean must be positive.")
    if var_x < 0:
        raise ValueError("Sample variance must be nonnegative.")
    if var_x == 0:
        raise ValueError("Sample variance must be positive for Lognormal method-of-moments fit.")

    sigma2_hat = np.log(1.0 + var_x / (mean_x**2))
    sigma_hat = float(np.sqrt(sigma2_hat))
    mu_hat = float(np.log(mean_x) - 0.5 * sigma2_hat)

    return Lognormal(mu=mu_hat, sigma=sigma_hat)


def fit_weibull_moments(data) -> Weibull:
    """
    Fit a Weibull severity model by the method of moments.

    Uses the coefficient of variation (CV) equation:

        CV^2(k) = Gamma(1 + 2/k) / Gamma(1 + 1/k)^2 - 1

    and solves numerically for k. Then:

        lambda_hat = mean / Gamma(1 + 1/k)

    Notes
    -----
    This is a numerical method-of-moments fit, since Weibull does not have
    a simple closed-form moment solution.
    """
    data = _validate_positive_data(data)

    mean_x = float(np.mean(data))
    var_x = float(np.var(data, ddof=0))

    if mean_x <= 0:
        raise ValueError("Sample mean must be positive.")
    if var_x <= 0:
        raise ValueError("Sample variance must be positive.")

    cv2_target = var_x / (mean_x**2)

    from scipy.optimize import brentq
    from scipy.special import gamma as gamma_func

    def cv2_weibull(k):
        g1 = gamma_func(1.0 + 1.0 / k)
        g2 = gamma_func(1.0 + 2.0 / k)
        return g2 / (g1**2) - 1.0

    def objective(k):
        return cv2_weibull(k) - cv2_target

    # Weibull CV^2 decreases with k, and this bracket works well in practice.
    k_hat = float(brentq(objective, 0.1, 100.0))
    lam_hat = float(mean_x / gamma_func(1.0 + 1.0 / k_hat))

    return Weibull(k=k_hat, lam=lam_hat)