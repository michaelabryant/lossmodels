import numpy as np
from scipy.optimize import minimize
from scipy.stats import gamma as gamma_dist
from scipy.stats import weibull_min

from ..frequency import NegativeBinomial, Poisson
from ..severity import Exponential, Gamma, Lognormal, Weibull


def _validate_positive_data(data, name: str = "data") -> np.ndarray:
    """Validate that input data are nonempty and strictly positive."""
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if np.any(data <= 0):
        raise ValueError(f"{name} must contain only positive values.")
    return data


def _validate_count_data(data, name: str = "data") -> np.ndarray:
    """Validate that input data are nonempty, nonnegative, and integer-valued."""
    data = np.asarray(data)
    if data.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if np.any(data < 0):
        raise ValueError(f"{name} must contain only nonnegative values.")
    if not np.all(np.equal(data, np.floor(data))):
        raise ValueError(f"{name} must contain only integer-valued counts.")
    return data.astype(int)


def fit_exponential(data) -> Exponential:
    """
    Fit an Exponential severity model by maximum likelihood.

    For X_i ~ Exponential(rate), the MLE is:
        rate_hat = 1 / mean(data)
    """
    data = _validate_positive_data(data)
    mean_x = np.mean(data)
    if mean_x <= 0:
        raise ValueError("Mean of data must be positive.")
    rate_hat = 1.0 / mean_x
    return Exponential(rate=float(rate_hat))


def fit_lognormal(data) -> Lognormal:
    """
    Fit a Lognormal severity model by maximum likelihood.

    If log(X) ~ Normal(mu, sigma^2), the MLEs are:
        mu_hat = mean(log(data))
        sigma_hat = sqrt(mean((log(data) - mu_hat)^2))

    Notes
    -----
    This uses the MLE version of the variance (ddof=0).
    """
    data = _validate_positive_data(data)
    log_data = np.log(data)
    mu_hat = float(np.mean(log_data))
    sigma_hat = float(np.sqrt(np.mean((log_data - mu_hat) ** 2)))
    return Lognormal(mu=mu_hat, sigma=sigma_hat)


def fit_negbinomial(data) -> NegativeBinomial:
    """
    Fit a Negative Binomial frequency model by numerical maximum likelihood.

    Parameterization
    ----------------
    N = number of failures before the r-th success

    Support: {0, 1, 2, ...}
    Mean = r(1-p)/p
    Variance = r(1-p)/p^2
    """
    data = _validate_count_data(data)
    mean_x = float(np.mean(data))
    var_x = float(np.var(data, ddof=0))

    if var_x > mean_x and mean_x > 0:
        p0 = mean_x / var_x
        r0 = mean_x**2 / (var_x - mean_x)
        initial = np.array([r0, p0], dtype=float)
    else:
        initial = np.array([1.0, 0.5], dtype=float)

    bounds = [
        (1e-8, None),
        (1e-8, 1.0 - 1e-8),
    ]

    def neg_log_likelihood(params):
        r, p = params
        try:
            model = NegativeBinomial(r=r, p=p)
            pmf_vals = np.array([model.pmf(int(x)) for x in data], dtype=float)
            if np.any(~np.isfinite(pmf_vals)) or np.any(pmf_vals <= 0):
                return np.inf
            return float(-np.sum(np.log(pmf_vals)))
        except Exception:
            return np.inf

    result = minimize(
        neg_log_likelihood,
        x0=initial,
        bounds=bounds,
        method="L-BFGS-B",
    )
    if not result.success:
        raise RuntimeError(f"Negative Binomial MLE optimization failed: {result.message}")

    r_hat, p_hat = result.x
    return NegativeBinomial(r=float(r_hat), p=float(p_hat))


def fit_poisson(data) -> Poisson:
    """
    Fit a Poisson frequency model by maximum likelihood.

    For N_i ~ Poisson(lam), the MLE is:
        lam_hat = mean(data)

    Notes
    -----
    An all-zero dataset is valid and yields lam_hat = 0.
    """
    data = _validate_count_data(data)
    lam_hat = float(np.mean(data))
    if lam_hat < 0:
        raise ValueError("Estimated lambda must be nonnegative.")
    return Poisson(lam=lam_hat)


def fit_gamma(data) -> Gamma:
    """
    Fit a Gamma severity model by maximum likelihood using SciPy.

    Returns
    -------
    Gamma
        Fitted Gamma(alpha, theta) model.

    Notes
    -----
    This constrains loc = 0 so the support is x > 0, consistent with the
    severity model implementation.
    """
    data = _validate_positive_data(data)
    alpha_hat, loc_hat, theta_hat = gamma_dist.fit(data, floc=0)
    if loc_hat != 0:
        raise RuntimeError("Gamma fit returned nonzero location despite floc=0.")
    return Gamma(alpha=float(alpha_hat), theta=float(theta_hat))


def fit_weibull(data) -> Weibull:
    """
    Fit a Weibull severity model by maximum likelihood using SciPy.

    Returns
    -------
    Weibull
        Fitted Weibull(k, lam) model.

    Notes
    -----
    This constrains loc = 0 so the support is x > 0, consistent with the
    severity model implementation.
    """
    data = _validate_positive_data(data)
    k_hat, loc_hat, lam_hat = weibull_min.fit(data, floc=0)
    if loc_hat != 0:
        raise RuntimeError("Weibull fit returned nonzero location despite floc=0.")
    return Weibull(k=float(k_hat), lam=float(lam_hat))


def fit_mle(model_class, data, initial_params, bounds=None):
    """
    Generic numerical maximum likelihood estimation for models with a pdf method.

    Parameters
    ----------
    model_class : class
        A model class that can be instantiated as model_class(*params) and
        provides a pdf(x) method.
    data : array-like
        Observed data.
    initial_params : array-like
        Initial parameter guess for the optimizer.
    bounds : list of tuple, optional
        Bounds passed to scipy.optimize.minimize.

    Returns
    -------
    object
        Fitted model instance of type model_class.
    """
    data = _validate_positive_data(data)
    initial_params = np.asarray(initial_params, dtype=float)
    if initial_params.size == 0:
        raise ValueError("initial_params must not be empty.")

    def neg_log_likelihood(params):
        try:
            model = model_class(*params)
            pdf_vals = np.array([model.pdf(x) for x in data], dtype=float)
            if np.any(~np.isfinite(pdf_vals)) or np.any(pdf_vals <= 0):
                return np.inf
            return float(-np.sum(np.log(pdf_vals)))
        except Exception:
            return np.inf

    result = minimize(
        neg_log_likelihood,
        x0=initial_params,
        bounds=bounds,
        method="L-BFGS-B" if bounds is not None else "BFGS",
    )
    if not result.success:
        raise RuntimeError(f"MLE optimization failed: {result.message}")

    return model_class(*result.x)