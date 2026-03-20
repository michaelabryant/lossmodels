import numpy as np


def log_likelihood(model, data) -> float:
    """
    Compute the log-likelihood of observed data under a fitted model.

    Parameters
    ----------
    model : object
        Model instance with a pdf(x) method for continuous models
        or pmf(x) method for discrete models.
    data : array-like
        Observed data.

    Returns
    -------
    float
        Log-likelihood value.
    """
    data = np.asarray(data)
    if data.size == 0:
        raise ValueError("data must not be empty.")

    if hasattr(model, "pdf"):
        vals = np.array([model.pdf(x) for x in data], dtype=float)
    elif hasattr(model, "pmf"):
        vals = np.array([model.pmf(x) for x in data], dtype=float)
    else:
        raise TypeError("model must implement either pdf(x) or pmf(x).")

    if np.any(~np.isfinite(vals)) or np.any(vals <= 0):
        return float(-np.inf)

    return float(np.sum(np.log(vals)))


def aic(model, data, k: int) -> float:
    """
    Compute Akaike Information Criterion.

    Parameters
    ----------
    model : object
        Fitted model.
    data : array-like
        Observed data.
    k : int
        Number of estimated parameters.

    Returns
    -------
    float
        AIC value.
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    ll = log_likelihood(model, data)
    if not np.isfinite(ll):
        return float(np.inf)

    return float(2 * k - 2 * ll)


def bic(model, data, k: int) -> float:
    """
    Compute Bayesian Information Criterion.

    Parameters
    ----------
    model : object
        Fitted model.
    data : array-like
        Observed data.
    k : int
        Number of estimated parameters.

    Returns
    -------
    float
        BIC value.
    """
    data = np.asarray(data)
    if data.size == 0:
        raise ValueError("data must not be empty.")
    if k <= 0:
        raise ValueError("k must be positive.")

    ll = log_likelihood(model, data)
    if not np.isfinite(ll):
        return float(np.inf)

    n = data.size
    return float(np.log(n) * k - 2 * ll)