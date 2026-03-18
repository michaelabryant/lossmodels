import numpy as np


def var(losses: np.ndarray, q: float) -> float:
    """
    Value-at-Risk at level q.

    Parameters
    ----------
    losses : np.ndarray
        Array of loss samples.
    q : float
        Quantile level, with 0 < q < 1.

    Returns
    -------
    float
        VaR at level q.
    """
    if not (0 < q < 1):
        raise ValueError("q must be between 0 and 1.")
    if len(losses) == 0:
        raise ValueError("losses must not be empty.")

    return float(np.quantile(losses, q))


def tvar(losses: np.ndarray, q: float) -> float:
    """
    Tail Value-at-Risk at level q.

    Parameters
    ----------
    losses : np.ndarray
        Array of loss samples.
    q : float
        Quantile level, with 0 < q < 1.

    Returns
    -------
    float
        TVaR at level q.
    """
    if not (0 < q < 1):
        raise ValueError("q must be between 0 and 1.")
    if len(losses) == 0:
        raise ValueError("losses must not be empty.")

    var_q = np.quantile(losses, q)
    tail = losses[losses > var_q]

    if len(tail) == 0:
        return float(var_q)

    return float(np.mean(tail))


def stop_loss(losses: np.ndarray, d: float) -> float:
    """
    Expected stop-loss premium E[(S - d)+].

    Parameters
    ----------
    losses : np.ndarray
        Array of loss samples.
    d : float
        Deductible / attachment point, with d >= 0.

    Returns
    -------
    float
        Expected stop-loss value.
    """
    if d < 0:
        raise ValueError("d must be nonnegative.")
    if len(losses) == 0:
        raise ValueError("losses must not be empty.")

    return float(np.mean(np.maximum(losses - d, 0.0)))


def lev(losses: np.ndarray, d: float) -> float:
    """
    Limited expected value E[min(S, d)].

    Parameters
    ----------
    losses : np.ndarray
        Array of loss samples.
    d : float
        Limit, with d >= 0.

    Returns
    -------
    float
        Limited expected value.
    """
    if d < 0:
        raise ValueError("d must be nonnegative.")
    if len(losses) == 0:
        raise ValueError("losses must not be empty.")

    return float(np.mean(np.minimum(losses, d)))


def exceedance_probability(losses: np.ndarray, d: float) -> float:
    """
    Probability P(S > d).

    Parameters
    ----------
    losses : np.ndarray
        Array of loss samples.
    d : float
        Threshold.

    Returns
    -------
    float
        Estimated exceedance probability.
    """
    if len(losses) == 0:
        raise ValueError("losses must not be empty.")

    return float(np.mean(losses > d))