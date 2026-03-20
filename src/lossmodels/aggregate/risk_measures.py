import numpy as np


def _validate_losses(losses: np.ndarray) -> np.ndarray:
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 1:
        raise ValueError("losses must be 1D.")
    if len(losses) == 0:
        raise ValueError("losses must not be empty.")
    return losses


def _empirical_var(losses: np.ndarray, q: float) -> float:
    """
    Empirical VaR defined as the smallest observed loss whose empirical CDF
    is at least q.
    """
    sorted_losses = np.sort(losses)
    ecdf = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)
    idx = np.searchsorted(ecdf, q, side="left")
    return float(sorted_losses[idx])


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
        Empirical VaR at level q, defined as the smallest observed loss
        whose empirical CDF reaches q.
    """
    if not (0 < q < 1):
        raise ValueError("q must be between 0 and 1.")
    losses = _validate_losses(losses)
    return _empirical_var(losses, q)


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
        Empirical TVaR at level q, computed as the mean of losses
        greater than or equal to VaR(q).
    """
    if not (0 < q < 1):
        raise ValueError("q must be between 0 and 1.")
    losses = _validate_losses(losses)
    var_q = _empirical_var(losses, q)
    tail = losses[losses >= var_q]
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
    losses = _validate_losses(losses)
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
    losses = _validate_losses(losses)
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
    losses = _validate_losses(losses)
    return float(np.mean(losses > d))