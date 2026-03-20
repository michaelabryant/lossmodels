import numpy as np


def _validate_pmf(pmf):
    pmf = np.asarray(pmf, dtype=float)

    if pmf.ndim != 1:
        raise ValueError("pmf must be 1D.")
    if len(pmf) == 0:
        raise ValueError("pmf must not be empty.")
    if np.any(pmf < 0):
        raise ValueError("pmf must be nonnegative.")

    total = pmf.sum()
    if total <= 0:
        raise ValueError("pmf must sum to a positive value.")

    return pmf / total


def var_from_pmf(pmf, h: float, q: float):
    """
    Compute VaR from a lattice pmf.
    """
    if not (0 < q < 1):
        raise ValueError("q must be in (0,1).")
    if h <= 0:
        raise ValueError("h must be positive.")

    pmf = _validate_pmf(pmf)
    cdf = np.cumsum(pmf)

    idx = np.searchsorted(cdf, q)
    return float(idx * h)


def tvar_from_pmf(pmf, h: float, q: float):
    """
    Compute TVaR from a lattice pmf.
    """
    if not (0 < q < 1):
        raise ValueError("q must be in (0,1).")
    if h <= 0:
        raise ValueError("h must be positive.")

    pmf = _validate_pmf(pmf)
    x = h * np.arange(len(pmf))

    var = var_from_pmf(pmf, h, q)

    tail_mask = x >= var
    tail_prob = pmf[tail_mask].sum()

    if tail_prob == 0:
        return var

    return float(np.sum(x[tail_mask] * pmf[tail_mask]) / tail_prob)


def stop_loss_from_pmf(pmf, h: float, d: float):
    """
    Compute stop-loss premium E[(S - d)^+].
    """
    if h <= 0:
        raise ValueError("h must be positive.")

    pmf = _validate_pmf(pmf)
    x = h * np.arange(len(pmf))

    excess = np.maximum(x - d, 0.0)
    return float(np.sum(excess * pmf))


def mean_from_pmf(pmf, h: float):
    pmf = _validate_pmf(pmf)
    x = h * np.arange(len(pmf))
    return float(np.sum(x * pmf))