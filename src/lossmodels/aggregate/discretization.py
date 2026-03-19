import numpy as np


def discretize_severity(severity, h: float, max_loss: float):
    """
    Discretize a severity model onto a lattice with spacing h.

    Parameters
    ----------
    severity : object
        Severity model with a cdf(x) method.
    h : float
        Lattice step size.
    max_loss : float
        Maximum loss level for discretization. The final bucket absorbs all
        remaining tail probability.

    Returns
    -------
    np.ndarray
        Probability mass vector f where f[j] approximates:

            P(j h <= X < (j+1) h),   for j < m
            P(X >= m h),             for j = m

        where m = int(max_loss / h).

    Notes
    -----
    This is a simple "upper discretization" suitable for a first Panjer
    implementation.
    """
    if h <= 0:
        raise ValueError("h must be positive.")
    if max_loss <= 0:
        raise ValueError("max_loss must be positive.")
    if not hasattr(severity, "cdf"):
        raise TypeError("severity must implement cdf(x).")

    m = int(np.floor(max_loss / h))
    if m < 1:
        raise ValueError("max_loss must be at least as large as h.")

    probs = np.zeros(m + 1, dtype=float)

    # Interior buckets
    for j in range(m):
        left = j * h
        right = (j + 1) * h
        probs[j] = float(severity.cdf(right) - severity.cdf(left))

    # Final bucket absorbs the tail
    probs[m] = float(1.0 - severity.cdf(m * h))

    # Numerical cleanup
    probs = np.maximum(probs, 0.0)
    total = probs.sum()

    if total <= 0:
        raise ValueError("Discretization produced zero total probability.")

    probs /= total
    return probs


def bucket_representatives(h: float, size: int) -> np.ndarray:
    """
    Return bucket representatives j*h for j = 0, ..., size-1.
    """
    if h <= 0:
        raise ValueError("h must be positive.")
    if size <= 0:
        raise ValueError("size must be positive.")

    return h * np.arange(size, dtype=float)


def mean_from_discretized_pmf(pmf: np.ndarray, h: float) -> float:
    """
    Approximate the mean from a discretized severity pmf.

    Uses bucket representatives j*h.
    """
    pmf = np.asarray(pmf, dtype=float)

    if pmf.ndim != 1:
        raise ValueError("pmf must be a 1D array.")
    if len(pmf) == 0:
        raise ValueError("pmf must not be empty.")
    if h <= 0:
        raise ValueError("h must be positive.")
    if np.any(pmf < 0):
        raise ValueError("pmf must be nonnegative.")

    total = pmf.sum()
    if total <= 0:
        raise ValueError("pmf must sum to a positive value.")

    pmf = pmf / total
    x = bucket_representatives(h, len(pmf))
    return float(np.sum(x * pmf))