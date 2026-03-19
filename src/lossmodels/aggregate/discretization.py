import numpy as np


def discretize_severity(severity, h: float, max_loss: float, method: str = "upper"):
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
    method : {"upper", "lower", "midpoint"}
        Discretization scheme.

    Returns
    -------
    np.ndarray
        Probability mass vector on the lattice.
    """
    if h <= 0:
        raise ValueError("h must be positive.")
    if max_loss <= 0:
        raise ValueError("max_loss must be positive.")
    if not hasattr(severity, "cdf"):
        raise TypeError("severity must implement cdf(x).")
    if method not in {"upper", "lower", "midpoint"}:
        raise ValueError("method must be 'upper', 'lower', or 'midpoint'.")

    m = int(np.floor(max_loss / h))
    if m < 1:
        raise ValueError("max_loss must be at least as large as h.")

    probs = np.zeros(m + 1, dtype=float)

    if method == "upper":
        for j in range(m):
            left = j * h
            right = (j + 1) * h
            probs[j] = float(severity.cdf(right) - severity.cdf(left))
        probs[m] = float(1.0 - severity.cdf(m * h))

    elif method == "lower":
        probs[0] = float(severity.cdf(h))
        for j in range(1, m):
            left = (j - 1) * h
            right = j * h
            probs[j] = float(severity.cdf(right) - severity.cdf(left))
        probs[m] = float(1.0 - severity.cdf((m - 1) * h))

    elif method == "midpoint":
        probs[0] = float(severity.cdf(h / 2.0))
        for j in range(1, m):
            left = (j - 0.5) * h
            right = (j + 0.5) * h
            probs[j] = float(severity.cdf(right) - severity.cdf(left))
        probs[m] = float(1.0 - severity.cdf((m - 0.5) * h))

    probs = np.maximum(probs, 0.0)
    total = probs.sum()

    if total <= 0:
        raise ValueError("Discretization produced zero total probability.")

    probs /= total
    return probs


def bucket_representatives(h: float, size: int) -> np.ndarray:
    if h <= 0:
        raise ValueError("h must be positive.")
    if size <= 0:
        raise ValueError("size must be positive.")

    return h * np.arange(size, dtype=float)


def mean_from_discretized_pmf(pmf: np.ndarray, h: float) -> float:
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