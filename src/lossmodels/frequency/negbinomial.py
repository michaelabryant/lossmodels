import numpy as np
from scipy.stats import nbinom

from .base import FrequencyModel


class NegativeBinomial(FrequencyModel):
    """
    Negative Binomial frequency model.

    Parameterization
    ----------------
    N = number of failures before the r-th success
    Support: {0, 1, 2, ...}

    Parameters
    ----------
    r : float
        Number of successes, with r > 0.
    p : float
        Probability of success, with 0 < p <= 1.

    Notes
    -----
    This matches SciPy's negative binomial parameterization:
    scipy.stats.nbinom(r, p)

    Under this convention:
        E[N] = r(1 - p) / p
        Var(N) = r(1 - p) / p^2
    """

    def __init__(self, r: float, p: float):
        if r <= 0:
            raise ValueError("r must be positive.")
        if not (0 < p <= 1):
            raise ValueError("p must be in (0, 1].")

        self.r = r
        self.p = p

    def sample(self, size: int = 1) -> np.ndarray:
        """
        Generate random samples of claim counts.
        """
        if size <= 0:
            raise ValueError("size must be positive.")

        return np.random.negative_binomial(self.r, self.p, size=size)

    def mean(self) -> float:
        """
        E[N] = r(1 - p) / p
        """
        return self.r * (1 - self.p) / self.p

    def variance(self) -> float:
        """
        Var(N) = r(1 - p) / p^2
        """
        return self.r * (1 - self.p) / (self.p ** 2)

    def pmf(self, k: int) -> float:
        """
        Probability mass function P(N = k).
        """
        if k < 0:
            return 0.0

        return float(nbinom.pmf(k, self.r, self.p))

    def cdf(self, k: int) -> float:
        """
        Cumulative distribution function P(N <= k).
        """
        if k < 0:
            return 0.0

        return float(nbinom.cdf(k, self.r, self.p))

    def __repr__(self) -> str:
        return f"NegativeBinomial(r={self.r}, p={self.p})"