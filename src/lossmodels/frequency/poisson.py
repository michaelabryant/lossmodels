import numpy as np
from scipy.stats import poisson

from .base import FrequencyModel


class Poisson(FrequencyModel):
    """
    Poisson frequency model.

    Parameterization
    ----------------
    N ~ Poisson(lam)
    Support: {0, 1, 2, ...}

    Parameters
    ----------
    lam : float
        Expected claim count, with lam > 0.
    """

    def __init__(self, lam: float):
        if lam <= 0:
            raise ValueError("lam must be positive.")

        self.lam = lam

    def sample(self, size: int = 1) -> np.ndarray:
        """
        Generate random samples of claim counts.
        """
        if size <= 0:
            raise ValueError("size must be positive.")

        return np.random.poisson(self.lam, size=size)

    def mean(self) -> float:
        """
        E[N] = lam
        """
        return self.lam

    def variance(self) -> float:
        """
        Var(N) = lam
        """
        return self.lam

    def pmf(self, k: int) -> float:
        """
        Probability mass function P(N = k).
        """
        if k < 0:
            return 0.0

        return float(poisson.pmf(k, self.lam))

    def cdf(self, k: int) -> float:
        """
        Cumulative distribution function P(N <= k).
        """
        if k < 0:
            return 0.0

        return float(poisson.cdf(k, self.lam))

    def __repr__(self) -> str:
        return f"Poisson(lam={self.lam})"