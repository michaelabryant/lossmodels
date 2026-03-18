import numpy as np
from scipy.stats import geom

from .base import FrequencyModel


class Geometric(FrequencyModel):
    """
    Geometric frequency model.

    Parameters
    ----------
    p : float
        Probability of success
    """

    def __init__(self, p: float):
        if not (0 <= p <= 1):
            raise ValueError("p must be between 0 and 1")

        self.p = p

    def sample(self, size: int = 1) -> np.ndarray:
        return np.random.geometric(self.p, size=size)

    def mean(self) -> float:
        return 1 / self.p

    def variance(self) -> float:
        return (1 - self.p) / self.p**2

    def pmf(self, k: int) -> float:
        """
        Probability mass function P(N = k)
        """
        return geom.pmf(k, self.p)

    def __repr__(self):
        return f"Geometric(p={self.p})"