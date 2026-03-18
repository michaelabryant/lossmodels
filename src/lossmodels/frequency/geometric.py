import numpy as np
from scipy.stats import geom

from .base import FrequencyModel


class Geometric(FrequencyModel):
    """
    Geometric frequency model.

    Support starting at 0: {0, 1, 2, 3, ...}

    Parameters
    ----------
    p : float
        Probability of success

    Notes
    -----
    NumPy and SciPy define the geometric distribution on {1, 2, 3, ...}
    as the number of trials until first success. This implementation shifts
    that convention by 1 so the support starts at 0, which is more natural
    for claim counts.
    """

    def __init__(self, p: float):
        if not (0 <= p <= 1):
            raise ValueError("p must be between 0 and 1")

        self.p = p

    def sample(self, size: int = 1) -> np.ndarray:
        return np.random.geometric(self.p, size=size) - 1

    def mean(self) -> float:
        return 1 / self.p

    def variance(self) -> float:
        return (1 - self.p) / (self.p ** 2)

    def pmf(self, k: int) -> float:
        """
        Probability mass function P(N = k)
        """
        if k < 0:
            return 0.0
        
        return geom.pmf(k + 1, self.p)
    
    def cdf(self, k: int) -> float:
        """
        Cumulative distribution function P(N <= k)
        """
        if k < 0:
            return 0.0
        
        return geom.cdf(k + 1, self.p)

    def __repr__(self):
        return f"Geometric(p={self.p})"