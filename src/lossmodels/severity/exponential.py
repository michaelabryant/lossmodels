import numpy as np
from scipy.stats import expon

from .base import SeverityModel


class Exponential(SeverityModel):
    """
    Exponential severity model.

    Parameterization
    ----------------
    X ~ Exponential(rate)

    Support: x >= 0

    Mean = 1 / rate
    Variance = 1 / rate^2

    Parameters
    ----------
    rate : float
        Rate parameter (lambda), with rate > 0.
    """

    def __init__(self, rate: float):
        if rate <= 0:
            raise ValueError("rate must be positive.")

        self.rate = rate
        self.scale = 1.0 / rate  # SciPy uses scale = 1 / rate

    def sample(self, size: int = 1) -> np.ndarray:
        """
        Generate random samples.
        """
        if size <= 0:
            raise ValueError("size must be positive.")

        return np.random.exponential(scale=self.scale, size=size)

    def mean(self) -> float:
        return 1.0 / self.rate

    def variance(self) -> float:
        return 1.0 / (self.rate ** 2)

    def pdf(self, x: float) -> float:
        if x < 0:
            return 0.0
        return float(expon.pdf(x, scale=self.scale))

    def cdf(self, x: float) -> float:
        if x < 0:
            return 0.0
        return float(expon.cdf(x, scale=self.scale))

    def excess_loss(self, d: float) -> float:
        """
        E[(X - d)+] = exp(-rate * d) / rate
        """
        if d < 0:
            raise ValueError("d must be nonnegative.")

        return float(np.exp(-self.rate * d) / self.rate)

    def limited_expected_value(self, d: float) -> float:
        """
        E[min(X, d)] = (1 - exp(-rate * d)) / rate
        """
        if d < 0:
            raise ValueError("d must be nonnegative.")

        return float((1.0 - np.exp(-self.rate * d)) / self.rate)

    def __repr__(self) -> str:
        return f"Exponential(rate={self.rate})"