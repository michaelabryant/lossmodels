import numpy as np
from scipy.stats import lognorm

from .base import SeverityModel


class Lognormal(SeverityModel):
    """
    Lognormal severity model.

    Parameterization
    ----------------
    If Y = log(X) ~ Normal(mu, sigma^2), then X is Lognormal(mu, sigma).
    Support: x > 0

    Parameters
    ----------
    mu : float
        Mean of log(X).
    sigma : float
        Standard deviation of log(X), with sigma > 0.
    """

    def __init__(self, mu: float, sigma: float):
        if sigma <= 0:
            raise ValueError("sigma must be positive.")

        self.mu = mu
        self.sigma = sigma

    def sample(self, size: int = 1) -> np.ndarray:
        if size <= 0:
            raise ValueError("size must be positive.")

        return np.random.lognormal(mean=self.mu, sigma=self.sigma, size=size)

    def mean(self) -> float:
        return float(np.exp(self.mu + 0.5 * self.sigma ** 2))

    def variance(self) -> float:
        sigma2 = self.sigma ** 2
        return float((np.exp(sigma2) - 1) * np.exp(2 * self.mu + sigma2))

    def pdf(self, x: float) -> float:
        if x <= 0:
            return 0.0
        return float(lognorm.pdf(x, s=self.sigma, scale=np.exp(self.mu)))

    def cdf(self, x: float) -> float:
        if x <= 0:
            return 0.0
        return float(lognorm.cdf(x, s=self.sigma, scale=np.exp(self.mu)))

    def __repr__(self) -> str:
        return f"Lognormal(mu={self.mu}, sigma={self.sigma})"