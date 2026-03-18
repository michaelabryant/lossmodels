import numpy as np

from ..severity.base import SeverityModel


class Layer(SeverityModel):
    """
    Severity model with a deductible and a payment limit applied.

    If X is the ground-up loss, the payment per loss is:

        Y = min((X - d)+, u)

    where:
    - d is the deductible / attachment point
    - u is the maximum payment in the layer

    Parameters
    ----------
    severity : SeverityModel
        Ground-up severity model.
    d : float
        Deductible / attachment point, with d >= 0.
    u : float
        Layer width (maximum payment), with u >= 0.
    """

    def __init__(self, severity: SeverityModel, d: float, u: float):
        if d < 0:
            raise ValueError("d must be nonnegative.")
        if u < 0:
            raise ValueError("u must be nonnegative.")

        self.severity = severity
        self.d = d
        self.u = u

    def sample(self, size: int = 1) -> np.ndarray:
        """
        Generate random samples of payment per loss in the layer.
        """
        if size <= 0:
            raise ValueError("size must be positive.")

        ground_up = self.severity.sample(size=size)
        return np.minimum(np.maximum(ground_up - self.d, 0.0), self.u)

    def mean(self) -> float:
        """
        Expected payment per loss in the layer:

            E[min((X-d)+, u)]

        This can be written as:

            E[(X-d)+] - E[(X-d-u)+]
        """
        return self.severity.excess_loss(self.d) - self.severity.excess_loss(self.d + self.u)

    def variance(self, n_sim: int = 100_000) -> float:
        """
        Variance of payment per loss in the layer.

        Default implementation uses simulation.
        """
        samples = self.sample(size=n_sim)
        return float(np.var(samples, ddof=0))

    def payment_probability(self) -> float:
        """
        Probability that the layer pays anything:

            P(X > d)
        """
        return 1.0 - self.severity.cdf(self.d)

    def exhaustion_probability(self) -> float:
        """
        Probability that the layer is fully exhausted:

            P(X > d + u)
        """
        return 1.0 - self.severity.cdf(self.d + self.u)

    def __repr__(self) -> str:
        return f"Layer(severity={repr(self.severity)}, d={self.d}, u={self.u})"