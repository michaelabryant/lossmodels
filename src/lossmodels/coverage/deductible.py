import numpy as np

from ..severity.base import SeverityModel


class OrdinaryDeductible(SeverityModel):
    """
    Severity model with an ordinary deductible applied.

    If X is the ground-up loss, the payment per loss is:

        Y = max(X - d, 0)

    Parameters
    ----------
    severity : SeverityModel
        Ground-up severity model.
    d : float
        Deductible amount, with d >= 0.
    """

    def __init__(self, severity: SeverityModel, d: float):
        if d < 0:
            raise ValueError("d must be nonnegative.")

        self.severity = severity
        self.d = d

    def sample(self, size: int = 1) -> np.ndarray:
        """
        Generate random samples of payment per loss after deductible.
        """
        if size <= 0:
            raise ValueError("size must be positive.")

        ground_up = self.severity.sample(size=size)
        return np.maximum(ground_up - self.d, 0.0)

    def mean(self) -> float:
        """
        Expected payment per loss after deductible:

            E[(X - d)+]

        Uses the underlying severity's excess_loss method if available.
        """
        return self.severity.excess_loss(self.d)

    def variance(self, n_sim: int = 100_000) -> float:
        """
        Variance of payment per loss after deductible.

        Default implementation uses simulation.
        """
        samples = self.sample(size=n_sim)
        return float(np.var(samples, ddof=0))

    def payment_probability(self) -> float:
        """
        P(X > d): probability that a payment is made.
        """
        return 1.0 - self.severity.cdf(self.d)

    def loss_elimination_ratio(self) -> float:
        """
        Loss Elimination Ratio (LER):

            LER = (E[X] - E[(X-d)+]) / E[X]
        """
        ex = self.severity.mean()
        if ex == 0:
            return 0.0

        return (ex - self.mean()) / ex

    def __repr__(self) -> str:
        return f"OrdinaryDeductible(severity={repr(self.severity)}, d={self.d})"