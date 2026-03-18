import numpy as np

from ..severity.base import SeverityModel


class PolicyLimit(SeverityModel):
    """
    Severity model with a policy limit applied.

    If X is the ground-up loss, the payment per loss is:

        Y = min(X, u)

    Parameters
    ----------
    severity : SeverityModel
        Ground-up severity model.
    u : float
        Policy limit, with u >= 0.
    """

    def __init__(self, severity: SeverityModel, u: float):
        if u < 0:
            raise ValueError("u must be nonnegative.")

        self.severity = severity
        self.u = u

    def sample(self, size: int = 1) -> np.ndarray:
        """
        Generate random samples of payment per loss after policy limit.
        """
        if size <= 0:
            raise ValueError("size must be positive.")

        ground_up = self.severity.sample(size=size)
        return np.minimum(ground_up, self.u)

    def mean(self) -> float:
        """
        Expected payment per loss after policy limit:

            E[min(X, u)]

        Uses the underlying severity's limited_expected_value method.
        """
        return self.severity.limited_expected_value(self.u)

    def variance(self, n_sim: int = 100_000) -> float:
        """
        Variance of payment per loss after policy limit.

        Default implementation uses simulation.
        """
        samples = self.sample(size=n_sim)
        return float(np.var(samples, ddof=0))

    def probability_capped(self) -> float:
        """
        P(X > u): probability that the limit is binding.
        """
        return 1.0 - self.severity.cdf(self.u)

    def loss_elimination_ratio(self) -> float:
        """
        Loss Elimination Ratio (LER):

            LER = (E[X] - E[min(X,u)]) / E[X]
        """
        ex = self.severity.mean()
        if ex == 0:
            return 0.0

        return (ex - self.mean()) / ex

    def __repr__(self) -> str:
        return f"PolicyLimit(severity={repr(self.severity)}, u={self.u})"