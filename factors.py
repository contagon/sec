import core
from overrides import overrides
import jax.numpy as np


class LinearSystem(core.Factor):
    def __init__(self, keys, A, B, Q, R):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        super().__init__(keys)

    @overrides
    def cost(self, values: list[core.Variable]) -> float:
        x, _, u = values
        return x.T @ self.Q @ x / 2 + u.T @ self.R @ u / 2

    @overrides
    def constraints(self, values: list[core.Variable]) -> np.ndarray:
        x_curr, x_next, u = values
        return x_next - self.A @ x_curr - self.B @ u

    @property
    @overrides
    def constraints_dim(self) -> int:
        return self.A.shape[0]


class FixConstraint(core.Factor):
    def __init__(self, keys: list[str], value):
        self.value = value
        super().__init__(keys)

    @overrides
    def constraints(self, values: list[core.Variable]) -> np.ndarray:
        return values[0] - self.value

    @property
    @overrides
    def constraints_dim(self) -> int:
        return self.value.shape[0]


class FinalCost(core.Factor):
    def __init__(self, keys: list[str], Qf):
        self.Qf = Qf
        super().__init__(keys)

    @overrides
    def cost(self, values: list[core.Variable]) -> float:
        x = values[0]
        return x.T @ self.Qf @ x / 2


# TODO: Test these
class PriorFactor(core.Factor):
    def __init__(self, keys: list[str], mu, sigma):
        self.mu = mu
        self.sigma = sigma
        super().__init__(keys)

    @overrides
    def cost(self, values: list[core.Variable]) -> float:
        x = values[0]
        return (x - self.mu).T @ np.linalg.inv(self.sigma) @ (x - self.mu) / 2


class ProbLinearSystem(core.Factor):
    def __init__(self, keys, A, B, u, sigma):
        self.A = A
        self.B = B
        self.u = u
        self.sigma = sigma
        super().__init__(keys)

    @overrides
    def cost(self, values: list[core.Variable]) -> float:
        x_curr, x_next = values
        x_next_est = self.A @ x_curr + self.B @ self.u
        return (
            (x_next - x_next_est).T
            @ np.linalg.inv(self.sigma)
            @ (x_next - x_next_est)
            / 2
        )
