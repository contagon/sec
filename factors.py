import core
from overrides import overrides
import jax.numpy as np
import jax_dataclasses as jdc
from dataclasses import dataclass
from helpers import jitclass


@jitclass
@jdc.pytree_dataclass
class LinearSystem(core.Factor):
    A: jdc.Static[np.ndarray]
    B: jdc.Static[np.ndarray]
    Q: jdc.Static[np.ndarray]
    R: jdc.Static[np.ndarray]

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


@jitclass
@jdc.pytree_dataclass
class FixConstraint(core.Factor):
    value: jdc.Static[np.ndarray]

    @overrides
    def constraints(self, values: list[core.Variable]) -> np.ndarray:
        return values[0] - self.value

    @property
    @overrides
    def constraints_dim(self) -> int:
        return self.value.shape[0]


@jitclass
@jdc.pytree_dataclass
class FinalCost(core.Factor):
    Qf: jdc.Static[np.ndarray]

    @overrides
    def cost(self, values: list[core.Variable]) -> float:
        x = values[0]
        return x.T @ self.Qf @ x / 2


# TODO: Test these
@jitclass
@jdc.pytree_dataclass
class PriorFactor(core.Factor):
    mu: jdc.Static[np.ndarray]
    sigma: jdc.Static[np.ndarray]

    @overrides
    def cost(self, values: list[core.Variable]) -> float:
        x = values[0]
        return (x - self.mu).T @ np.linalg.inv(self.sigma) @ (x - self.mu) / 2


# class ProbLinearSystem(core.Factor):
#     def __init__(self, keys, A, B, u, sigma):
#         self.A = A
#         self.B = B
#         self.u = u
#         self.sigma = sigma
#         super().__init__(keys)

#     @overrides
#     def cost(self, values: list[core.Variable]) -> float:
#         x_curr, x_next = values
#         x_next_est = self.A @ x_curr + self.B @ self.u
#         return (
#             (x_next - x_next_est).T
#             @ np.linalg.inv(self.sigma)
#             @ (x_next - x_next_est)
#             / 2
#         )
