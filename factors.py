from jax.numpy import ndarray
import core
from overrides import overrides
import jax.numpy as np
import jax_dataclasses as jdc
from dataclasses import dataclass
from helpers import jitclass


@jitclass
@jdc.pytree_dataclass
class LinearSystem(core.Factor):
    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    R: np.ndarray

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
    value: np.ndarray

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
    Qf: np.ndarray

    @overrides
    def cost(self, values: list[core.Variable]) -> float:
        x = values[0]
        return x.T @ self.Qf @ x / 2


# ------------------------- Getting state estimation working ------------------------- #
# TODO: Test these
@jitclass
@jdc.pytree_dataclass
class PriorFactor(core.Factor):
    mu: np.ndarray
    sigma: np.ndarray

    @overrides
    def cost(self, values: list[core.Variable]) -> float:
        x = values[0]
        return (x - self.mu).T @ np.linalg.inv(self.sigma) @ (x - self.mu) / 2


@jitclass
@jdc.pytree_dataclass
class LandmarkMeasure(core.Factor):
    mm: np.ndarray
    cov: np.ndarray

    @overrides
    def cost(self, values: list[core.Variable]) -> float:
        x, l = values
        r = (l - x[:2]) - self.mm
        return r.T @ np.linalg.inv(self.cov) @ r / 2


@jitclass
@jdc.pytree_dataclass
class PastDynamics(core.Factor):
    dyn: jdc.Static[callable]
    u_mm: np.ndarray
    cov: np.ndarray

    @overrides
    def cost(self, values: list[core.Variable]) -> float:
        params, x, x_next = values
        r = x_next - self.dyn(params, x, self.u_mm)
        return r.T @ np.linalg.inv(self.cov) @ r / 2
