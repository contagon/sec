from .. import core
from ..core import jitclass

from overrides import overrides
import jax.numpy as np
import jax_dataclasses as jdc


# ------------------------- Controls Factors ------------------------- #
@jitclass
@jdc.pytree_dataclass
class FixConstraint(core.Factor):
    value: np.ndarray

    @overrides
    def constraints(self, values: list[core.Variable]) -> np.ndarray:
        return values[0] - self.value

    @property
    @overrides
    def constraints_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        d = self.value.shape[0]
        return np.zeros(d), np.zeros(d)


@jitclass
@jdc.pytree_dataclass
class BoundedConstraint(core.Factor):
    lb: np.ndarray
    ub: np.ndarray

    @overrides
    def constraints(self, values: list[core.Variable]) -> np.ndarray:
        return values[0]

    @property
    @overrides
    def constraints_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.lb, self.ub


@jitclass
@jdc.pytree_dataclass
class System(core.Factor):
    dyn: jdc.Static[callable]
    xg: np.ndarray
    Q: np.ndarray
    R: np.ndarray

    @overrides
    def cost(self, values: list[core.Variable]) -> float:
        _, x, _, u = values
        x_diff = x - self.xg
        return x_diff.T @ self.Q @ x_diff / 2 + u.T @ self.R @ u / 2

    @overrides
    def constraints(self, values: list[core.Variable]) -> np.ndarray:
        p, x_curr, x_next, u = values
        return x_next - self.dyn(p, x_curr, u)

    @property
    @overrides
    def constraints_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(2), np.zeros(2)

    @overrides
    def constraints_jac(self, values: list[core.Variable]):
        jac = super().constraints_jac(values)
        jac[0] = np.zeros((2, 2))
        return jac


# @jitclass
@jdc.pytree_dataclass
class FinalCost(core.Factor):
    xg: np.ndarray
    Qf: np.ndarray

    @overrides
    def cost(self, values: list[core.Variable]) -> float:
        x_diff = values[0] - self.xg
        return x_diff.T @ self.Qf @ x_diff / 2


# ------------------------- State Estimation Factors ------------------------- #
@jitclass
@jdc.pytree_dataclass
class PriorFactor(core.Factor):
    mu: np.ndarray
    W: np.ndarray

    @overrides
    def residual(self, values: list[core.Variable]) -> np.ndarray:
        x = values[0]
        return x - self.mu

    @property
    def residual_dim(self) -> int:
        return self.mu.shape[0]


@jitclass
@jdc.pytree_dataclass
class EncoderMeasure(core.Factor):
    mm: np.ndarray
    W: np.ndarray

    @overrides
    def residual(self, values: list[core.Variable]) -> np.ndarray:
        x = values[0]
        return x[0] - self.mm

    @property
    def residual_dim(self) -> int:
        return self.mm.shape[0]


@jitclass
@jdc.pytree_dataclass
class PastDynamics(core.Factor):
    dyn: jdc.Static[callable]
    W: np.ndarray

    @overrides
    def residual(self, values: list[core.Variable]) -> np.ndarray:
        params, x, x_next, u, w = values
        return w

    @overrides
    def constraints(self, values: list[core.Variable]) -> np.ndarray:
        params, x, x_next, u, w = values
        return x_next - self.dyn(params, x, u) + w

    @property
    @overrides
    def constraints_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(2), np.zeros(2)
