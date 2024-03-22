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
        return np.zeros(4), np.zeros(4)


# @jitclass
@jdc.pytree_dataclass
class FinalCost(core.Factor):
    xg: np.ndarray
    Qf: np.ndarray

    @overrides
    def cost(self, values: list[core.Variable]) -> float:
        x_diff = values[0] - self.xg
        return x_diff.T @ self.Qf @ x_diff / 2


@jitclass
@jdc.pytree_dataclass
class LandmarkAvoid(core.Factor):
    dist: float = 0.5

    @overrides
    def constraints(self, values: list[core.Variable]) -> np.ndarray:
        x, l = values
        d = x[:2] - l
        return np.array([d.T @ d])

    @property
    @overrides
    def constraints_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.full(1, self.dist**2), np.inf * np.ones(1)


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
class LandmarkMeasure(core.Factor):
    mm: np.ndarray
    W: np.ndarray

    @overrides
    def residual(self, values: list[core.Variable]) -> np.ndarray:
        x, l = values
        return (l - x[:2]) - self.mm

    @property
    def residual_dim(self) -> int:
        return self.mm.shape[0]


@jitclass
@jdc.pytree_dataclass
class PastDynamics(core.Factor):
    dyn: jdc.Static[callable]
    u_mm: np.ndarray
    W: np.ndarray

    @overrides
    def residual(self, values: list[core.Variable]) -> np.ndarray:
        params, x, x_next = values
        return x_next - self.dyn(params, x, self.u_mm)

    # TODO: WRONG
    @property
    def residual_dim(self) -> int:
        return self.u_mm.shape[0]
