from .. import core
from sec.core import jitclass, wrap2pi

from overrides import overrides
import jax.numpy as np
import jax_dataclasses as jdc
from typing import Optional
from sec.core import Variable
import sec.operators as op


# ------------------------- Controls Factors ------------------------- #
@jitclass
@jdc.pytree_dataclass
class FixConstraint(core.Factor):
    value: np.ndarray

    @overrides
    def constraints(self, values: list[core.Variable]) -> np.ndarray:
        return op.sub(values[0], self.value)

    @property
    @overrides
    def constraints_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        d = op.dim(self.value)
        return np.zeros(d), np.zeros(d)


@jitclass
@jdc.pytree_dataclass
class BoundedConstraint(core.Factor):
    lb: np.ndarray
    ub: np.ndarray

    @overrides
    def constraints(self, values: list[core.Variable]) -> np.ndarray:
        return op.sub(values[0], op.eye(values[0]))

    @property
    @overrides
    def constraints_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.lb, self.ub


@jitclass
@jdc.pytree_dataclass
class BoundedConstraintLambda(core.Factor):
    lam: jdc.Static[callable]
    lb: np.ndarray
    ub: np.ndarray

    @overrides
    def constraints(self, values: list[core.Variable]) -> np.ndarray:
        return self.lam(values)

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
        x_diff = op.sub(x, self.xg)
        return x_diff.T @ self.Q @ x_diff / 2 + u.T @ self.R @ u / 2

    @overrides
    def constraints(self, values: list[core.Variable]) -> np.ndarray:
        p, x_curr, x_next, u = values
        return op.sub(x_next, self.dyn(p, x_curr, u))

    @property
    @overrides
    def constraints_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(3), np.zeros(3)

    @overrides
    def constraints_jac(
        self, values: list[core.Variable], delta: Optional[list[Variable]] = None
    ):
        jac = super().constraints_jac(values, delta)
        jac[0] = np.zeros((3, 2))
        return jac


# @jitclass
@jdc.pytree_dataclass
class FinalCost(core.Factor):
    xg: np.ndarray
    Qf: np.ndarray

    @overrides
    def cost(self, values: list[core.Variable]) -> float:
        x_diff = op.sub(values[0], self.xg)
        return x_diff.T @ self.Qf @ x_diff / 2


@jitclass
@jdc.pytree_dataclass
class LandmarkAvoid(core.Factor):
    dist: float = 0.5

    @overrides
    def constraints(self, values: list[core.Variable]) -> np.ndarray:
        # TODO: Use tangent line for distance from circle?
        x, l = values
        d = x.translation() - l
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
        return op.sub(x, self.mu)

    @property
    def residual_dim(self) -> int:
        return self.mu.shape[0]


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
        return op.sub(x_next, self.dyn(params, x, u)) + w

    @property
    @overrides
    def constraints_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(3), np.zeros(3)


@jitclass
@jdc.pytree_dataclass
class LandmarkMeasure(core.Factor):
    mm: np.ndarray
    W: np.ndarray

    @overrides
    def residual(self, values: list[core.Variable]) -> np.ndarray:
        x, l = values

        px, py = x.translation()
        theta = x.rotation().as_radians()
        lx, ly = l

        angle = wrap2pi(np.arctan2(ly - py, lx - px) - theta)
        r = np.sqrt((lx - px) ** 2 + (ly - py) ** 2)

        return np.array([r, angle]) - self.mm

    @property
    def residual_dim(self) -> int:
        return self.mm.shape[0]
