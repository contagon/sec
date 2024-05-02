from .. import core
from sec.helpers import jitclass, wrap2pi

from overrides import overrides
import jax.numpy as np
import jax_dataclasses as jdc
from typing import Optional
from sec.core import Variable


@jitclass
@jdc.pytree_dataclass
class LandmarkAvoid(core.Factor):
    dist: float = 0.5

    @overrides
    def constraints(self, values: list[core.Variable]) -> np.ndarray:
        # TODO: Use tangent line for distance from circle?
        x, l = values
        d = x[1:] - l
        return np.array([d.T @ d])

    @property
    @overrides
    def constraints_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.full(1, self.dist**2), np.inf * np.ones(1)


@jitclass
@jdc.pytree_dataclass
class LandmarkMeasure(core.Factor):
    mm: np.ndarray
    W: np.ndarray

    @overrides
    def residual(self, values: list[core.Variable]) -> np.ndarray:
        x, l = values

        theta, px, py = x
        lx, ly = l

        angle = wrap2pi(np.arctan2(ly - py, lx - px) - theta)
        r = np.sqrt((lx - px) ** 2 + (ly - py) ** 2)

        return np.array([r, angle]) - self.mm

    @property
    def residual_dim(self) -> int:
        return self.mm.shape[0]
