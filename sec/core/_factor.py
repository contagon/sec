import jax.numpy as np
import jax
from typing import Optional
import jaxlie
from ..helpers import jitclass
from ._variables import Variables, Variable
import jax_dataclasses as jdc
from functools import cached_property
import sec.operators as op
from ..helpers import jacfwd, jacrev, grad


@jitclass
@jdc.pytree_dataclass
class Factor:
    keys: list[str]

    # ------------------------- Potential Overrides ------------------------- #
    # Nonlinear least-squares
    @property
    def residual_dim(self):
        return 0

    def residual(self, values: list[Variable]) -> np.ndarray:
        pass

    # Generic cost function instead
    def cost(self, values: list[Variable]) -> float:
        r = self.residual(values)
        return r @ self.Winv @ r / 2

    # Constraints
    @property
    def constraints_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(0), np.zeros(0)

    def constraints(self, values: list[Variable]) -> np.ndarray:
        return None

    # ------------------------- Helpers ------------------------- #
    @property
    def has_cost(self):
        return self.has_res or (type(self).cost != Factor.cost)

    @property
    def has_con(self):
        return type(self).constraints != Factor.constraints

    @property
    def has_res(self):
        return type(self).residual != Factor.residual

    @cached_property
    def Winv(self):
        return np.linalg.inv(self.W)

    # ------------------------- All autodiff jacobians & such ------------------------- #
    # TODO: Use manifold diff operators!
    def residual_jac(self, values: list[Variable]) -> np.ndarray:
        return jax.jacfwd(self.residual)(values)

    @property
    def constraints_dim(self) -> int:
        return self.constraints_bounds[0].size

    # Defaults to using residual cost (similar speed to using J^T r, easier to code)
    def cost_grad(self, values: list[Variable], delta: Optional[list[Variable]] = None):
        if delta is not None:
            return grad(wrapper(self.cost))(delta, values)
        else:
            return grad(self.cost)(values)

    def constraints_jac(
        self, values: list[Variable], delta: Optional[list[Variable]] = None
    ):
        if delta is not None:
            return jacfwd(wrapper(self.constraints))(delta, values)
        else:
            return jacfwd(self.constraints)(values)


def wrapper(func):
    def wrapped(delta, values):
        together = [op.add(val, d) for val, d in zip(values, delta)]
        return func(together)

    return wrapped
