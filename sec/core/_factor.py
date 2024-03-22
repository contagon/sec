import jax.numpy as np
import jax
from typing import Union, TypeVar
import jaxlie
from ._helpers import jitclass
import jax_dataclasses as jdc
from functools import cached_property

GroupType = TypeVar("GroupType", bound=jaxlie.MatrixLieGroup)
Variable = Union[jax.Array, GroupType]


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
    def constraints_dim(self) -> int:
        return 0

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
    def residual_jac(self, values: list[Variable]) -> np.ndarray:
        return jax.jacfwd(self.residual)(values)

    # Defaults to using residual cost (similar speed to using J^T r, easier to code)
    def cost_grad(self, values: list[Variable]):
        return jax.grad(self.cost)(values)

    def cost_hess(self, values: list[Variable]):
        return jax.jacrev(jax.jacfwd(self.cost))(values)

    def constraints_jac(self, values: list[Variable]):
        return jax.jacfwd(self.constraints)(values)

    def constraints_hess(self, values: list[Variable]):
        return jax.jacrev(jax.jacfwd(self.constraints))(values)
