from __future__ import annotations

import jaxlie
import jax
import jax.numpy as np
from typing import TypeVar, Union
from .helpers import jitclass, jitmethod
import jax_dataclasses as jdc

GroupType = TypeVar("GroupType", bound=jaxlie.MatrixLieGroup)
Variable = Union[jax.Array, GroupType, "DroneState"]


# @jitclass
@jdc.pytree_dataclass
class DroneState:
    """Functions as SO(3) x R^3 x R^3"""

    q: jaxlie.SO3
    w: np.ndarray
    p: np.ndarray
    v: np.ndarray

    @staticmethod
    def make(
        q: jaxlie.SO3 = jaxlie.SO3.identity(),
        w: np.ndarray = np.zeros(3),
        p: np.ndarray = np.zeros(3),
        v: np.ndarray = np.zeros(3),
    ):
        return DroneState(q, w, p, v)

    @staticmethod
    def identity():
        return DroneState(jaxlie.SO3.identity(), np.zeros(3), np.zeros(3), np.zeros(3))

    def __matmul__(self, delta):
        return DroneState(
            self.q @ delta.q, self.w + delta.w, self.p + delta.p, self.v + delta.v
        )

    @staticmethod
    def exp(delta):
        return DroneState(
            jaxlie.SO3.exp(delta[:3]), delta[3:6], delta[6:9], delta[9:12]
        )

    @property
    def tangent_dim(self):
        return 12

    def inverse(self):
        return DroneState(self.q.inverse(), -self.w, -self.p, -self.v)

    def log(self):
        return np.concatenate([self.q.log(), self.w, self.p, self.v])

    def copy(self):
        return DroneState(
            jaxlie.SO3(self.q.wxyz), self.w.copy(), self.p.copy(), self.v.copy()
        )


def dim(val: Variable) -> int:
    if isinstance(val, Union[jaxlie.MatrixLieGroup, DroneState]):
        return val.tangent_dim
    else:
        return val.size


def add(val: Variable, delta: jax.Array) -> Variable:
    assert delta.size == dim(val), "Dimension mismatch in add"
    if isinstance(val, Union[jaxlie.MatrixLieGroup, DroneState]):
        return val @ type(val).exp(delta)
    else:
        return val + delta


def sub(val1: Variable, val2: Variable) -> Variable:
    assert type(val1) == type(val2), "Dimension mismatch in sub"
    if isinstance(val1, Union[jaxlie.MatrixLieGroup, DroneState]):
        return (val2.inverse() @ val1).log()
    else:
        return val1 - val2


def eye(val: Variable) -> Variable:
    if isinstance(val, Union[jaxlie.MatrixLieGroup, DroneState]):
        return type(val).identity(val.shape[0])
    else:
        return np.zeros(val.size)


def apply(X: Variable, v: jax.Array) -> Variable:
    if type(X) == jaxlie.SE2 or type(X) == jaxlie.SE3:
        assert dim(X) == dim(v) + 1, "Dimension mismatch in apply"
        return X.rotation().apply(v) + X.translation()

    else:
        assert dim(X) == dim(v), "Dimension mismatch in apply"
        return X.apply(v)


def copy(X: Variable) -> Variable:
    if isinstance(X, jaxlie.MatrixLieGroup):
        return type(X)(X.parameters())
    else:
        return X.copy()


@jax.jit
def stack(pytrees, axis=0):
    results = jax.tree_map(lambda *values: np.stack(values, axis=axis), *pytrees)
    return results


@jitmethod
def rk4(dynamics: callable, params: jax.Array, x: jax.Array, u: jax.Array, dt: float):
    # vanilla RK4
    k1 = dt * dynamics(params, x, u)
    k2 = dt * dynamics(params, add(x, k1 / 2), u)
    k3 = dt * dynamics(params, add(x, k2 / 2), u)
    k4 = dt * dynamics(params, add(x, k3), u)
    delta = (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return add(x, delta)
