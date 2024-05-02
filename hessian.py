import jax
import jax.numpy as np
from jax.numpy import ndarray
from sec.symbols import X, U, P, L
from sec.pend import System, PendulumSim
from sec.core import Variables, Graph, Factor, Variable
from jax.experimental.sparse import BCOO
import jax_dataclasses as jdc
import jaxlie
from overrides import overrides
import sec.operators as op

jax.config.update("jax_enable_x64", True)

sys = PendulumSim(5, 0.1)


@jdc.pytree_dataclass
class FakeFactor(Factor):
    @overrides
    def cost(self, values: list[Variable]) -> float:
        A = np.arange(36).reshape((6, 6))
        A = A + A.T
        x1, x2, x3 = values
        x = np.concatenate([x1, x2, x3])
        return x.T @ A @ x

    @overrides
    def constraints(self, values: list[Variable]) -> jax.Array:
        x1, x2, x3 = values
        return np.concatenate([x1**2, x2**2])

    @property
    @overrides
    def constraints_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.full(4, -2), np.full(4, 2)


@jdc.pytree_dataclass
class FakeLieFactor(Factor):
    @overrides
    def cost(self, values: list[Variable]) -> float:
        x1 = values[0]
        log = x1.log()
        return -log.T @ log

    @overrides
    def constraints(self, values: list[Variable]) -> jax.Array:
        return values[0].log() ** 2

    @property
    @overrides
    def constraints_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.full(3, -2), np.full(3, 2)


# ------------------------- Vector Version Works ------------------------- #
# f = FakeFactor([X(0), X(1), X(2)])
# values = [np.full(2, 0.1), np.full(2, 0.1), np.full(2, 0.1)]
# delta = [np.full(2, 0.1), np.full(2, 0.1), np.full(2, 0.1)]
# together = [op.add(val, d) for val, d in zip(values, delta)]

# print("Cost")
# print(f.cost_grad(together))
# print(f.cost_grad(values, delta))
# print("Constraint")
# print(f.constraints_jac(together))
# print(f.constraints_jac(values, delta))
# print()

# ------------------------- Lie Version ------------------------- #
f = FakeLieFactor([X(0)])
values = [jaxlie.SO3.from_rpy_radians(*np.full(3, 0.1))]
delta = [np.full(3, 0.5)]
together = [op.add(val, d) for val, d in zip(values, delta)]

print("Cost")
print(f.cost_grad(together))
print(f.cost_grad(values, delta))
print("Constraint")
print(f.constraints_jac(together))
print(f.constraints_jac(values, delta))
