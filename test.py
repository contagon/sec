import core
import jax.numpy as np
from dataclasses import dataclass
from overrides import overrides
from double_integrator import DoubleIntegratorSim


# TODO: Figure out how to handle keys in the factors class
@dataclass
class LinearSystem(core.Factor):
    A: np.ndarray
    B: np.ndarray

    @overrides
    def constraints(self, values: core.Values) -> np.ndarray:
        x_curr = values[self.keys[0]].val
        x_next = values[self.keys[1]].val
        u = values[self.keys[2]].val
        return x_next - self.A @ x_curr - self.B @ u

    @property
    @overrides
    def constraints_dim(self) -> int:
        return self.A.shape[0]


@dataclass
class FixConstraint(core.Factor):
    value: np.ndarray

    @overrides
    def constraints(self, values: core.Values) -> np.ndarray:
        return values[self.keys[0]] - self.value

    @property
    @overrides
    def constraints_dim(self) -> int:
        return self.value.shape[0]


sys = DoubleIntegratorSim(5, 0.1, np.zeros(4))

graph = core.Graph(
    [
        FixConstraint("x0", sys.x0),
        LinearSystem(["x1", "x2", "u1"], sys.A, sys.B),
    ]
)

fac = LinearSystem(["x1", "x2", "u1"], sys.A, sys.B)
val = core.Values(
    {
        "x1": core.Value(np.zeros(4)),
        "x2": core.Value(np.ones(4)),
        "u1": core.Value(np.ones(2)),
    }
)
print(fac.constraints_jac(val))
