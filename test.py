import core
import jax.numpy as np
import jax.experimental.sparse as sp
from dataclasses import dataclass
from overrides import overrides
from double_integrator import DoubleIntegratorSim

np.set_printoptions(precision=3, suppress=True, linewidth=200)


# TODO: Figure out how to handle keys in the factors class
@dataclass
class LinearSystem(core.Factor):
    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    R: np.ndarray

    @property
    @overrides
    def has_cost(self) -> bool:
        return True

    @overrides
    def cost(self, values: list[core.Variable]) -> float:
        x, _, u = values
        return x.T @ self.Q @ x + u.T @ self.R @ u

    @overrides
    def constraints(self, values: list[core.Variable]) -> np.ndarray:
        x_curr, x_next, u = values
        return x_next - self.A @ x_curr - self.B @ u

    @property
    @overrides
    def constraints_dim(self) -> int:
        return self.A.shape[0]


@dataclass
class FixConstraint(core.Factor):
    value: np.ndarray

    @overrides
    def constraints(self, values: list[core.Variable]) -> np.ndarray:
        return values[0] - self.value

    @property
    @overrides
    def constraints_dim(self) -> int:
        return self.value.shape[0]


sys = DoubleIntegratorSim(5, 0.1, np.ones(4))

graph = core.Graph(
    [
        FixConstraint(["x1"], sys.x0),
        LinearSystem(["x1", "x2", "u1"], sys.A, sys.B, 4 * np.eye(4), 2 * np.eye(2)),
    ]
)

vals = core.Variables(
    {
        "x1": np.full(4, 2.0),
        "x2": np.ones(4),
        "u1": np.ones(2),
    }
)

final = graph.solve(vals)
print(final[1])
