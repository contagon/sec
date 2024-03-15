import core
import jax.numpy as np
import jax.experimental.sparse as sp
from dataclasses import dataclass
from overrides import overrides
from double_integrator import DoubleIntegratorSim
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True, linewidth=200)


def vals2state(vals: core.Variables) -> np.ndarray:
    x = []
    u = []
    for key, val in vals.vals.items():
        if "x" in key:
            x.append(val)
        elif "u" in key:
            u.append(val)
    return np.vstack(x), np.vstack(u)


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
        return x.T @ self.Q @ x / 2 + u.T @ self.R @ u / 2

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


@dataclass
class FinalCost(core.Factor):
    Qf: np.ndarray

    @property
    @overrides
    def has_cost(self) -> bool:
        return True

    @overrides
    def cost(self, values: list[core.Variable]) -> float:
        x = values[0]
        return x.T @ self.Qf @ x / 2


x0 = np.array([1, 1, -1, 1])
sys = DoubleIntegratorSim(5, 0.1, x0)
graph = core.Graph()
vals = core.Variables()

graph.add(FixConstraint(["x0"], sys.x0))
vals.add("x0", sys.x0)

for i in range(sys.N - 1):
    graph.add(
        LinearSystem(
            [f"x{i}", f"x{i+1}", f"u{i}"], sys.A, sys.B, np.eye(4), 0.1 * np.eye(2)
        )
    )
    vals.add(f"x{i+1}", np.ones(4))
    vals.add(f"u{i}", np.ones(2))

graph.add(FinalCost([f"x{sys.N-1}"], 10 * np.eye(4)))

final, info = graph.solve(vals)
x, u = vals2state(final)

t = np.linspace(0, sys.T, sys.N)
plt.plot(t, x[:, 0], label="x")
plt.plot(t, x[:, 1], label="y")
plt.plot(t, x[:, 2], label="vx")
plt.plot(t, x[:, 3], label="vy")
plt.legend()
plt.show()
