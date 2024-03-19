import core
import jax.numpy as np
import jax.experimental.sparse as sp
from dataclasses import dataclass
from overrides import overrides
from double_integrator import DoubleIntegratorSim
import matplotlib.pyplot as plt
from helpers import jitclass
import jax
from functools import partial
from factors import *

np.set_printoptions(precision=3, suppress=True, linewidth=200)
jax.config.update("jax_platforms", "cpu")


def vals2state(vals: core.Variables) -> np.ndarray:
    x = []
    u = []
    for key, val in vals.vals.items():
        if "x" in key:
            x.append(val)
        elif "u" in key:
            u.append(val)
    return np.vstack(x), np.vstack(u)


x0 = np.array([1.0, 1, -1, 1])
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
    vals.add(f"x{i+1}", np.ones(4, dtype=np.float32))
    vals.add(f"u{i}", np.ones(2, dtype=np.float32))

graph.add(FinalCost([f"x{sys.N-1}"], 10 * np.eye(4)))

# graph.template = vals

final, info = graph.solve(vals)
x, u = vals2state(final)

# t = np.linspace(0, sys.T, sys.N)
# plt.plot(t, x[:, 0], label="x")
# plt.plot(t, x[:, 1], label="y")
# plt.plot(t, x[:, 2], label="vx")
# plt.plot(t, x[:, 3], label="vy")
# plt.legend()
# plt.show()

# f = FinalCost(["x0"], 0.1 * np.eye(4))
# f.cost(vals[f.keys])
# print()
# print(f.cost_grad(vals[f.keys]))
# # print(f.constraints_jac([np.ones(4)]))
