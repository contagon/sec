import sec.core as core
import sec.symbols as sym
from sec.dint import (
    DoubleIntegratorSim,
    FixConstraint,
    System,
    LandmarkAvoid,
    FinalCost,
)
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

jax.config.update("jax_platforms", "cpu")
np.set_printoptions(precision=3, suppress=True)


def vals2state(vals: core.Variables) -> np.ndarray:
    x = []
    u = []
    l = []
    for key, val in vals.vals.items():
        if "X" in sym.str(key):
            x.append(val)
        elif "U" in sym.str(key):
            u.append(val)
        elif "L" in sym.str(key):
            l.append(val)

    if len(x) == 0:
        x = None
    else:
        x = np.vstack(x)

    if len(u) == 0:
        u = None
    else:
        u = np.vstack(u)

    if len(l) == 0:
        l = None
    else:
        l = np.vstack(l)

    return x, u, l


# Set up the simulation
xg = np.array([10.0, 0, 0, 0])
sys = DoubleIntegratorSim(5, 0.1, num_landmarks=10, dist=0.7)
graph = core.Graph()
vals = core.Variables()

# Setup trajectory opt
graph.add(FixConstraint([sym.X(0)], sys.x0))
vals.add(sym.X(0), sys.x0)
graph.add(FixConstraint([sym.P(0)], sys.params))
vals.add(sym.P(0), sys.params)

Q = np.zeros((4, 4))
Q = Q.at[:2, :2].set(np.eye(2))
x = sys.x0.copy()
for i in range(sys.N):
    graph.add(
        System(
            [sym.P(0), sym.X(i), sym.X(i + 1), sym.U(i)],
            sys.dynamics,
            xg,
            np.eye(4),
            0.1 * np.eye(2),
        )
    )
    key = sys.getkey()
    u = jax.random.normal(key, (2,)) * 0.001
    x = sys.dynamics(sys.params, x, u)
    vals.add(sym.X(i + 1), x)
    vals.add(sym.U(i), u)

    for idx, l in enumerate(sys.landmarks):
        graph.add(LandmarkAvoid([sym.X(i), sym.L(idx)], sys.dist))

for i, l in enumerate(sys.landmarks):
    vals.add(sym.L(i), l)
    graph.add(FixConstraint([sym.L(i)], l))

graph.add(FinalCost([sym.X(sys.N)], xg, 10 * np.eye(4)))

sol, _ = graph.solve(vals)
xsim, usim, lsim = vals2state(sol)

t = np.linspace(0, sys.T, sys.N + 1)
fig, ax = plt.subplots(1, 1)
ax = [ax]
ax[0].plot(xsim[:, 0], xsim[:, 1], label="sim", marker="o")
ax[0].plot(sys.landmarks[:, 0], sys.landmarks[:, 1], "ro", label="landmarks")
for l in sys.landmarks:
    ax[0].add_artist(plt.Circle(l, sys.dist, fill=False, color="r"))
ax[0].set_aspect("equal")
plt.show()
