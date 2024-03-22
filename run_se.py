import sec.core as core
import factors
from double_integrator import DoubleIntegratorSim
import jax.numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import jax
import sec.symbols as sym


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


# ------------------------- Run the simulation ------------------------- #
sys = DoubleIntegratorSim(5, 0.1)
gt = core.Variables()

key = sys.getkey()
# Make u a little less boring
us = (
    np.tile(np.array([1.0, 0.05]), (sys.N, 1))
    + jax.random.normal(key, (sys.N, 2)) * 0.1
)
x = sys.x0
gt.add(sym.X(0), x)
gt.add(sym.P(0), sys.params)

meas = []
for i in range(sys.N):
    x = sys.step(x, us[i])
    z = sys.measure(x)

    meas.append(z)
    gt.add(sym.X(i + 1), x)

for i, l in enumerate(sys.landmarks):
    gt.add(sym.L(i), l)

# ------------------------- Make the graph & initial estimates ------------------------- #
graph = core.Graph()
init = core.Variables()

# priors for inits
x = sys.x0
p = np.full(4, 1.2)
init.add(sym.X(0), x)
init.add(sym.P(0), p)
graph.add(factors.PriorFactor([sym.X(0)], sys.x0, np.eye(4) * 1e-2))
graph.add(factors.PriorFactor([sym.P(0)], p, np.eye(4) * 1))

# landmark estimates
for i, l in enumerate(sys.landmarks):
    key = sys.getkey()
    est = l + jax.random.normal(key, (2,), dtype=np.float32)
    init.add(sym.L(i), est)

# trajectory estimates
for i in range(sys.N):
    graph.add(
        factors.PastDynamics(
            [sym.P(0), sym.X(i), sym.X(i + 1)],
            sys.dynamics,
            us[i],
            np.eye(4) * sys.std_Q**2,
        )
    )
    x = sys.dynamics(p, x, us[i])
    init.add(sym.X(i + 1), x)
    for idx, z in enumerate(meas[i]):
        graph.add(
            factors.LandmarkMeasure(
                [sym.X(i + 1), sym.L(idx)], z, np.eye(2) * sys.std_R**2
            )
        )

# Make sure they're indexed properly
init.reindex()
gt.reindex()

final, info = graph.solve(init, verbose=True)
x_est, _, l_est = vals2state(final)
x_gt, _, l_gt = vals2state(gt)
x_init, _, l_init = vals2state(init)

print("p_init", init[sym.P(0)])
print("p_est", final[sym.P(0)])
print("p_gt", gt[sym.P(0)])

t = np.linspace(0, sys.T, sys.N + 1)
fig, ax = plt.subplots(1, 1)
ax.plot(x_est[:, 0], x_est[:, 1], label="est")
ax.plot(x_gt[:, 0], x_gt[:, 1], label="gt")
ax.plot(x_init[:, 0], x_init[:, 1], label="init")

ax.scatter(l_est[:, 0], l_est[:, 1], label="est")
ax.scatter(l_gt[:, 0], l_gt[:, 1], label="gt")
ax.scatter(l_init[:, 0], l_init[:, 1], label="init")

ax.legend()
plt.show()
