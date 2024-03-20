import core
import factors
from double_integrator import DoubleIntegratorSim
import jax.numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import jax


def vals2state(vals: core.Variables) -> np.ndarray:
    x = []
    u = []
    l = []
    for key, val in vals.vals.items():
        if "x" in key:
            x.append(val)
        elif "u" in key:
            u.append(val)
        elif "l" in key:
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
us = (
    np.tile(np.array([1.0, 0.0]), (sys.N, 1)) + jax.random.normal(key, (sys.N, 2)) * 0.1
)
x = sys.x0
gt.add("x0", x)
gt.add("p", sys.params)

meas = []
for i in range(sys.N):
    x = sys.step(x, us[i])
    z = sys.measure(x)

    meas.append(z)
    gt.add(f"x{i+1}", x)

for i, l in enumerate(sys.landmarks):
    gt.add(f"l{i}", l)

# ------------------------- Make the graph & initial estimates ------------------------- #
graph = core.Graph()
init = core.Variables()

# priors for inits
x = sys.x0
p = np.full(4, 1.2)
init.add("x0", x)
init.add("p", p)
graph.add(factors.PriorFactor(["x0"], sys.x0, np.eye(4) * 1e-2))
graph.add(factors.PriorFactor(["p"], p, np.eye(4) * 1))

# landmark estimates
for i, l in enumerate(sys.landmarks):
    key = sys.getkey()
    est = l + jax.random.normal(key, (2,), dtype=np.float32)
    init.add(f"l{i}", est)

# trajectory estimates
for i in range(sys.N):
    graph.add(
        factors.PastDynamics(
            ["p", f"x{i}", f"x{i+1}"], sys.dynamics, us[i], np.eye(4) * sys.std_Q
        )
    )
    x = sys.dynamics(p, x, us[i])
    init.add(f"x{i+1}", x)
    for idx, z in enumerate(meas[i]):
        graph.add(
            factors.LandmarkMeasure([f"x{i+1}", f"l{idx}"], z, np.eye(2) * sys.std_R)
        )

graph.template = init
expr = jax.make_jaxpr(graph.objective)(init.to_vec())
expr = expr._repr_pretty_()
print(expr.count("\n"))
quit()
final, info = graph.solve(init, verbose=True, jit=True)
x_est, _, l_est = vals2state(final)
x_gt, _, l_gt = vals2state(gt)
x_init, _, l_init = vals2state(init)

print("p_init", init["p"])
print("p_est", final["p"])
print("p_gt", gt["p"])

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
