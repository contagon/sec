import core
import factors
from double_integrator import DoubleIntegratorSim
import jax.numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


def vals2state(vals: core.Variables) -> np.ndarray:
    x = []
    u = []
    for key, val in vals.vals.items():
        if "x" in key:
            x.append(val)
        elif "u" in key:
            u.append(val)

    if len(x) == 0:
        x = None
    else:
        x = np.vstack(x)

    if len(u) == 0:
        u = None
    else:
        u = np.vstack(u)

    return x, u


# Set up the simulation
x0 = np.array([1.0, 1, -1, -1])
sys = DoubleIntegratorSim(5, 0.1, x0)
graph = core.Graph()
vals = core.Variables()

# Setup trajectory opt
graph.add(factors.FixConstraint(["x0"], sys.x0))
vals.add("x0", sys.x0)

for i in range(sys.N):
    graph.add(
        factors.LinearSystem(
            [f"x{i}", f"x{i+1}", f"u{i}"], sys.A, sys.B, np.eye(4), 0.1 * np.eye(2)
        )
    )
    vals.add(f"x{i+1}", np.ones(4, dtype=np.float32))
    vals.add(f"u{i}", np.ones(2, dtype=np.float32))

graph.add(factors.FinalCost([f"x{sys.N}"], 10 * np.eye(4)))

graph_se = core.Graph()
vals_se = core.Variables()
vals_se.add("x0", sys.x0)
graph_se.add(factors.PriorFactor(["x0"], sys.x0, np.eye(4) * 1e-2))

sol, _ = graph.solve(vals)
gt = vals_se.copy()

# Run open loop controller
sys.x = sol[f"x0"]
for i in trange(sys.N):
    # closed loop dynamics
    u = sol[f"u{i}"]
    x, z = sys.dynamics(vals_se[f"x{i}"], u)

    # Estimate state
    vals_se.add(f"x{i+1}", x)
    graph_se.add(
        factors.ProbLinearSystem(
            [f"x{i}", f"x{i+1}"], sys.A, sys.B, u, np.eye(4) * sys.std_Q
        )
    )
    graph_se.add(factors.PriorFactor([f"x{i+1}"], z, np.eye(4) * sys.std_R))
    vals_se, _ = graph_se.solve(vals_se)

    gt[f"x{i+1}"] = x


xsim, usim = vals2state(sol)
xact, uact = vals2state(vals_se)
xgt, ugt = vals2state(gt)

t = np.linspace(0, sys.T, sys.N + 1)
fig, ax = plt.subplots(1, 2)
ax[0].plot(t, xsim[:, 0], label="Xsim")
ax[1].plot(t, xsim[:, 1], label="Ysim")

ax[0].plot(t, xact[:, 0], label="Xact")
ax[1].plot(t, xact[:, 1], label="Yact")

# ax[0].plot(t, xclo[:, 0], label="Xclo")
# ax[1].plot(t, xclo[:, 1], label="Yclo")

ax[0].plot(t, xgt[:, 0], label="Xgt")
ax[1].plot(t, xgt[:, 1], label="Ygt")

ax[0].legend()
ax[1].legend()
plt.show()
