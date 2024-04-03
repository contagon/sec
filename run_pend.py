import sec.core as core
import sec.symbols as sym
from sec.pend import (
    PendulumSim,
    FixConstraint,
    System,
    FinalCost,
    EncoderMeasure,
    PastDynamics,
    BoundedConstraint,
    PriorFactor,
)
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)
np.set_printoptions(precision=4)


# Set up the simulation
sys = PendulumSim(4, 0.1, max_u=1, plot_live=True)
graph = core.Graph()
vals = core.Variables()

# ------------------------- Setup the initial graph & values ------------------------- #
graph.add(FixConstraint([sym.X(0)], sys.x0))
vals.add(sym.X(0), sys.x0)
idx_fix_params = graph.add(FixConstraint([sym.P(0)], sys.params * 0.9))
vals.add(sym.P(0), sys.params * 0.9)

indices = [[] for i in range(sys.N)]

x = sys.x0.copy()
for i in range(sys.N):
    f_idx = graph.add(
        System(
            [sym.P(0), sym.X(i), sym.X(i + 1), sym.U(i)],
            sys.dynamics,
            sys.xg,
            np.eye(2),
            0.1 * np.eye(1),
        )
    )
    indices[i].append(f_idx)

    f_idx = graph.add(BoundedConstraint([sym.U(i)], -1 * np.ones(1), 1 * np.ones(1)))
    indices[i].append(f_idx)

xs = np.linspace(sys.x0, sys.xg, sys.N + 1)
for i, x in enumerate(xs[1:]):
    vals.add(sym.X(i + 1), x + sys.perturb(2))
    vals.add(sym.U(i), sys.perturb(1))

graph.add(FinalCost([sym.X(sys.N)], sys.xg, 100 * np.eye(2)))

vals, _ = graph.solve(vals, verbose=True, max_iter=1000, check_derivative=False)
print("Step 0 done", vals[sym.P(0)])
sys.plot(0, vals)

graph.remove(idx_fix_params)
graph.add(BoundedConstraint([sym.P(0)], np.array([0.8, 0.3]), np.array([1.4, 0.9])))

# ------------------------- Iterate through the simulation ------------------------- #
x = sys.x0.copy()
gt = core.Variables({sym.X(0): sys.x0})
for i in range(sys.N):
    # Step
    u = vals[sym.U(i)]
    x = sys.dynamics(sys.params, x, u)
    z = sys.measure(x)
    gt[sym.X(i + 1)] = x.copy()
    gt[sym.U(i)] = vals[sym.U(i)]

    # Remove old factors/values
    for idx in indices[i]:
        graph.remove(idx)

    # Put new ones in
    graph.add(
        PastDynamics(
            [sym.P(0), sym.X(i), sym.X(i + 1), sym.U(i), sym.W(i)],
            sys.dynamics,
            np.eye(2) * sys.std_Q**2,
        )
    )
    graph.add(FixConstraint([sym.U(i)], u))
    vals.add(sym.W(i), np.zeros(2))

    graph.add(EncoderMeasure([sym.X(i + 1)], z, np.eye(1) * sys.std_R**2))

    if i < 3:
        continue

    graph.template = vals
    c_before = graph.objective(vals.to_vec())

    vals_new, info = graph.solve(vals, verbose=False, max_iter=500)
    print(info["status_msg"])

    c_after = graph.objective(vals_new.to_vec())

    print(f"Step {i+1} done", c_before, c_after, c_before - c_after)
    sys.plot(i + 1, vals, gt)
    if c_before - c_after > -1e3:
        print("Accepted", vals[sym.P(0)])
        vals = vals_new

plt.show(block=True)
