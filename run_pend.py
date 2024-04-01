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

# TODO: Once parameters are re-estimated it really struggles to refind a trajectory.
# I'm not really sure why. Maybe need to code up the Hessian? Remove dependence on params better in control factors?

# Set up the simulation
sys = PendulumSim(4, 0.05, plot_live=True)
graph = core.Graph()
vals = core.Variables()

# ------------------------- Setup the initial graph & values ------------------------- #
graph.add(FixConstraint([sym.X(0)], sys.x0))
vals.add(sym.X(0), sys.x0)
graph.add(PriorFactor([sym.P(0)], sys.params * 1.1, np.eye(2) * 0.05**2))
vals.add(sym.P(0), sys.params)

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

    f_idx = graph.add(BoundedConstraint([sym.U(i)], -3 * np.ones(1), 3 * np.ones(1)))
    indices[i].append(f_idx)

xs = np.linspace(sys.x0, sys.xg, sys.N + 1)
for i, x in enumerate(xs[1:]):
    vals.add(sym.X(i + 1), x + sys.perturb(2))
    vals.add(sym.U(i), sys.perturb(1))

graph.add(FinalCost([sym.X(sys.N)], sys.xg, 100 * np.eye(2)))

vals, _ = graph.solve(vals, verbose=True, max_iter=1000)
print("Step 0 done", vals[sym.P(0)])
sys.plot(0, vals)

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
    vals.remove(sym.U(i))
    for idx in indices[i]:
        graph.remove(idx)

    # Put new ones in
    graph.add(
        PastDynamics(
            [sym.P(0), sym.X(i), sym.X(i + 1)],
            sys.dynamics,
            u,
            np.eye(2) * sys.std_Q**2,
        )
    )

    graph.add(EncoderMeasure([sym.X(i + 1)], z, np.eye(1) * sys.std_R**2))

    if i < 5:
        continue

    graph.template = vals
    c_before = graph.objective(vals.to_vec())

    vals_new, _ = graph.solve(vals, verbose=False, max_iter=1000)

    c_after = graph.objective(vals_new.to_vec())

    print(f"Step {i+1} done", c_before, c_after, c_before - c_after)
    sys.plot(i + 1, vals, gt)
    if c_before - c_after > -1e3:
        print("Accepted", vals[sym.P(0)])
        vals = vals_new

plt.show(block=True)
