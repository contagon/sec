import sec.core as core
import sec.symbols as sym
from sec.dint import (
    DoubleIntegratorSim,
    FixConstraint,
    System,
    LandmarkAvoid,
    FinalCost,
    PriorFactor,
    LandmarkMeasure,
    PastDynamics,
)
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)
# np.set_printoptions(precision=3, suppress=True)

# TODO: Double check have only factors you'd expect at the end - that we didn't miss anything
# TODO: Begin estimating parameters
# TODO: Unfix landmarks and start estimating them. Initialize as they're seen
# TODO: Only measure landmarks within a certain distance
# TODO: Update landmark circles on the plot as we go

# Set up the simulation
sys = DoubleIntegratorSim(5, 0.1, num_landmarks=10, dist=0.7, plot_live=True)
graph = core.Graph()
vals = core.Variables()

# ------------------------- Setup the initial graph & values ------------------------- #
graph.add(FixConstraint([sym.X(0)], sys.x0))
vals.add(sym.X(0), sys.x0)
graph.add(FixConstraint([sym.P(0)], sys.params))
vals.add(sym.P(0), sys.params)

indices = [[] for i in range(sys.N)]

x = sys.x0.copy()
for i in range(sys.N):
    f_idx = graph.add(
        System(
            [sym.P(0), sym.X(i), sym.X(i + 1), sym.U(i)],
            sys.dynamics,
            sys.xg,
            np.eye(4),
            1 * np.eye(2),
        )
    )
    indices[i].append(f_idx)

    key = sys.getkey()
    u = jax.random.normal(key, (2,)) * 0.01
    x = sys.dynamics(sys.params, x, u)
    vals.add(sym.X(i + 1), x)
    vals.add(sym.U(i), u)

    for idx, l in enumerate(sys.landmarks):
        f_idx = graph.add(LandmarkAvoid([sym.X(i), sym.L(idx)], sys.dist))
        indices[i].append(f_idx)

for i, l in enumerate(sys.landmarks):
    vals.add(sym.L(i), l)
    graph.add(FixConstraint([sym.L(i)], l))

graph.add(FinalCost([sym.X(sys.N)], sys.xg, 100 * np.eye(4)))

vals, _ = graph.solve(vals, verbose=False, max_iter=1000)
print("Step 0 done")
sys.plot(vals, 0)

# ------------------------- Iterate through the simulation ------------------------- #
x = sys.x0.copy()
for i in range(sys.N):
    # Step
    u = vals[sym.U(i)]
    x = sys.dynamics(sys.params, x, u)
    z = sys.measure(x)

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
            np.eye(4) * sys.std_Q**2,
        )
    )

    for idx, mm in enumerate(z):
        graph.add(
            LandmarkMeasure([sym.X(i + 1), sym.L(idx)], mm, np.eye(2) * sys.std_R**2)
        )

    graph.template = vals
    c_before = graph.objective(vals.to_vec())

    vals_new, _ = graph.solve(vals, verbose=False)

    c_after = graph.objective(vals_new.to_vec())

    print(f"Step {i+1} done", c_before, c_after, c_before - c_after)
    sys.plot(vals, i + 1)
    if c_before - c_after > -1e3:
        print("Accepted")
        vals = vals_new

plt.show(block=True)

print(i, vals.vals.keys())
