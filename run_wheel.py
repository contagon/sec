import sec.core as core
import sec.symbols as sym
from sec.wheel import (
    WheelSim,
    FixConstraint,
    System,
    LandmarkAvoid,
    FinalCost,
    PriorFactor,
    LandmarkMeasure,
    PastDynamics,
    BoundedConstraint,
    BoundedConstraintLambda,
)
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)
np.set_printoptions(precision=4)

# TODO: Unfix landmarks and start estimating them. Initialize as they're seen
# TODO: Only measure landmarks within a certain distance

# Set up the simulation
sys = WheelSim(
    5, 0.1, num_landmarks=10, dist=0.45, params=np.array([0.4, 0.41]), plot_live=True
)
graph = core.Graph()
vals = core.Variables()

# ------------------------- Setup the initial graph & values ------------------------- #
graph.add(FixConstraint([sym.X(0)], sys.x0))
vals.add(sym.X(0), sys.x0)
idx_fix_params = graph.add(FixConstraint([sym.P(0)], sys.params * 1.25))
vals.add(sym.P(0), sys.params * 1.1)

indices = [[] for i in range(sys.N)]
sum_vals = lambda vals: np.sum(vals[0])

x = sys.x0.copy()
for i in range(sys.N):
    f_idx = graph.add(
        System(
            [sym.P(0), sym.X(i), sym.X(i + 1), sym.U(i)],
            sys.dynamics,
            sys.xg,
            np.eye(3),
            0.1 * np.eye(2),
        )
    )
    indices[i].append(f_idx)

    f_idx = graph.add(
        BoundedConstraint([sym.U(i)], -sys.max_u * np.ones(2), sys.max_u * np.ones(2))
    )
    indices[i].append(f_idx)

    f_idx = graph.add(
        BoundedConstraintLambda(
            [sym.U(i)],
            sum_vals,
            np.array([0]),
            np.array([np.inf]),
        )
    )
    indices[i].append(f_idx)

    for idx, l in enumerate(sys.landmarks):
        f_idx = graph.add(LandmarkAvoid([sym.X(i), sym.L(idx)], sys.dist))
        indices[i].append(f_idx)

xs = np.linspace(sys.x0, sys.xg, sys.N + 1)
for i, x in enumerate(xs[1:]):
    vals.add(sym.X(i + 1), x + sys.perturb(3))
    vals.add(sym.U(i), sys.perturb(2))

for i, l in enumerate(sys.landmarks):
    vals.add(sym.L(i), l)
    graph.add(FixConstraint([sym.L(i)], l))

graph.add(FinalCost([sym.X(sys.N)], sys.xg, 100 * np.eye(3)))

vals, _ = graph.solve(vals, verbose=True, tol=1e-1, max_iter=2000)
print("Step 0 done", vals[sym.P(0)])
sys.plot(0, vals)

# plt.show(block=True)

graph.remove(idx_fix_params)
graph.add(BoundedConstraint([sym.P(0)], np.array([0.2, 0.2]), np.array([0.6, 0.6])))

# ------------------------- Iterate through the simulation ------------------------- #
x = sys.x0.copy()
gt = core.Variables({sym.X(0): sys.x0})
for i in range(sys.N):
    # Step
    u = vals[sym.U(i)]
    x = sys.dynamics(sys.params, x, u)
    z = sys.measure(x)
    gt[sym.X(i + 1)] = x.copy()

    # Remove old factors/values
    for idx in indices[i]:
        graph.remove(idx)

    # Put new ones in
    graph.add(
        PastDynamics(
            [sym.P(0), sym.X(i), sym.X(i + 1), sym.U(i), sym.W(i)],
            sys.dynamics,
            np.eye(3) * sys.std_Q**2,
        )
    )
    graph.add(FixConstraint([sym.U(i)], u))
    vals.add(sym.W(i), np.zeros(3))

    for idx, mm in enumerate(z):
        graph.add(
            LandmarkMeasure([sym.X(i + 1), sym.L(idx)], mm, np.eye(2) * sys.std_R**2)
        )

    if i < 1:
        continue

    graph.template = vals
    c_before = graph.objective(vals.to_vec())

    vals_new, info = graph.solve(vals, verbose=False, max_iter=500)
    print(info["status_msg"])

    c_after = graph.objective(vals_new.to_vec())

    print(f"Step {i+1} done", c_before, c_after, c_before - c_after)
    sys.plot(i + 1, vals, gt)
    if c_before - c_after > -1e2:
        print("Accepted", vals[sym.P(0)])
        vals = vals_new

plt.show(block=True)
