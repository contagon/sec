import sec.core as core
import sec.symbols as sym
from sec.symbols import X, U, P, L, W
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
from time import time

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)
np.set_printoptions(precision=4)

start = time()


def step(start, name):
    print(f"{name}: {time()-start:.2f}s")
    return time()


# Set up the simulation
sys = WheelSim(5, 0.1, dist=0.55, params=np.array([0.4, 0.43]), plot_live=True)
graph = core.Graph()
vals = core.Variables()

# # ------------------------- Setup the initial graph & values ------------------------- #
vals.add(X(0), sys.x0)
graph.add(FixConstraint([X(0)], vals[X(0)]))
vals.add(P(0), np.full(2, 0.5))
idx_fix_params = graph.add(FixConstraint([P(0)], vals[P(0)]))

indices = [[] for i in range(sys.N)]
sum_vals = lambda vals: np.sum(vals[0])

x = sys.x0.copy()
for i in range(sys.N):
    f_idx = graph.add(
        System(
            [P(0), X(i), X(i + 1), U(i)],
            sys.dynamics,
            sys.xg,
            np.eye(3),
            0.1 * np.eye(2),
        )
    )
    indices[i].append(f_idx)

    f_idx = graph.add(
        BoundedConstraint([U(i)], -sys.max_u * np.ones(2), sys.max_u * np.ones(2))
    )
    indices[i].append(f_idx)

    f_idx = graph.add(
        BoundedConstraintLambda(
            [U(i)],
            sum_vals,
            np.array([0]),
            np.array([np.inf]),
        )
    )
    indices[i].append(f_idx)

xs = np.linspace(sys.x0, sys.xg, sys.N + 1)
for i, x in enumerate(xs[1:]):
    vals.add(X(i + 1), x + sys.perturb(3))
    vals.add(U(i), sys.perturb(2))

graph.add(FinalCost([X(sys.N)], sys.xg, 100 * np.eye(3)))
graph.template = vals

# t = step(start, "Setup")
# graph.objective(vals.to_vec())
# t = step(t, "Objective")
# graph.gradient(vals.to_vec())
# t = step(t, "Gradient")
# graph.constraints(vals.to_vec())
# t = step(t, "Constraints")
# graph.constraints_bounds()
# t = step(t, "constraints_bounds")
# graph.jacobian(vals.to_vec())
# t = step(t, "Jacobian")
# graph.jacobianstructure()
# t = step(t, "Jacobian Structure")
# print()
# graph.objective(vals.to_vec())
# t = step(t, "Objective")
# graph.gradient(vals.to_vec())
# t = step(t, "Gradient")
# graph.constraints(vals.to_vec())
# t = step(t, "Constraints")
# graph.constraints_bounds()
# t = step(t, "constraints_bounds")
# graph.jacobian(vals.to_vec())
# t = step(t, "Jacobian")
# graph.jacobianstructure()
# t = step(t, "Jacobian Structure")

# plt.show(block=True)
# quit()

vals, _ = graph.solve(vals, verbose=True, tol=1e-1, max_iter=2000)
print("Step 0 done", vals[P(0)])
sys.plot(0, vals)

graph.remove(idx_fix_params)
graph.add(BoundedConstraint([P(0)], np.array([0.2, 0.2]), np.array([0.6, 0.6])))

# ------------------------- Iterate through the simulation ------------------------- #
x = sys.x0.copy()
gt = core.Variables({X(0): sys.x0})
lm_seen = set()
for i in range(sys.N):
    # Step from Xi to Xi+1 using Ui
    u = vals[U(i)]
    x = sys.dynamics(sys.params, x, u)
    z = sys.measure(x)  # measure at Xi+1
    gt[X(i + 1)] = x.copy()

    # Remove old factors/values
    for idx in indices[i]:
        graph.remove(idx)

    # Put new ones in
    graph.add(
        PastDynamics(
            [P(0), X(i), X(i + 1), U(i), W(i)],
            sys.dynamics,
            np.eye(3) * sys.std_Q**2,
        )
    )
    graph.add(FixConstraint([U(i)], u))
    vals.add(W(i), np.zeros(3))

    # Process landmarks
    lm_new = set(z.keys()) - lm_seen
    lm_seen.update(lm_new)
    if len(lm_new) > 0:
        print("New landmarks detected", lm_new)
    for lm in lm_new:
        loc = x[1:3] + z[lm]
        vals.add(L(lm), loc)

        for j in range(i + 2, sys.N):
            f_idx = graph.add(LandmarkAvoid([X(j), L(lm)], sys.dist))
            # We want to remove this constraint on the timestep where we move from j-1 to j
            indices[j - 1].append(f_idx)

    for idx, mm in z.items():
        graph.add(LandmarkMeasure([X(i + 1), L(idx)], mm, np.eye(2) * sys.std_R**2))

    if i < 1:
        continue

    graph.template = vals
    c_before = graph.objective(vals.to_vec())

    vals_new, info = graph.solve(vals, verbose=False, max_iter=1000)
    print(info["status_msg"])

    c_after = graph.objective(vals_new.to_vec())

    print(f"Step {i+1} done", vals[P(0)], c_before - c_after)
    if c_before - c_after > -1e4:
        print("Accepted")
        vals = vals_new

    sys.plot(i + 1, vals, gt)

plt.show(block=True)
