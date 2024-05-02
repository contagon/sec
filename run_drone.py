import sec.core as core
from sec.symbols import X, U, P, L, W
import sec.operators as op
from sec.operators import DroneState
from sec.drone import (
    DroneSim,
    FixConstraint,
    System,
    LandmarkAvoid,
    FinalCost,
    PriorFactor,
    LandmarkMeasure,
    PastDynamics,
    BoundedConstraint,
)
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from time import time

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)
np.set_printoptions(precision=4)


# Set up the simulation
sys = DroneSim(
    5,
    0.1,
    dist=1.0,
    plot_live=True,
    snapshots=[0, 4, 8, 14, 22],
    filename="drone.pdf",
)
graph = core.Graph()
vals = core.Variables()

# # ------------------------- Setup the initial graph & values ------------------------- #
vals.add(X(0), sys.x0)
graph.add(FixConstraint([X(0)], vals[X(0)]))
vals.add(P(0), sys.params * 1.05)
idx_fix_params = graph.add(FixConstraint([P(0)], vals[P(0)]))

# Q = np.diag(np.concatenate([np.ones(3), np.ones(3), 10 * np.ones(3), np.ones(3)]))
Q = np.eye(12)
Qf = 10 * Q


indices = [[] for i in range(sys.N)]
sum_vals = lambda vals: np.sum(vals[0])

for i in range(sys.N):
    f_idx = graph.add(
        System(
            [P(0), X(i), X(i + 1), U(i)],
            sys.dynamics,
            sys.xg,
            Q,
            0.1 * np.eye(4),
        )
    )
    indices[i].append(f_idx)

    f_idx = graph.add(
        BoundedConstraint([U(i)], -sys.max_u * np.ones(4), sys.max_u * np.ones(4))
    )
    indices[i].append(f_idx)


xs = np.linspace(sys.x0.p, sys.xg.p, sys.N + 1)
for i, x in enumerate(xs[1:]):
    vals.add(X(i + 1), op.add(DroneState.make(p=x), sys.perturb(12)))
    vals.add(U(i), sys.perturb(4))

graph.add(FinalCost([X(sys.N)], sys.xg, Qf))

vals, _ = graph.solve(vals, verbose=False, tol=1e-1, max_iter=2000)
print("Step 0 done", vals[P(0)])
sys.plot(0, vals)

graph.remove(idx_fix_params)
graph.add(BoundedConstraint([P(0)], np.array([0.4, 0.125]), np.array([0.6, 0.2])))

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
            np.eye(12) * sys.std_Q**2,
        )
    )
    graph.add(FixConstraint([U(i)], u))
    vals.add(W(i), np.zeros(12))

    # Process landmarks
    lm_new = set(z.keys()) - lm_seen
    lm_seen.update(lm_new)
    if len(lm_new) > 0:
        print("New landmarks detected", lm_new)
    x_est = vals[X(i + 1)]
    for lm in lm_new:
        l = x_est.p + x_est.q.inverse().apply(z[lm])
        vals.add(L(lm), l)

        for j in range(i + 2, sys.N):
            f_idx = graph.add(LandmarkAvoid([X(j), L(lm)], sys.dist))
            # We want to remove this constraint on the timestep where we move from j-1 to j
            indices[j - 1].append(f_idx)

    for idx, mm in z.items():
        graph.add(LandmarkMeasure([X(i + 1), L(idx)], mm, np.eye(3) * sys.std_R**2))

    # if i < 1:  # or i % 2 == 0:
    #     continue

    c_before = graph.objective(x0=vals)

    max_iter = 500
    if len(lm_new) > 0:
        max_iter = 1_000

    vals_new, info = graph.solve(vals, verbose=False, max_iter=max_iter, tol=1e-1)
    print(info["status_msg"])

    c_after = graph.objective(x0=vals_new)

    accept = False
    diff = c_before - c_after
    if 1e6 > diff and diff > -1e4:
        accept = True
        vals = vals_new

    print(
        f"Step {i+1} done, accepted: {accept}",
        vals[P(0)],
        vals[X(sys.N)].p,
        c_before - c_after,
    )
    sys.plot(i + 1, vals, gt)

plt.show(block=True)
