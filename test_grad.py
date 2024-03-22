import sec.core as core
import factors
from sec.dint import DoubleIntegratorSim
import jax.numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import jax
from timeit import default_timer as timer
import sec.symbols as sym


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
sys = DoubleIntegratorSim(1, 0.1)
gt = core.Variables()

key = sys.getkey()
us = (
    np.tile(np.array([1.0, 0.0]), (sys.N, 1)) + jax.random.normal(key, (sys.N, 2)) * 0.1
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
            np.eye(4) * sys.std_Q,
        )
    )
    x = sys.dynamics(p, x, us[i])
    init.add(sym.X(i + 1), x)
    for idx, z in enumerate(meas[i]):
        graph.add(
            factors.LandmarkMeasure(
                [sym.X(i + 1), sym.L(idx)], z, np.eye(2) * sys.std_R
            )
        )


graph.template = init

start = timer()
out = graph.gradient(init.to_vec())
end = timer()
print(f"JIT time: {end - start}", out.shape)

start = timer()
out = graph.gradient(init.to_vec())
end = timer()
print(f"JIT time: {end - start}", out.shape)

# Run time with new factor
graph.add(factors.PriorFactor([sym.P(0)], p, np.eye(4) * 1))
init.add("r", np.zeros(4))
print()
start = timer()
out = graph.gradient(init.to_vec())
end = timer()
print(f"JIT time: {end - start}", len(out))

start = timer()
out = graph.gradient(init.to_vec())
end = timer()
print(f"JIT time: {end - start}", len(out))
