import core
import factors
from double_integrator import DoubleIntegratorSim
import jax.numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import jax
from timeit import default_timer as timer


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
    # graph.add(
    #     factors.PastDynamics(
    #         ["p", f"x{i}", f"x{i+1}"], sys.dynamics, us[i], np.eye(4) * sys.std_Q
    #     )
    # )
    x = sys.dynamics(p, x, us[i])
    init.add(f"x{i+1}", x)
    for idx, z in enumerate(meas[i]):
        graph.add(
            factors.LandmarkMeasure([f"x{i+1}", f"l{idx}"], z, np.eye(2) * sys.std_R)
        )


graph.template = init
start = timer()
grad_jit = jax.jit(graph.gradient)
out = grad_jit(init.to_vec()).block_until_ready()
end = timer()
print(f"JIT time: {end - start}", out.shape)
start = timer()
out = grad_jit(init.to_vec()).block_until_ready()
end = timer()
print(f"JIT time: {end - start}", out.shape)
start = timer()


@jax.jit
def gradient(factors: list[core.Factor], values: core.Variables):
    out = jax.tree_map(lambda x: np.zeros(core.get_dim(x)), values.vals)

    for f in factors:
        if f.has_cost:
            grad = f.cost_grad(values[f.keys])
            for key, g in zip(f.keys, grad):
                out[key] += g

    return out


out = gradient(graph.factors, init)
end = timer()
print(f"JIT time: {end - start}", len(out))
start = timer()
out = gradient(graph.factors, init)
end = timer()
print(f"JIT time: {end - start}", len(out))

# Run time with new factor
graph.add(factors.PriorFactor(["r"], p, np.eye(4) * 1))
init.add("r", np.zeros(4))
print()
graph.template = init
start = timer()
out = grad_jit(init.to_vec()).block_until_ready()
end = timer()
print(f"JIT time: {end - start}", out.shape)

start = timer()
out = gradient(graph.factors, init)
end = timer()
print(f"JIT time: {end - start}", len(out))
