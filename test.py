from timeit import timeit

SETUP = """
import core
from double_integrator import DoubleIntegratorSim
import symbols as sym
import jax
import jax.numpy as np
from factors import PastDynamics
np.set_printoptions(precision=3, suppress=True, linewidth=200)
jax.config.update("jax_platforms", "cpu")

x0 = np.array([1.0, 1, -1, 1])
p0 = np.ones(4)
sys = DoubleIntegratorSim(5, 0.1, x0)
graph = core.Graph()
vals = core.Variables()

f = PastDynamics([sym.P(0), sym.X(0), sym.X(1)], sys.dynamics, np.ones(2), np.eye(4))
"""

# print(f.residual([p0, x0, x0]))
# print(j := f.residual_jac([p0, x0, x0]))
# print()

# print(np.hstack(j).shape)

print(timeit("f.cost_grad([p0, x0, x0])", setup=SETUP))
print(timeit("f.cost_grad1([p0, x0, x0])", setup=SETUP))
