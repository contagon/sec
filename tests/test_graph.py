import pytest
import sec
import sec.symbols as sym
import sec.dint
import sec.core
import jax
import jax.numpy as np
from jax.experimental.sparse import BCOO


@pytest.fixture(scope="session")  # one server to rule'em all
def graph_vals():
    # Set up the simulation
    xg = np.array([10.0, 0, 0, 0])
    sys = sec.dint.DoubleIntegratorSim(1, 0.1, num_landmarks=5, dist=0.7)
    graph = sec.core.Graph()
    vals = sec.core.Variables()

    # ------------------------- Setup the initial graph & values ------------------------- #
    graph.add(sec.dint.FixConstraint([sym.X(0)], sys.x0))
    vals.add(sym.X(0), sys.x0)
    graph.add(sec.dint.FixConstraint([sym.P(0)], sys.params))
    vals.add(sym.P(0), sys.params)

    indices = [[] for i in range(sys.N)]

    x = sys.x0.copy()
    for i in range(sys.N):
        f_idx = graph.add(
            sec.dint.System(
                [sym.P(0), sym.X(i), sym.X(i + 1), sym.U(i)],
                sys.dynamics,
                xg,
                np.eye(4),
                0.1 * np.eye(2),
            )
        )
        indices[i].append(f_idx)

        key = sys.getkey()
        u = jax.random.normal(key, (2,)) * 0.001
        x = sys.dynamics(sys.params, x, u)
        vals.add(sym.X(i + 1), x)
        vals.add(sym.U(i), u)

        for idx, l in enumerate(sys.landmarks):
            f_idx = graph.add(sec.dint.LandmarkAvoid([sym.X(i), sym.L(idx)], sys.dist))
            indices[i].append(f_idx)

    for i, l in enumerate(sys.landmarks):
        vals.add(sym.L(i), l)
        graph.add(sec.dint.FixConstraint([sym.L(i)], l))

    graph.add(sec.dint.FinalCost([sym.X(sys.N)], xg, 100 * np.eye(4)))

    for i in indices[3]:
        graph.remove(i)

    return graph, vals


def test_gradient(graph_vals):
    graph, vals = graph_vals

    graph.template = vals
    grad_expect = jax.grad(graph.objective)(vals.to_vec())
    grad_actual = graph.gradient(vals.to_vec())

    assert np.array_equal(grad_expect, grad_actual)


def test_jacobian(graph_vals):
    graph, vals = graph_vals

    graph.template = vals
    jac_expect = jax.jacfwd(graph.constraints)(vals.to_vec())

    jac_vals = graph.jacobian(vals.to_vec())
    coords = np.column_stack(graph.jacobianstructure())
    jac_actual = BCOO((jac_vals, coords), shape=(graph.dim_con, vals.dim)).todense()

    assert np.array_equal(jac_expect, jac_actual)
