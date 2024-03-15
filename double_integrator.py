import cyipopt
from jax.scipy.linalg import expm
import jax.numpy as np
import jax
import matplotlib.pyplot as plt
from functools import partial


class DoubleIntegratorSim:
    def __init__(self, T, dt, x0):
        self.T = T
        self.dt = dt
        self.N = int(T / dt) + 1
        self.x0 = x0
        self.nx = 4
        self.nu = 2
        self.nz = self.nx * self.N + self.nu * (self.N - 1)
        self.nc = self.nx * self.N

        self.reset()

        delta = np.zeros((6, 6))
        delta = delta.at[:4, 2:].set(np.eye(4))
        out = expm(delta * dt)
        self.A = out[:4, :4]
        self.B = out[:4, 4:]

    def dynamics(self, x, u):
        return self.A @ x + self.B @ u

    def step(self, u):
        self.x = self.dynamics(self.x, u)
        return self.x

    def reset(self, x0=None):
        if x0 is not None:
            self.x0 = x0
        else:
            self.x = self.x0.copy()


class LinearSolver:
    def __init__(self, sim, Q, R, Qf, xg):
        self.sim = sim
        self.Q = Q
        self.R = R
        self.Qf = Qf

        self.xg = xg

        self.gradient = jax.grad(self.objective)
        # TODO: Do this iteratively for sparsity reasons
        self._jacobian = jax.jacfwd(self.constraints)
        self._hessian_obj = jax.jacrev(jax.jacfwd(self.objective))
        self._hessian_con = jax.jacrev(jax.jacfwd(self.constraints))

    def Z2XU(self, Z):
        Xmat = Z[: self.sim.N * self.sim.nx].reshape(self.sim.N, self.sim.nx)
        Umat = Z[self.sim.N * self.sim.nx :].reshape(self.sim.N - 1, self.sim.nu)

        return Xmat, Umat

    @partial(jax.jit, static_argnums=(0,))
    def objective(self, Z):
        Xmat, Umat = self.Z2XU(Z)

        cost = 0
        for i in range(self.sim.N - 1):
            x_diff = Xmat[i] - self.xg
            cost += x_diff @ self.Q @ x_diff / 2

            cost += Umat[i] @ self.R @ Umat[i] / 2

        x_diff = Xmat[-1] - self.xg
        cost += x_diff @ self.Qf @ x_diff / 2
        return cost

    @partial(jax.jit, static_argnums=(0,))
    def constraints(self, Z):
        Xmat, Umat = self.Z2XU(Z)

        con = [Xmat[0] - self.sim.x0]
        for i in range(self.sim.N - 1):
            con.append(Xmat[i + 1] - self.sim.dynamics(Xmat[i], Umat[i]))

        return np.concatenate(con)

    def jacobianstructure(self):
        return np.nonzero(np.ones((self.sim.nc, self.sim.nz)))

    def jacobian(self, Z):
        J = self._jacobian(Z)
        row, col = self.jacobianstructure()
        return J[row, col]

    def hessianstructure(self):
        return np.nonzero(np.tril(np.ones((self.sim.nz, self.sim.nz))))

    # @partial(jax.jit, static_argnums=(0,))
    def hessian(self, Z, lagrange, obj_factor):
        H = self._hessian_obj(Z) * obj_factor + (
            self._hessian_con(Z) * lagrange.reshape((-1, 1, 1))
        ).sum(axis=0)
        row, col = self.hessianstructure()
        return H[row, col]

    def intermediate(
        self,
        alg_mod,
        iter_count,
        obj_value,
        inf_pr,
        inf_du,
        mu,
        d_norm,
        regularization_size,
        alpha_du,
        alpha_pr,
        ls_trials,
    ):
        """Prints information at every Ipopt iteration."""

        msg = "Objective value at iteration #{:d} is - {:g}"

        print(msg.format(iter_count, obj_value))


if __name__ == "__main__":
    x0 = np.array([1, 1, 0.2, 0])
    sim = DoubleIntegratorSim(5, 0.1, x0)

    Q = np.eye(4)
    R = np.eye(2) * 0.1
    Qf = np.eye(4) * 10
    xg = np.array([0.0, 0, 0, 0])
    solver = LinearSolver(sim, Q, R, Qf, xg)

    key = jax.random.key(0)
    Z0 = jax.random.uniform(key, [sim.nz])

    lb = np.full(sim.nz, -np.inf)
    ub = np.full(sim.nz, np.inf)
    cl = np.zeros(sim.nc)
    cu = np.zeros(sim.nc)
    nlp = cyipopt.Problem(
        n=len(Z0),
        m=len(cl),
        problem_obj=solver,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
    )

    x, info = nlp.solve(Z0)

    # build the derivatives and jit them
    obj = jax.jit(solver.objective)
    con = jax.jit(solver.constraints)

    obj_grad = jax.jit(jax.grad(obj))  # objective gradient
    obj_hess = jax.jit(jax.jacrev(jax.jacfwd(obj)))  # objective hessian

    con_eq_jac = jax.jit(jax.jacfwd(con))  # jacobian
    con_eq_hess = jax.jit(jax.jacrev(jax.jacfwd(con)))  # hessian
    con_eq_hessvp = lambda x, v: con_eq_hess(x) @ v  # hessian vector-product

    def con_eq_hessvp(x, v):
        return (con_eq_hess(x) * v.reshape((-1, 1, 1))).sum(axis=0)

    # constraints
    # Note that 'hess' is the hessian-vector-product
    cons = [
        {
            "type": "eq",
            "fun": solver.constraints,
            "jac": con_eq_jac,
            "hess": con_eq_hessvp,
        },
    ]

    # initial guess
    bnds = [(-np.inf, np.inf) for _ in range(sim.nz)]

    res = cyipopt.minimize_ipopt(
        solver.objective,
        jac=obj_grad,
        hess=obj_hess,
        x0=Z0,
        bounds=bnds,
        constraints=cons,
        options={"disp": 5},
    )

    Xmat, Umat = solver.Z2XU(res.x)

    # plt.plot(Xmat[:, 0], Xmat[:, 1])
    t = np.linspace(0, sim.T, sim.N)
    plt.plot(t, Xmat[:, 0], label="x")
    plt.plot(t, Xmat[:, 1], label="y")
    plt.plot(t, Xmat[:, 2], label="vx")
    plt.plot(t, Xmat[:, 3], label="vy")
    plt.legend()
    plt.show()
