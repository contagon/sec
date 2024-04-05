import jax.numpy as np
import jax
from sec.helpers import jitmethod, wrap2pi, jitclass
import matplotlib.pyplot as plt
import jax_dataclasses as jdc
from jaxlie import SO3
import sec.operators as op
from sec.operators import DroneState


# @jitmethod
def rk4(dynamics: callable, params: jax.Array, x: jax.Array, u: jax.Array, dt: float):
    # vanilla RK4
    k1 = dt * dynamics(params, x, u)
    k2 = dt * dynamics(params, op.add(x, k1 / 2), u)
    k3 = dt * dynamics(params, op.add(x, k2 / 2), u)
    k4 = dt * dynamics(params, op.add(x, k3), u)
    delta = (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return op.add(x, delta)


class DroneSim:
    def __init__(
        self,
        T,
        dt,
        std_Q=0.005,
        std_R=0.005,
        max_u=8,
        dist=0.5,
        range=2.5,
        params=np.array([0.5, 0.175]),  # [r_l, r_r]
        baseline=1,
        plot_live=False,
        filename=None,
    ):
        self.T = T
        self.dt = dt
        self.N = int(T / dt)
        self.std_Q = std_Q
        self.std_R = std_R
        self.params = params
        self.max_u = max_u
        self.dist = dist
        self.baseline = baseline
        self.range = range
        self.key = jax.random.PRNGKey(0)

        self.x0 = DroneState.identity()
        self.xg = DroneState(
            SO3.from_z_radians(np.pi / 4), np.zeros(3), np.array([5, 5, 5]), np.zeros(3)
        )
        x = np.array([1, 2.5, 4])
        self.landmarks = np.array([[i, j, k] for i in x for j in x for k in x])

        self.plot_started = False
        self.plot_live = plot_live
        self.filename = filename

    def getkey(self):
        self.key, subkey = jax.random.split(self.key)
        return subkey

    def perturb(self, size, std=1e-4):
        if type(size) is int:
            size = (size,)
        return jax.random.normal(self.getkey(), size) * std

    @staticmethod
    # @jax.jit
    def _cont_system(params, state, u):
        mass, L = params
        J = np.diag(np.array([0.0023, 0.0023, 0.0040]))
        w1, w2, w3, w4 = u
        g = np.array([0, 0, -9.81])
        kf = 1.0
        km = 0.0245

        f1 = w1 * kf
        f2 = w2 * kf
        f3 = w3 * kf
        f4 = w4 * kf
        F = np.array([0, 0, f1 + f2 + f3 + f4])
        f = mass * g + state.q.apply(F)

        m1 = km * w1
        m2 = km * w2
        m3 = km * w3
        m4 = km * w4
        tau = np.array([L * (-f2 + f4), L * (-f1 + f3), m1 - m2 + m3 - m4])

        w = state.w
        wdot = np.linalg.inv(J) @ (tau - np.cross(w, J @ w))
        v = state.v
        vdot = f / mass

        return np.concatenate([w, wdot, v, vdot])

    # @jitmethod
    def dynamics(self, params, x, u):
        return rk4(self._cont_system, params, x, u, self.dt)

    def step(self, x, u):
        return self.dynamics(self.params, x, u) + self.perturb(3, self.std_Q)

    def measure(self, x):
        # TODO
        dist = np.linalg.norm(self.landmarks - x[1:3], axis=1)
        use = np.where(dist < self.range)[0]

        measure = {}
        for i in use:
            theta, px, py = x
            lx, ly = self.landmarks[int(i)]

            angle = wrap2pi(np.arctan2(ly - py, lx - px) - theta)
            r = np.sqrt((lx - px) ** 2 + (ly - py) ** 2)

            measure[int(i)] = np.array([r, angle]) + self.perturb(2, self.std_R)

        return measure

    def plot(self, idx, vals, gt=None):
        # TODO
        import matplotlib.pyplot as plt

        if not self.plot_started:
            self.plot_started = True
            self.fig, self.ax = plt.subplots(1, 1)
            self.ax = [self.ax]

            color_gt = "k"
            color_est = "b"
            color_fut = "r"

            (self.traj_gt,) = self.ax[0].plot(
                [],
                [],
                c=color_gt,
                marker="o",
                label="Estimate",
                ms=3,
            )
            (self.traj_est,) = self.ax[0].plot(
                [],
                [],
                c=color_est,
                marker="o",
                label="Estimate",
                ms=3,
            )
            (self.traj_fut,) = self.ax[0].plot(
                [], [], c=color_fut, marker="o", label="Plan", ms=3
            )

            if self.landmarks.size > 0:
                self.lm_est = self.ax[0].scatter([], [], c=color_est)
                self.lm_true = self.ax[0].scatter(
                    self.landmarks[:, 0], self.landmarks[:, 1], c=color_gt, alpha=0.5
                )

                self.idx_dist = plt.Circle(
                    self.x0, self.range, fill=False, color=color_gt, alpha=0.3
                )
                self.ax[0].add_artist(self.idx_dist)

            self.circle_est = []
            for l in self.landmarks:
                self.circle_est.append(
                    plt.Circle(l, self.dist, fill=False, color=color_est, alpha=0.0)
                )
                self.ax[0].add_artist(self.circle_est[-1])

            self.ax[0].set_xlim([self.x0[1] - 1, self.xg[1] + 1])
            self.ax[0].set_ylim([self.x0[2] - 1, self.xg[2] + 1])
            self.ax[0].set_aspect("equal")

            if self.plot_live:
                plt.ion()
                plt.show()
                plt.pause(0.001)

        s = vals.stacked()

        X = s["X"]
        self.traj_est.set_data(X[: idx + 1, 1], X[: idx + 1, 2])
        self.traj_fut.set_data(X[idx:, 1], X[idx:, 2])

        if "L" in s:
            L = s["L"]
            self.lm_est.set_offsets(L)
            for i, l in enumerate(L):
                self.circle_est[i].set(center=l, alpha=0.5)

            self.idx_dist.set(center=X[idx, 1:3])

        if gt is not None:
            X = gt.stacked()["X"]
            self.traj_gt.set_data(X[:, 1], X[:, 2])

        plt.draw()
        if self.plot_live:
            plt.pause(0.001)

        if self.filename is not None:
            plt.savefig(self.filename.format(idx))
