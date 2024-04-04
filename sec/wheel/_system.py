import jax.numpy as np
import jax
from sec.core import jitmethod, wrap2pi
import matplotlib.pyplot as plt


@jitmethod
def rk4(dynamics: callable, params: jax.Array, x: jax.Array, u: jax.Array, dt: float):
    # vanilla RK4
    k1 = dt * dynamics(params, x, u)
    k2 = dt * dynamics(params, x + k1 / 2, u)
    k3 = dt * dynamics(params, x + k2 / 2, u)
    k4 = dt * dynamics(params, x + k3, u)
    return x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


class WheelSim:
    def __init__(
        self,
        T,
        dt,
        std_Q=0.005,
        std_R=0.005,
        max_u=8,
        dist=0.5,
        range=2.5,
        params=np.array([0.4, 0.4]),  # [r_l, r_r]
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

        self.x0 = np.array([0, 0, 0])
        self.xg = np.array([np.pi / 4, 5, 5])
        x = np.array([1, 2.5, 4])
        self.landmarks = np.array([[i, j] for i in x for j in x])

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
    @jax.jit
    def _cont_system(params, state, u):
        r_l, r_r = params
        w_l, w_r = u
        baseline = 1

        w = (r_r * w_r - r_l * w_l) / baseline
        v = (r_r * w_r + r_l * w_l) / 2

        theta, x, y = state
        theta_dot = w
        xdot = v * np.cos(theta)
        ydot = v * np.sin(theta)

        return np.array([theta_dot, xdot, ydot])

    @jitmethod
    def dynamics(self, params, x, u):
        return rk4(self._cont_system, params, x, u, self.dt)

    def step(self, x, u):
        return self.dynamics(self.params, x, u) + self.perturb(3, self.std_Q)

    def measure(self, x):
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
