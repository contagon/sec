import jax.numpy as np
import jax
from sec.core import jitmethod
import matplotlib.pyplot as plt


@jitmethod
def rk4(dynamics: callable, params: jax.Array, x: jax.Array, u: jax.Array, dt: float):
    # vanilla RK4
    k1 = dt * dynamics(params, x, u)
    k2 = dt * dynamics(params, x + k1 / 2, u)
    k3 = dt * dynamics(params, x + k2 / 2, u)
    k4 = dt * dynamics(params, x + k3, u)
    return x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


class PendulumSim:
    def __init__(
        self,
        T,
        dt,
        std_Q=0.01,
        std_R=0.01,
        params=np.array([1, 0.5]),  # [m, l]
        plot_live=False,
        filename=None,
    ):
        self.T = T
        self.dt = dt
        self.N = int(T / dt)
        self.std_Q = std_Q
        self.std_R = std_R
        self.params = params
        self.key = jax.random.PRNGKey(0)

        self.x0 = np.array([1e-6, 0])
        self.xg = np.array([np.pi, 0])

        self.plot_started = False
        self.plot_live = plot_live
        self.filename = filename

    def getkey(self):
        self.key, subkey = jax.random.split(self.key)
        return subkey

    @staticmethod
    @jax.jit
    def _cont_system(params, x, u):
        # TODO: Fix these!
        m, l = params
        g = 9.81
        theta, theta_dot = x
        theta_ddot = -g / l * np.sin(theta) + u[0] / (m * l**2) - 0.1 * theta_dot
        return np.array([theta_dot, theta_ddot])

    @jitmethod
    def dynamics(self, params, x, u):
        return rk4(self._cont_system, params, x, u, self.dt)

    def step(self, x, u):
        subkey = self.getkey()
        return (
            self.dynamics(self.params, x, u)
            + jax.random.normal(subkey, (2,)) * self.std_Q
        )

    def measure(self, x):
        subkey = self.getkey()
        return x[0:1] + jax.random.normal(subkey, (1,)) * self.std_R

    def plot(self, idx, vals, gt=None):
        import matplotlib.pyplot as plt

        if not self.plot_started:
            self.plot_started = True
            self.fig, self.ax = plt.subplots(1, 1)
            self.ax = [self.ax]

            color_gt = "k"
            color_est = "b"
            color_fut = "r"

            self.traj_gt = self.ax[0].plot(
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

            l = self.params[1]
            self.ax[0].set_xlim([-0.1, self.T + 1])
            self.ax[0].set_ylim([-0.8, np.pi + 1])
            self.ax[0].set_aspect("equal")

            if self.plot_live:
                plt.ion()
                plt.show()
                plt.pause(0.001)

        s = vals.stacked()

        t = np.linspace(0, self.T, self.N + 1)
        X = s["X"][:, 0]
        self.traj_est.set_data(t[: idx + 1], X[: idx + 1])
        self.traj_fut.set_data(t[idx:], X[idx:])

        if gt is not None:
            X = gt.stacked()["X"]
            self.traj_gt[0].set_data(t[: len(X)], X[:, 0])

        plt.draw()
        plt.pause(0.001)

        if self.filename is not None:
            plt.savefig(self.filename.format(idx))
