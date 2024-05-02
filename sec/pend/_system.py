import jax.numpy as np
import jax
from sec.helpers import jitmethod, setup_plot
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
        std_Q=0.005,
        std_R=0.005,
        max_u=3,
        params=np.array([1, 0.5]),  # [m, l]
        plot_live=False,
        snapshots=None,
        filename=None,
    ):
        self.T = T
        self.dt = dt
        self.N = int(T / dt)
        self.std_Q = std_Q
        self.std_R = std_R
        self.params = params
        self.max_u = max_u
        self.key = jax.random.PRNGKey(0)

        self.x0 = np.array([1e-6, 0])
        self.xg = np.array([np.pi, 0])
        self.params_est = np.zeros([0, 2])

        self.plot_started = False
        self.plot_live = plot_live
        if snapshots is None:
            self.snapshots = []
        else:
            self.snapshots = snapshots
        self.snapshots.append(self.N + 1)

        self.filename = filename

    def getkey(self):
        self.key, subkey = jax.random.split(self.key)
        return subkey

    def perturb(self, size, std=1e-4):
        return jax.random.normal(self.getkey(), (size,)) * std

    @staticmethod
    @jax.jit
    def _cont_system(params, x, u):
        m, l = params
        g = 9.81
        theta, theta_dot = x
        theta_ddot = -g / l * np.sin(theta) + u[0] / (m * l**2) - 0.01 * theta_dot
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

    def _setup_ax(self):
        self.traj_gt = self.ax[self.ax_num].plot(
            [],
            [],
            c=self.color_gt,
            marker="o",
            label="GT",
            ms=3,
        )
        (self.traj_est,) = self.ax[self.ax_num].plot(
            [],
            [],
            c=self.color_est,
            marker="o",
            label="Estimate",
            ms=3,
        )
        (self.traj_fut,) = self.ax[self.ax_num].plot(
            [], [], c=self.color_fut, marker="o", label="Plan", ms=3
        )

        self.ax[self.ax_num].set_xlim([-0.1, self.T + 0.1])
        self.ax[self.ax_num].set_ylim([-1.2, np.pi + 0.3])
        self.ax[self.ax_num].set_title(f"i = {self.snapshots[self.ax_num]}")

    def plot(self, idx, vals, gt=None):
        import matplotlib.pyplot as plt

        if not self.plot_started:
            self.c = setup_plot()
            self.color_gt = self.c[7]
            self.color_est = self.c[0]
            self.color_fut = self.c[1]

            # make figure
            self.plot_started = True
            self.fig, self.ax = plt.subplots(
                1,
                len(self.snapshots) + 1,
                figsize=(8, 2.5),
                layout="constrained",
                sharex=True,
            )
            for ax in self.ax[1:-1]:
                ax.tick_params(labelleft=False)
            self.ax[0].set_ylabel(r"$\theta$")
            self.ax[len(self.snapshots) // 2].set_xlabel("Time")

            # fill in first snapshot
            self.ax_num = 0
            self._setup_ax()

            # fill in parameters
            self.ax[-1].plot(
                [0, self.T], [self.params[0], self.params[0]], c=self.color_gt
            )
            self.ax[-1].plot(
                [0, self.T], [self.params[1], self.params[1]], c=self.color_gt
            )
            (self.l_plot,) = self.ax[-1].plot(
                [], [], c=self.c[2], label=r"$\ell$ Estimate"
            )
            (self.m_plot,) = self.ax[-1].plot(
                [], [], c=self.c[3], label=r"$m$ Estimate"
            )
            self.ax[-1].set_xlim([-0.1, self.T + 0.1])
            self.ax[-1].set_ylim([self.params.min() - 0.15, self.params.max() + 0.15])
            self.ax[-1].set_title("Model Parameters")

            self.fig.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, -0.15),
                fancybox=True,
                ncol=5,
            )

            if self.plot_live:
                plt.ion()
                plt.show()
                plt.pause(0.001)

        s = vals.stacked()

        t = np.linspace(0, self.T, self.N + 1)
        X = s["X"][:, 0]
        self.traj_est.set_data(t[: idx + 1], X[: idx + 1])
        self.traj_fut.set_data(t[idx:], X[idx:])

        P = s["P"]
        while self.params_est.shape[0] < idx + 1:
            self.params_est = np.vstack([self.params_est, P])
        self.l_plot.set_data(t[: idx + 1], self.params_est[:, 0])
        self.m_plot.set_data(t[: idx + 1], self.params_est[:, 1])

        if gt is not None:
            X = gt.stacked()["X"]
            self.traj_gt[0].set_data(t[: len(X)], X[:, 0])

        plt.draw()
        if self.plot_live:
            plt.pause(0.001)

        if self.filename is not None:
            plt.savefig(self.filename.format(idx), dpi=300, bbox_inches="tight")

        if idx in self.snapshots:
            self.ax_num += 1
            self._setup_ax()
