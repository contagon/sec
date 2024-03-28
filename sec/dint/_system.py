import jax.numpy as np
import jax
from sec.core import jitmethod
import matplotlib.pyplot as plt


class DoubleIntegratorSim:
    def __init__(
        self,
        T,
        dt,
        std_Q=0.01,
        std_R=0.2,
        num_landmarks=10,
        params=np.ones(4),
        dist=0.5,
        plot_live=False,
        filename=None,
    ):
        self.T = T
        self.dt = dt
        self.N = int(T / dt)
        self.std_Q = std_Q
        self.std_R = std_R
        self.params = params
        self.dist = dist
        self.key = jax.random.PRNGKey(0)

        self.x0 = np.zeros(4)
        self.xg = np.array([10, 0, 0, 0])

        self.key, subkey = jax.random.split(self.key)
        self.landmarks = jax.random.uniform(
            subkey,
            (num_landmarks, 2),
            minval=np.array([1, -2]),
            maxval=np.array([9, 2]),
        )

        self.plot_started = False
        self.plot_live = plot_live
        self.filename = filename

    def getkey(self):
        self.key, subkey = jax.random.split(self.key)
        return subkey

    @staticmethod
    @jax.jit
    def _makeAB(params, dt):
        a, b, c, d = params

        A = np.array(
            [
                [1, 0, a * dt, 0],
                [0, 1, 0, b * dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        B = np.array(
            [
                [a * c * dt**2 / 2, 0],
                [0, b * d * dt**2 / 2],
                [c * dt, 0],
                [0, d * dt],
            ]
        )

        return A, B

    @jitmethod
    def dynamics(self, params, x, u):
        A, B = self._makeAB(params, self.dt)
        x_next = A @ x + B @ u
        return x_next

    def step(self, x, u):
        subkey = self.getkey()
        return (
            self.dynamics(self.params, x, u)
            + jax.random.normal(subkey, (4,)) * self.std_Q
        )

    def measure(self, x):
        subkey = self.getkey()
        return (
            self.landmarks
            - x[:2]
            + jax.random.normal(subkey, self.landmarks.shape) * self.std_R
        )

    def plot(self, vals, idx):
        import matplotlib.pyplot as plt

        if not self.plot_started:
            self.plot_started = True
            self.fig, self.ax = plt.subplots(1, 1)
            self.ax = [self.ax]

            (self.traj_est,) = self.ax[0].plot(
                [], [], c="b", marker="o", label="Estimate"
            )
            (self.traj_fut,) = self.ax[0].plot([], [], c="r", marker="o", label="Plan")

            self.lm_est = self.ax[0].scatter([], [], c="b")
            self.lm_true = self.ax[0].scatter(
                self.landmarks[:, 0], self.landmarks[:, 1], c="k", alpha=0.5
            )

            self.ax[0].set_xlim([-1, 11])
            self.ax[0].set_ylim([-3, 3])

            if self.plot_live:
                plt.ion()
                plt.show()
                plt.pause(0.001)

        t = np.linspace(0, self.T, self.N + 1)
        s = vals.stacked()

        X = s["X"]
        L = s["L"]

        self.traj_est.set_data(X[: idx + 1, 0], X[: idx + 1, 1])
        self.traj_fut.set_data(X[idx:, 0], X[idx:, 1])

        self.lm_est.set_offsets(L)

        # ax[0].scatter(self.landmarks[:, 0], self.landmarks[:, 1], c="k", alpha=0.5)
        # ax[0].scatter(L[:, 0], L[:, 1], c="b")
        # for l in L:
        #     ax[0].add_artist(plt.Circle(l, self.dist, fill=False, color="b"))

        plt.draw()
        plt.pause(0.001)

        if self.filename is not None:
            plt.savefig(self.filename.format(idx))
