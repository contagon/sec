import jax.numpy as np
import jax
from sec.helpers import jitmethod, wrap2pi, jitclass
import matplotlib.pyplot as plt
import jax_dataclasses as jdc
from jaxlie import SO3
import sec.operators as op
from sec.operators import DroneState
import mpl_toolkits.mplot3d.axes3d as Axes3D


@jitmethod
def rk4(dynamics: callable, params: jax.Array, x: jax.Array, u: jax.Array, dt: float):
    # vanilla RK4
    k1 = dt * dynamics(params, x, u)
    k2 = dt * dynamics(params, op.add(x, k1 / 2), u)
    k3 = dt * dynamics(params, op.add(x, k2 / 2), u)
    k4 = dt * dynamics(params, op.add(x, k3), u)
    delta = (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return op.add(x, delta)


def sphere_pts(c, r):
    u, v = np.mgrid[0 : 2 * np.pi : 50j, 0 : np.pi : 50j]
    x = r * np.cos(u) * np.sin(v)
    y = r * np.sin(u) * np.sin(v)
    z = r * np.cos(v)

    return c[0] + x, c[1] + y, c[2] + z


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
        params=np.array([0.5, 0.175]),  # [mass, L]
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
    @jax.jit
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

    @jitmethod
    def dynamics(self, params, x, u):
        return rk4(self._cont_system, params, x, u, self.dt)

    def step(self, x, u):
        return self.dynamics(self.params, x, u) + self.perturb(3, self.std_Q)

    def measure(self, x):
        dist = np.linalg.norm(self.landmarks - x.p, axis=1)
        use = np.where(dist < self.range)[0]
        R = x.q.as_matrix()

        measure = {}
        for i in use:
            mm = R @ (self.landmarks[int(i)] - x.p)
            measure[int(i)] = mm + self.perturb(3, self.std_R)

        return measure

    def plot(self, idx, vals, gt=None):
        import matplotlib.pyplot as plt

        if not self.plot_started:
            self.plot_started = True
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(projection="3d")
            self.ax = [self.ax]

            color_gt = "k"
            color_est = "b"
            color_fut = "r"

            (self.traj_gt,) = self.ax[0].plot(
                [],
                [],
                [],
                c=color_gt,
                marker="o",
                ms=3,
            )
            (self.traj_est,) = self.ax[0].plot(
                [], [], [], c=color_est, marker="o", ms=3
            )
            (self.traj_fut,) = self.ax[0].plot(
                [], [], [], c=color_fut, marker="o", ms=3
            )

            (self.drone1,) = self.ax[0].plot([], [], [], c=color_gt, marker="o", ms=3)
            (self.drone2,) = self.ax[0].plot([], [], [], c=color_gt, marker="o", ms=3)

            if self.landmarks.size > 0:
                self.lm_est = self.ax[0].scatter([], [], [], c=color_est, alpha=1.0)
                self.lm_true = self.ax[0].scatter(
                    self.landmarks[:, 0],
                    self.landmarks[:, 1],
                    self.landmarks[:, 2],
                    c=color_gt,
                    alpha=0.5,
                )

            self.circle_est = []
            for l in self.landmarks:
                surf = self.ax[0].plot_surface(
                    *sphere_pts(l, self.dist), color=color_est, alpha=0.0
                )
                self.circle_est.append(surf)

            self.ax[0].set_xlim([self.x0.p[0] - 1, self.xg.p[0] + 1])
            self.ax[0].set_ylim([self.x0.p[1] - 1, self.xg.p[1] + 1])
            self.ax[0].set_zlim([self.x0.p[2] - 1, self.xg.p[2] + 1])
            self.ax[0].set_aspect("equal")

            if self.plot_live:
                plt.ion()
                plt.show()
                plt.pause(0.001)

        s = vals.stacked()

        # trajectory
        X = s["X"]
        self.traj_est.set_data_3d(
            X.p[: idx + 1, 0], X.p[: idx + 1, 1], X.p[: idx + 1, 2]
        )
        self.traj_fut.set_data_3d(X.p[idx:, 0], X.p[idx:, 1], X.p[idx:, 2])

        # Drone
        R = SO3(X.q.wxyz[idx]).as_matrix()
        t = X.p[idx]
        L = s["P"][0, 1]
        points = np.array(
            [[-L, 0, 0], [L, 0, 0], [0, 0, 0], [0, -L, 0], [0, L, 0], [0, 0, 0]]
        )
        points = points @ R.T + t
        self.drone1.set_data_3d(points[:3, 0], points[:3, 1], points[:3, 2])
        self.drone2.set_data_3d(points[3:, 0], points[3:, 1], points[3:, 2])

        # Landmarks
        if "L" in s:
            L = s["L"]
            self.lm_est._offsets3d = (L[:, 0], L[:, 1], L[:, 2])
            # for i, l in enumerate(L):
            #     self.circle_est[i].set_verts(sphere_pts(l, self.dist))
            #     self.circle_est[i].set_alpha(0.5)

        # GT
        if gt is not None:
            X = gt.stacked()["X"]
            self.traj_gt.set_data_3d(X.p[:, 0], X.p[:, 1], X.p[:, 2])

        plt.draw()
        if self.plot_live:
            plt.pause(0.001)

        if self.filename is not None:
            plt.savefig(self.filename.format(idx))
