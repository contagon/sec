import cyipopt
from jax.scipy.linalg import expm
import jax.numpy as np
import jax
import matplotlib.pyplot as plt
from helpers import jitmethod


class DoubleIntegratorSim:
    def __init__(
        self, T, dt, std_Q=0.01, std_R=0.05, num_landmarks=10, params=np.ones(4)
    ):
        self.T = T
        self.dt = dt
        self.N = int(T / dt)
        self.std_Q = std_Q
        self.std_R = std_R
        self.params = params
        self.key = jax.random.PRNGKey(0)

        self.x0 = np.zeros(4)
        self.x = self.x0.copy()
        self.xg = np.array([10, 0, 0, 0])

        self.key, subkey = jax.random.split(self.key)
        self.landmarks = jax.random.uniform(
            subkey,
            (num_landmarks, 2),
            minval=np.array([2, -2]),
            maxval=np.array([8, 2]),
        )

    def reset(self):
        self.x = self.x0.copy()

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
