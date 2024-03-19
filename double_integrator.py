import cyipopt
from jax.scipy.linalg import expm
import jax.numpy as np
import jax
import matplotlib.pyplot as plt


class DoubleIntegratorSim:
    def __init__(self, T, dt, x0, std_Q=0.01, std_R=0.05):
        self.T = T
        self.dt = dt
        self.N = int(T / dt)
        self.x0 = x0
        self.std_Q = std_Q
        self.std_R = std_R
        self.key = jax.random.PRNGKey(0)

        self.reset()

        delta = np.zeros((6, 6))
        delta = delta.at[:4, 2:].set(np.eye(4))
        out = expm(delta * dt)
        self.A = out[:4, :4]
        self.B = out[:4, 4:]

    def dynamics(self, x, u):
        self.key, subkey = jax.random.split(self.key)
        x_next = self.A @ x + self.B @ u + jax.random.normal(subkey, (4,)) * self.std_Q
        x_meas = x_next + jax.random.normal(subkey, (4,)) * self.std_R
        return x_next, x_meas

    def step(self, u):
        # Actual state & measured state are returned
        self.x = self.dynamics(self.x, u)
        return self.x, self.measure(self.x)

    def measure(self, x):
        self.key, subkey = jax.random.split(self.key)
        return x + jax.random.normal(subkey, (4,)) * self.std_R

    def reset(self, x0=None):
        if x0 is not None:
            self.x0 = x0
        else:
            self.x = self.x0.copy()
