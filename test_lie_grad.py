import jaxlie
import jax.numpy as np
import jax
from helpers import jacfwd

"""
Conclusion here:

jax.manifold.grad works great. 

Got it working for jacfwd as well!
"""
key = jax.random.PRNGKey(0)
R = jaxlie.SO3.sample_uniform(key)
p = np.array([1.0, 2, 3])


def skew(omega: jax.Array) -> jax.Array:
    """Returns the skew-symmetric form of a length-3 vector."""

    wx, wy, wz = omega
    return np.array(
        [
            [0.0, -wz, wy],
            [wz, 0.0, -wx],
            [-wy, wx, 0.0],
        ]
    )


# # Check both derivatives
# print("Expected derivatives")
# print(-p @ R.as_matrix() @ skew(p))
# print()
# print("Got derivatives")
# out = jaxlie.manifold.grad(lambda R, p: p @ R.as_matrix() @ p, 0)(R, p)
# print(out)

# # Check both derivatives
# print("Expected derivatives")
# print(-R.as_matrix() @ skew(p))
# print(R.as_matrix())
# print()
# print("Got derivatives")
# out = jacfwd(lambda x: x[0].apply(x[1]))([R, p])
# print(out[0])
# print(out[1])

print(type(R) is jax.Array)
