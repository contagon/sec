import jaxlie
import jax
import jax.numpy as np
from typing import TypeVar, Union

GroupType = TypeVar("GroupType", bound=jaxlie.MatrixLieGroup)
Variable = Union[jax.Array, GroupType]


def dim(val: Variable) -> int:
    if isinstance(val, jaxlie.MatrixLieGroup):
        return val.tangent_dim
    else:
        return val.size


def add(val: Variable, delta: jax.Array) -> Variable:
    assert delta.size == dim(val), "Dimension mismatch in add"
    if isinstance(val, jaxlie.MatrixLieGroup):
        return val @ type(val).exp(delta)
    else:
        return val + delta


def sub(val1: Variable, val2: Variable) -> Variable:
    assert type(val1) == type(val2), "Dimension mismatch in sub"
    if isinstance(val1, jaxlie.MatrixLieGroup):
        return (val2.inverse() @ val1).log()
    else:
        return val1 - val2


def eye(val: Variable) -> Variable:
    if isinstance(val, jaxlie.MatrixLieGroup):
        return type(val).identity(val.shape[0])
    else:
        return np.zeros(val.size)


def apply(X: Variable, v: jax.Array) -> Variable:
    if type(X) == jaxlie.SE2 or type(X) == jaxlie.SE3:
        assert dim(X) == dim(v) + 1, "Dimension mismatch in apply"
        return X.rotation().apply(v) + X.translation()

    else:
        assert dim(X) == dim(v), "Dimension mismatch in apply"
        return X.apply(v)


def copy(X: Variable) -> Variable:
    if isinstance(X, jaxlie.MatrixLieGroup):
        return type(X)(X.parameters())
    else:
        return X.copy()


@jax.jit
def stack(pytrees, axis=0):
    results = jax.tree_map(lambda *values: np.stack(values, axis=axis), *pytrees)
    return results
