import jaxlie
import jax
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
        # TODO: Verify I'm doing this in the right order
        return val1 * val2.inverse()
    else:
        return val1 - val2
