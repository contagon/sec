from typing import Any, Callable, Sequence, Union
from jaxlie.manifold import rplus, zero_tangents
import jax
from typing_extensions import ParamSpec

AxisName = Any
P = ParamSpec("P")


def jacfwd(
    fun: Callable[P, Any],
    argnums: Union[int, Sequence[int]] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
):
    """Same as `jax.value_and_grad`, but computes gradients of Lie groups with respect to
    tangent spaces."""

    def wrapped_grad(*args, **kwargs):
        def tangent_fun(*tangent_args, **tangent_kwargs):
            return fun(  # type: ignore
                *rplus(args, tangent_args),
                **rplus(kwargs, tangent_kwargs),
            )

        # Put arguments onto tangent space.
        tangent_args = map(zero_tangents, args)
        tangent_kwargs = {k: zero_tangents(v) for k, v in kwargs.items()}

        grad = jax.jacfwd(
            fun=tangent_fun,
            argnums=argnums,
            has_aux=has_aux,
            holomorphic=holomorphic,
        )(*tangent_args, **tangent_kwargs)
        return grad

    return wrapped_grad  # type: ignore
