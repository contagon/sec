from typing import Any, Callable, Sequence, Union, TypeVar, Type
from jaxlie.manifold import rplus, zero_tangents
import jax
from typing_extensions import ParamSpec
from functools import partial
import chex

AxisName = Any
P = ParamSpec("P")
T = TypeVar("T")


def jitmethod(fun: Callable[P, Any]) -> Callable[P, Any]:
    """Decorator for marking methods for JIT compilation."""

    return jax.jit(fun, static_argnums=(0,))


def jitclass(cls: T) -> T:
    """Decorator for registering Lie group dataclasses."""

    # JIT all methods.
    for f in filter(
        lambda f: not f.startswith("_")
        and callable(getattr(cls, f))
        and "jit" not in getattr(cls, f).__repr__(),
        dir(cls),
    ):
        setattr(cls, f, jax.jit(getattr(cls, f), static_argnums=(0,)))

    return cls


def jacfwd(
    fun: Callable[P, Any],
    argnums: Union[int, Sequence[int]] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
):
    """Same as `jax.jacfwd`, but computes gradients of Lie groups with respect to
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


def jacrev(
    fun: Callable[P, Any],
    argnums: Union[int, Sequence[int]] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
):
    """Same as `jax.jacrev`, but computes gradients of Lie groups with respect to
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

        grad = jax.jacrev(
            fun=tangent_fun,
            argnums=argnums,
            has_aux=has_aux,
            holomorphic=holomorphic,
        )(*tangent_args, **tangent_kwargs)
        return grad

    return wrapped_grad  # type: ignore
