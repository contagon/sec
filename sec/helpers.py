from typing import Any, Callable, Sequence, Union, TypeVar
from jaxlie.manifold import rplus, zero_tangents
import jax
from typing_extensions import ParamSpec
import jax.numpy as np
import matplotlib
import seaborn as sns

AxisName = Any
P = ParamSpec("P")
T = TypeVar("T")


def setup_plot():
    matplotlib.rc("pdf", fonttype=42)
    sns.set_context("paper")
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
    c = sns.color_palette("colorblind")
    # move gray to front for ground truth
    # c.insert(0, (0.2, 0.2, 0.2))
    return c


def wrap2pi(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def jitmethod(fun: Callable[P, Any]) -> Callable[P, Any]:
    """Decorator for marking methods for JIT compilation."""

    return jax.jit(fun, static_argnums=(0,))


def jitclass(cls: T) -> T:
    """Decorator for registering Lie group dataclasses."""

    # JIT all methods.
    for f in filter(
        lambda f: not f.startswith("_")
        and not "hash" in f
        and callable(getattr(cls, f))
        and "jit" not in getattr(cls, f).__repr__(),
        dir(cls),
    ):
        setattr(cls, f, jax.jit(getattr(cls, f)))

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


def grad(
    fun: Callable[P, Any],
    argnums: Union[int, Sequence[int]] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
):
    """Same as `jax.grad`, but computes gradients of Lie groups with respect to
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

        grad = jax.grad(
            fun=tangent_fun,
            argnums=argnums,
            has_aux=has_aux,
            holomorphic=holomorphic,
        )(*tangent_args, **tangent_kwargs)
        return grad

    return wrapped_grad  # type: ignore
