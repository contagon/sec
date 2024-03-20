from __future__ import annotations
import jax.numpy as np
import jax
from typing import Union, TypeVar
from collections import OrderedDict
import jaxlie
from collections.abc import Iterable
import cyipopt
from helpers import jitmethod, jitclass
import jax_dataclasses as jdc

GroupType = TypeVar("GroupType", bound=jaxlie.MatrixLieGroup)
Variable = Union[jax.Array, GroupType]


def get_dim(val: Variable) -> int:
    if isinstance(val, jaxlie.MatrixLieGroup):
        return val.tangent_dim
    else:
        return val.size


def add(val: Variable, delta: jax.Array) -> Variable:
    assert delta.size == get_dim(val), "Dimension mismatch in add"
    if isinstance(val, jaxlie.MatrixLieGroup):
        return val * type(val).exp(val.zero())
    else:
        return val + delta


def vec2var(vec: np.ndarray, template: Variables) -> Variable:
    # Only works for linear variables
    out = {}
    idx = 0
    for key, val in template.vals.items():
        dim = get_dim(val)
        out[key] = vec[idx : idx + dim]
        idx += dim
    return Variables(out)


# TODO: Check if variable already exists
class Variables:
    def __init__(self, vals: OrderedDict = None):
        if vals is None:
            self.vals = OrderedDict()
        else:
            self.vals = vals

        self.reindex()

    def reindex(self):
        self.dim = 0
        self.idx_start = {}
        self.idx_end = {}
        for key, val in self.vals.items():
            self.idx_start[key] = self.dim
            self.dim += get_dim(val)
            self.idx_end[key] = self.dim

    def copy(self):
        new = Variables(self.vals.copy())
        new.dim = self.dim
        new.idx_start = self.idx_start.copy()
        new.idx_end = self.idx_end.copy()
        return new

    def add(self, key, value):
        self.vals[key] = value
        self.idx_start[key] = self.dim
        self.dim += get_dim(value)
        self.idx_end[key] = self.dim

    def rm(self, key):
        del self.vals[key]
        self.reindex()

    def __add__(self, other: np.ndarray):
        assert self.dim == other.size, "Dimension mismatch"

        new_vals = self.vals.copy()
        idx = 0
        for key, val in self.vals.items():
            delta = other[idx : idx + val.dim]
            new_vals[key] = add(val, delta)
            idx += get_dim(val)

        return Variables(new_vals)

    def __getitem__(self, key):
        if isinstance(key, Iterable) and not isinstance(key, str):
            return [self.vals[k] for k in key]
        return self.vals[key]

    def __setitem__(self, key, value):
        self.vals[key] = value

    def idx(self, key) -> tuple[int, int]:
        return self.idx_start[key], self.idx_end[key]

    def to_vec(self):
        # Only works in the linear case!
        out = []
        for key, val in self.vals.items():
            out.append(val)
        return np.concatenate(out)

    def _tree_flatten(self):
        return jax.tree_util.tree_flatten(self.vals)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(jax.tree_util.tree_unflatten(aux_data, children))


jax.tree_util.register_pytree_node(
    Variables, Variables._tree_flatten, Variables._tree_unflatten
)


class GraphJit:
    def __init__(self, graph):
        self.objective = jax.jit(graph.objective)
        self.constraints = jax.jit(graph.constraints)
        self.gradient = jax.jit(graph.gradient)
        self.jacobian = jax.jit(graph.jacobian)
        self.jacobianstructure = jax.jit(graph.jacobianstructure)


class Graph:
    def __init__(self, factors: list[Factor] = None):
        if factors is None:
            self.factors = []
        else:
            self.factors = factors
        self.dim_res = sum([f.residual_dim for f in self.factors if f.has_res])

    @property
    def dim_con(self):
        return sum([f.constraints_dim for f in self.factors if f.has_con])

    def add(self, factor):
        self.factors.append(factor)

    def objective(self, x: jax.Array) -> float:
        values = vec2var(x, self.template)
        return sum([f.cost(values[f.keys]) for f in self.factors if f.has_cost])

    def constraints(self, x: jax.Array) -> np.ndarray:
        all = []
        values = vec2var(x, self.template)
        all = [f.constraints(values[f.keys]) for f in self.factors if f.has_con]
        return np.concatenate(all)

    def gradient(self, x: jax.Array) -> np.ndarray:
        values = vec2var(x, self.template)
        out = np.zeros(values.dim)

        for factor in self.factors:
            if factor.has_cost:
                grad = factor.cost_grad(values[factor.keys])
                for key, g in zip(factor.keys, grad):
                    start, end = self.template.idx(key)
                    out = out.at[start:end].add(g)

        return out

    def jacobian(self, x: jax.Array) -> np.ndarray:
        values = vec2var(x, self.template)
        all = []

        for factor in self.factors:
            if factor.has_con:
                jac = factor.constraints_jac(values[factor.keys])
                for j in jac:
                    all.append(j.flatten())

        if len(all) == 0:
            return np.zeros(0)

        return np.concatenate(all)

    def jacobianstructure(self):
        row_all, col_all = [], []
        row_idx = 0
        for factor in self.factors:
            if not factor.has_con:
                continue
            num_rows = factor.constraints_dim
            for key in factor.keys:
                col, row = np.meshgrid(
                    np.arange(*self.template.idx(key)),
                    np.arange(row_idx, row_idx + num_rows),
                )
                col_all.append(col.flatten())
                row_all.append(row.flatten())
            row_idx += num_rows

        if len(row_all) == 0:
            return np.zeros(0), np.zeros(0)

        return np.concatenate(row_all), np.concatenate(col_all)

    # def hessian(self, x: jax.Array, lagrange, obj_factor) -> np.ndarray:
    #     values = vec2var(x, self.template)
    #     all = []

    #     for factor in self.factors:
    #         if factor.has_cost:
    #             hess = factor.cost_hess(values[factor.keys])
    #             for h in hess:
    #                 all.append(h)

    #     return np.concatenate(all)

    def solve(self, x0: Variables, jit: bool = True, verbose: bool = False):
        if jit:
            graph = GraphJit(self)
        else:
            graph = self

        self.template = x0
        x = x0.to_vec()
        lb = np.full(x.size, -np.inf)
        ub = np.full(x.size, np.inf)
        if self.dim_con == 0:
            cl = None
            cu = None
        else:
            cl = np.zeros(self.dim_con)
            cu = np.zeros(self.dim_con)
        nlp = cyipopt.Problem(
            n=x0.dim,
            m=self.dim_con,
            problem_obj=graph,
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu,
        )
        nlp.add_option("max_iter", 200)
        # if not verbose:
        # nlp.add_option("print_level", 0)
        sol, info = nlp.solve(x)
        return vec2var(sol, self.template), info


@jitclass
@jdc.pytree_dataclass
class Factor:
    keys: jdc.Static[list[str]]

    @property
    def has_cost(self):
        return type(self).cost != Factor.cost

    @property
    def has_con(self):
        return type(self).constraints != Factor.constraints

    @property
    def has_res(self):
        return type(self).residual != Factor.residual

    # ------------------------- Nonlinear Least-Squares ------------------------- #
    # TODO: Use residuals instead of cost??? I think all costs will be nonlinear least squares
    def residual(self, values: list[Variable]) -> np.ndarray:
        pass

    @property
    def residual_dim(self):
        return None

    def residual_jac(self, values: list[Variable]) -> tuple:
        if not hasattr(self, "_residual_jac"):
            self._residual_jac = jax.jacfwd(self.residual)

        return self._residual_jac(values)

    # ------------------------- Costs ------------------------- #
    def cost(self, values: list[Variable]) -> float:
        pass

    def cost_grad(self, values: list[Variable]):
        return jax.grad(self.cost)(values)

    def cost_hess(self, values: list[Variable]):
        return jax.jacrev(jax.jacfwd(self.cost))(values)

    # ------------------------- Constraints ------------------------- #
    def constraints(self, values: list[Variable]) -> np.ndarray:
        return None

    @property
    def constraints_dim(self) -> int:
        return None

    def constraints_jac(self, values: list[Variable]):
        return jax.jacfwd(self.constraints)(values)

    def constraints_hess(self, values: list[Variable]):
        return jax.jacrev(jax.jacfwd(self.constraints))(values)
