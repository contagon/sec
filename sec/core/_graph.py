import jax
import jax.numpy as np
from collections import namedtuple
import cyipopt

from ._factor import Factor
from ._variables import Variables, vec2var
from ._helpers import jitmethod


class Graph:
    def __init__(self, factors: list[Factor] = None):
        self.factors = {}
        if factors is not None:
            for f in factors:
                self.add(f)

    def add(self, factor):
        key = (type(factor), factor.constraints_dim)
        if key in self.factors:
            self.factors[key].append(factor)
        else:
            self.factors[key] = [factor]

    @property
    def dim_con(self):
        return sum(
            [
                f[0].constraints_dim * len(f)
                for f in self.factors.values()
                if f[0].has_con
            ]
        )

    @jitmethod
    def objective(self, x: jax.Array) -> float:
        def loop(out, data):
            factor = data.factors
            vals = data.values
            idx = data.idx
            if factor.has_cost:
                out += factor.cost(vals)

            return out, None

        return self._loop_factors(x, loop, 0)

    @jitmethod
    def constraints(self, x: jax.Array) -> np.ndarray:
        all = []
        values = vec2var(x, self.template)

        for key, factor in self.factors.items():
            all += [f.constraints(values[f.keys]) for f in factor if f.has_con]

        return np.concatenate(all)

    @jitmethod
    def constraints_bounds(self):
        all_lb, all_ub = [], []
        for key, factor in self.factors.items():
            for f in factor:
                if f.has_con:
                    lb, ub = f.constraints_bounds
                    all_lb.append(lb)
                    all_ub.append(ub)

        if len(all_lb) == 0:
            return None, None

        return np.concatenate(all_lb), np.concatenate(all_ub)

    @jitmethod
    def gradient(self, x: jax.Array) -> np.ndarray:
        values = vec2var(x, self.template)
        out = np.zeros(values.dim)

        def loop(out, data):
            factor = data.factors
            vals = data.values
            idx = data.idx
            if factor.has_cost:
                grad = factor.cost_grad(vals)
                for g, i in zip(grad, idx):
                    insert = g + jax.lax.dynamic_slice(out, (i,), (g.size,))
                    out = jax.lax.dynamic_update_slice(out, insert, (i,))
            return out, None

        return self._loop_factors(x, loop, out)

    def _loop_factors(self, x: jax.Array, func: callable, init):
        Step = namedtuple("Step", ["factors", "values", "idx"])
        values = vec2var(x, self.template)

        def pytrees_stack(pytrees, axis=0):
            results = jax.tree_map(
                lambda *values: np.stack(values, axis=axis), *pytrees
            )
            return results

        for factor_type, factors in self.factors.items():
            stacked_f = pytrees_stack(factors)

            stacked_v = jax.tree_map(
                lambda f: values[f.keys],
                factors,
                is_leaf=lambda n: isinstance(n, Factor),
            )
            stacked_v = pytrees_stack(stacked_v)

            stacked_i = jax.tree_map(
                lambda f: values.start_idx(f.keys),
                factors,
                is_leaf=lambda n: isinstance(n, Factor),
            )
            stacked_i = pytrees_stack(stacked_i)

            s = Step(stacked_f, stacked_v, stacked_i)
            init = jax.lax.scan(func, init, s)[0]

        return init

    @jitmethod
    # TODO: THIS IS SLOW TO JIT!
    def jacobian(self, x: jax.Array) -> np.ndarray:
        values = vec2var(x, self.template)
        all = []

        for factors in self.factors.values():
            for f in factors:
                if f.has_con:
                    jac = f.constraints_jac(values[f.keys])
                    for j in jac:
                        all.append(j.flatten())

        if len(all) == 0:
            return np.zeros(0)

        return np.concatenate(all)

    @jitmethod
    # TODO: Can speed up at all?
    def jacobianstructure(self):
        row_all, col_all = [], []
        row_idx = 0
        for factors in self.factors.values():
            for f in factors:
                if not f.has_con:
                    continue
                num_rows = f.constraints_dim
                for key in f.keys:
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

        self.template = x0
        x = x0.to_vec()
        lb = np.full(x.size, -np.inf)
        ub = np.full(x.size, np.inf)
        cl, cu = self.constraints_bounds()
        nlp = cyipopt.Problem(
            n=x0.dim,
            m=self.dim_con,
            problem_obj=self,
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
