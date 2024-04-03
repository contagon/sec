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
        self.factor_idx = {}
        self.factor_count = 0
        if factors is not None:
            for f in factors:
                self.add(f)

    def add(self, factor):
        key = (type(factor), factor.constraints_dim)
        if key in self.factors:
            self.factors[key].append(factor)
        else:
            self.factors[key] = [factor]

        idx = len(self.factors[key]) - 1
        self.factor_idx[self.factor_count] = (key, idx)
        self.factor_count += 1
        return self.factor_count - 1

    def remove(self, idx):
        key, idx = self.factor_idx.pop(idx)
        self.factors[key][idx] = None

    @property
    def dim_con(self):
        count = 0
        for key, factor in self.factors.items():
            for f in factor:
                if f is not None and f.has_con:
                    count += f.constraints_dim

        return count

    @jitmethod
    def objective(self, x: jax.Array) -> float:
        def loop(out, data):
            factor = data.factors
            vals = data.values
            idx = data.idx
            if factor.has_cost:
                out += factor.cost(vals)

            return out, None

        return self._loop_factors(x, loop, 0)[0]

    @jitmethod
    def constraints(self, x: jax.Array) -> np.ndarray:
        def loop(out, data):
            factor = data.factors
            vals = data.values
            if factor.has_con:
                c = factor.constraints(vals)
                return out, c

            return out, None

        out = self._loop_factors(x, loop, 0)[1]
        return np.concatenate([j.flatten() for j in out if j is not None])

    def constraints_bounds(self):
        all_lb, all_ub = [], []
        for key, factor in self.factors.items():
            for f in factor:
                if f is None:
                    continue
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

        return self._loop_factors(x, loop, out)[0]

    @jitmethod
    def jacobian(self, x: jax.Array) -> np.ndarray:
        def loop(out, data):
            factor = data.factors
            vals = data.values
            if factor.has_con:
                jac = factor.constraints_jac(vals)
                return out, np.concatenate([j.flatten() for j in jac])

            return out, None

        out = self._loop_factors(x, loop, 0)[1]
        return np.concatenate([j.flatten() for j in out if j is not None])

    def _loop_factors(self, x: jax.Array, func: callable, init):
        Step = namedtuple("Step", ["factors", "values", "idx"])
        values = vec2var(x, self.template)

        def pytrees_stack(pytrees, axis=0):
            results = jax.tree_map(
                lambda *values: np.stack(values, axis=axis), *pytrees
            )
            return results

        arrays = []
        for factor_type, factors in self.factors.items():
            factors_parsed = [f for f in factors if f is not None]
            if len(factors_parsed) == 0:
                continue
            stacked_f = pytrees_stack(factors_parsed)

            stacked_v = jax.tree_map(
                lambda f: values[f.keys],
                factors_parsed,
                is_leaf=lambda n: isinstance(n, Factor),
            )
            stacked_v = pytrees_stack(stacked_v)

            stacked_i = jax.tree_map(
                lambda f: values.start_idx(f.keys),
                factors_parsed,
                is_leaf=lambda n: isinstance(n, Factor),
            )
            stacked_i = pytrees_stack(stacked_i)

            s = Step(stacked_f, stacked_v, stacked_i)
            init, arr = jax.lax.scan(func, init, s)
            arrays.append(arr)

        return init, arrays

    # TODO: Can speed up at all?
    def jacobianstructure(self):
        row_all, col_all = [], []
        row_idx = 0
        for factors in self.factors.values():
            for f in factors:
                if f is None:
                    continue
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

    def hessian(self, x: jax.Array, lagrange=None, obj_factor=None) -> np.ndarray:
        values = vec2var(x, self.template)
        all = []

        for factors in self.factors.values():
            for f in factors:
                if f.has_cost:
                    hess = f.cost_hess(values[f.keys])
                    for i, h in enumerate(hess):
                        indices = np.tril_indices_from(h[i])
                        all.append(h[i][indices].flatten())

                        for j in h[i + 1 :]:
                            all.append(j.flatten())

        return np.concatenate(all)

    def hessianstructure(self):
        row_all, col_all = [], []
        row_idx = 0
        for factors in self.factors.values():
            for f in factors:
                if f is None:
                    continue
                if not f.has_cost:
                    continue

                for i, key1 in enumerate(f.keys):
                    # Get the spot with itself
                    row_idx = self.template.idx(key1)
                    row, col = np.tril_indices(row_idx[1] - row_idx[0])
                    row += row_idx[0]
                    col += row_idx[0]
                    row_all.append(row)
                    col_all.append(col)

                    for key2 in f.keys[i + 1 :]:
                        col, row = np.meshgrid(
                            np.arange(*row_idx),
                            np.arange(*self.template.idx(key2)),
                        )
                        col_all.append(col.flatten())
                        row_all.append(row.flatten())

        if len(row_all) == 0:
            return np.zeros(0), np.zeros(0)

        return np.concatenate(row_all), np.concatenate(col_all)

    def solve(
        self,
        x0: Variables,
        verbose: bool = False,
        check_derivative: bool = False,
        max_iter: int = 100,
        tol=1e-2,
    ):

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
        # Turn off the printed header
        nlp.add_option("sb", "yes")
        nlp.add_option("max_iter", max_iter)
        nlp.add_option("tol", tol)
        if check_derivative:
            nlp.add_option("derivative_test", "first-order")
        if not verbose:
            nlp.add_option("print_level", 0)
        sol, info = nlp.solve(x)
        return vec2var(sol, self.template), info
