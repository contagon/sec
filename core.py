import jax.numpy as np
import jax
from dataclasses import dataclass, field
from typing import Any
from collections import OrderedDict
import jax_dataclasses as jdc


@jdc.pytree_dataclass
class Value:
    val: Any

    def __add__(self, tangent: np.ndarray):
        return Value(self.val + tangent)

    def __inv__(self):
        return Value(-self.val)

    @property
    def dim(self):
        return self.val.shape[0]


@dataclass
class Values:
    vals: OrderedDict = field(default_factory=OrderedDict)

    def __post_init__(self):
        self.dim = 0
        for key, val in self.vals.items():
            self.dim += val.dim

    def add_variable(self, key, value):
        self.vals[key] = value
        self.dim += value.dim

    def __add__(self, other: np.ndarray):
        assert self.dim == other.size, "Dimension mismatch"

        new_vals = self.vals.copy()
        idx = 0
        for key, val in self.vals.items():
            new_vals[key] = val + other[idx : idx + val.dim]
            idx += val.dim

        return Values(new_vals)

    def __getitem__(self, key):
        return self.vals[key]

    def __setitem__(self, key, value):
        self.vals[key] = value


@dataclass
class Graph:
    factors: list = field(default_factory=list)

    def __post_init__(self):
        self.dim_res = 0
        self.dim_con = 0
        for factor in self.factors:
            if (res_dim := factor.residual_dim) is not None:
                self.dim_res += res_dim
            if (con_dim := factor.constraints_dim) is not None:
                self.dim_con += con_dim

    def add_factor(self, factor):
        self.factors.append(factor)
        self.dim_res += factor.residual_dim
        self.dim_con += factor.constraints_dim

    def solve(self, values):
        pass


@dataclass
class Factor:
    keys: list

    # ------------------------- Nonlinear Least-Squares ------------------------- #
    def residual(self, values: Values) -> np.ndarray:
        pass

    @property
    def residual_dim(self):
        return None

    def residual_jac(self, values: Values) -> tuple:
        if not hasattr(self, "_residual_jac"):
            self._residual_jac = jax.jacfwd(self.residual)

        return self._residual_jac(values)

    # ------------------------- Costs ------------------------- #
    def cost(self, values: Values) -> float:
        pass

    def cost_jac(self, values: Values) -> tuple:
        if self._cost_jac is None:
            self._cost_jac = jax.jacfwd(self.cost)

        return self._cost_jac(values)

    # ------------------------- Constraints ------------------------- #
    def constraints(self, values: Values) -> np.ndarray:
        pass

    @property
    def constraints_dim(self) -> int:
        return None

    def constraints_jac(self, values: Values):
        if not hasattr(self, "_constraints_jac"):
            self._constraints_jac = jax.jacfwd(self.constraints)

        return self._constraints_jac(values)

    def constraints_hess(self, values: Values):
        if not hasattr(self, "_constraints_hess"):
            self._constraints_hess = jax.jacrev(jax.jacfwd(self.constraints))

        return self._constraints_hess(values)
