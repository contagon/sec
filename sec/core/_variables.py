from __future__ import annotations

import jax
import jaxlie
import jax.numpy as np
from typing import TypeVar, Union
from collections import OrderedDict
from collections.abc import Iterable
import sec.symbols as sym
import sec.operators as op

GroupType = TypeVar("GroupType", bound=jaxlie.MatrixLieGroup)
Variable = Union[jax.Array, GroupType]


def vec2var(vec: np.ndarray, template: Variables) -> Variable:
    # Only works for linear variables
    out = {}
    for key, val in template.vals.items():
        idx = template.start_idx(key)
        dim = op.dim(val)
        out[key] = vec[idx : idx + dim]
        idx += dim
    return Variables(out)


# TODO: Check if variable already exists
# TODO: Make variable insertion always in order so indexing is consistent.
class Variables:
    def __init__(self, vals: OrderedDict = None):
        if vals is None:
            self.vals = OrderedDict()
        else:
            self.vals = vals

        self.reindex()

    def reindex(self):
        # Sort the keys first
        self.vals = OrderedDict(sorted(self.vals.items(), key=lambda x: x[0]))

        # Then add all dimensions
        self.dim = 0
        self.idx_start = {}
        self.idx_end = {}
        for key, val in self.vals.items():
            self.idx_start[key] = self.dim
            self.dim += op.dim(val)
            self.idx_end[key] = self.dim

    def copy(self):
        new = Variables(self.vals.copy())
        new.dim = self.dim
        new.idx_start = self.idx_start.copy()
        new.idx_end = self.idx_end.copy()
        return new

    def add(self, key, value):
        self.vals[key] = value
        # TODO: This is really inefficent, but things go poorly if I don't do
        self.reindex()

    def remove(self, key):
        del self.vals[key]
        self.reindex()

    def __add__(self, other: np.ndarray):
        assert self.dim == other.size, "Dimension mismatch"

        new_vals = self.vals.copy()
        idx = 0
        for key, val in self.vals.items():
            delta = other[idx : idx + op.dim(val)]
            new_vals[key] = op.add(val, delta)
            idx += op.dim(val)

        return Variables(new_vals)

    def __getitem__(self, key):
        if isinstance(key, Iterable) and not isinstance(key, str):
            return [self.vals[k] for k in key]
        return self.vals[key]

    def __setitem__(self, key, value):
        self.vals[key] = value

    def __len__(self):
        return len(self.vals)

    def idx(self, key) -> tuple[int, int]:
        if isinstance(key, Iterable) and not isinstance(key, str):
            return [(self.idx_start[k], self.idx_end[k]) for k in key]
        return self.idx_start[key], self.idx_end[key]

    def start_idx(self, key) -> tuple[int, int]:
        if isinstance(key, Iterable) and not isinstance(key, str):
            return [self.idx_start[k] for k in key]
        return self.idx_start[key]

    def to_vec(self):
        # Only works in the linear case!
        out = []
        for key, val in self.vals.items():
            out.append(val)
        return np.concatenate(out)

    def stacked(self):
        # They're always in order b/c of the OrderedDict
        out = {}
        for key, val in self.vals.items():
            c = sym.str(key)[0]
            if c not in out:
                out[c] = []
            out[c].append(val)

        return {k: op.stack(v) for k, v in out.items()}

    def _tree_flatten(self):
        return jax.tree_util.tree_flatten(self.vals)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(jax.tree_util.tree_unflatten(aux_data, children))


jax.tree_util.register_pytree_node(
    Variables, Variables._tree_flatten, Variables._tree_unflatten
)
