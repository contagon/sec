from .. import core
from sec.helpers import jitclass

from overrides import overrides
import jax.numpy as np
import jax_dataclasses as jdc


@jitclass
@jdc.pytree_dataclass
class EncoderMeasure(core.Factor):
    mm: np.ndarray
    W: np.ndarray

    @overrides
    def residual(self, values: list[core.Variable]) -> np.ndarray:
        x = values[0]
        return x[0] - self.mm

    @property
    def residual_dim(self) -> int:
        return self.mm.shape[0]
