from core import Value
from overrides import overrides
import jax.numpy as np
import jaxlie
from dataclasses import dataclass

@dataclass
class SE3(Value):
    val: jaxlie.SE3

    @overrides
    def __add__(self, tangent: np.ndarray):
        return SE3(self.val @ jaxlie.SE3.Exp(tangent))
    
    @overrides
    def __inv__(self):
        return SE3(self.val.inv())
    
    @property
    @overrides
    def dim(self):
        return 6