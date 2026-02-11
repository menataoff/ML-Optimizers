from ..core import BaseOptimizer, Parameter
from typing import Union, Iterable
import numpy as np

class Momentum(BaseOptimizer):
    def __init__(self,
                 params: Iterable[Union[np.ndarray, Parameter]],
                 lr: float = 0.01,
                 weight_decay: float = 0.0,
                 beta: float = 0.9) -> None:

        if not 0 <= beta < 1:
            raise ValueError(f"beta must be in [0, 1), got {beta}")

        self.beta = beta
        super().__init__(params, lr, weight_decay)

        self._velocities = []
        for param in self._params:
            self._velocities.append(np.zeros_like(param.data))

    def __repr__(self):
        return f"Momentum(lr={self.lr}, beta={self.beta}, params={len(self._params)})"

    def _update_parameters(self):
        for param, velocity in zip(self._params, self._velocities):
            velocity *= self.beta
            if (self.weight_decay > 0):
                velocity -= self.lr*self.weight_decay*param.data #parametrization
            velocity -= self.lr*param.grad
            param.data += velocity