import numpy as np
from core import BaseOptimizer, Parameter
from typing import Union, Iterable

class GradientDescent(BaseOptimizer):
    """
    Classic gradient descent algorithm.
    """
    def __init__(self,
                 parameters: Iterable[Union[np.ndarray, list, tuple, Parameter]],
                 lr: float = 0.01,
                 weight_decay: float = 0.0) -> None:
        super().__init__(parameters, lr)
        self._weight_decay = weight_decay

    def __repr__(self):
        return f"GradientDescent(lr={self.lr}, params={len(self._params)})"

    def _update_parameters(self) -> None:
        for param in self._params:
            param.data -= self.lr * (param.grad + self._weight_decay * param.grad)

    #TODO: изменить работу с массивами, так как есть проблема с этим
