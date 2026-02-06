import numpy as np
from core import BaseOptimizer, Parameter
from typing import Union, Iterable

class GradientDescent(BaseOptimizer):
    """
    Classic gradient descent algorithm.
    """
    def __init__(self,
                 parameters: Iterable[Union[np.ndarray, list, tuple, Parameter]],
                 lr: float = 0.01) -> None:
        super().__init__(parameters, lr)

    def _update_parameters(self) -> None:
        for param in self._params:
            param.data -= self.lr * param.grad

    #TODO: изменить работу с массивами, так как есть проблема с этим
