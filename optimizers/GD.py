import numpy as np
from core import BaseOptimizer, Parameter

class GradientDescent(BaseOptimizer):
    """
    Classic gradient descent algorithm.
    """

    def __repr__(self):
        return f"GradientDescent(lr={self.lr}, params={len(self._params)})"

    def _update_parameters(self) -> None:
        for param in self._params:
            param.data *= (1 - self.lr*self.weight_decay)
            param.data -= self.lr * (param.grad)

    #TODO: изменить работу с массивами, так как есть проблема с этим
