from ..core import BaseOptimizer, Parameter
from typing import Union, Iterable
import numpy as np

class Adagrad(BaseOptimizer):
    def __init__(self,
                 params: Iterable[Union[np.ndarray, Parameter]],
                 lr: float = 0.01,
                 weight_decay: float = 0.0,
                 eps: float = 1e-8) -> None:

        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.eps = eps
        super().__init__(params, lr, weight_decay)

        self._Gs = []
        for param in self._params:
            self._Gs.append(np.zeros_like(param.data))

    def __repr__(self):
        return f"Adagrad(lr={self.lr}, eps={self.eps}, params={len(self._params)})"

    def _update_parameters(self):
        for param, G in zip(self._params, self._Gs):
            if self.weight_decay > 0:
                grad_with_decay = param.grad + self.weight_decay * param.data
            else:
                grad_with_decay = param.grad
            G += grad_with_decay**2
            adaptive_lr = self.lr / (np.sqrt(G) + self.eps)
            param.data -= adaptive_lr * grad_with_decay