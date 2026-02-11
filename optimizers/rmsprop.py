from ..core import BaseOptimizer, Parameter
from typing import Union, Iterable
import numpy as np

class RMSProp(BaseOptimizer):
    def __init__(self,
                 params: Iterable[Union[np.ndarray, Parameter]],
                 lr: float = 0.01,
                 weight_decay: float = 0.0,
                 gamma: float = 0.95,
                 eps: float = 1e-8) -> None:

        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")

        if not 0 <= gamma < 1:
            raise ValueError(f"gamma must be in [0, 1), got {gamma}")

        self.eps = eps
        self.gamma = gamma
        super().__init__(params, lr, weight_decay)

        self._Gs = []
        for param in self._params:
            self._Gs.append(np.zeros_like(param.data))

    def __repr__(self):
        return (f"RMSProp(lr={self.lr}, gamma={self.gamma}, "
                f"weight_decay={self.weight_decay}, eps={self.eps:.1e}, "
                f"params={len(self._params)})")

    def _update_parameters(self):
        for param, G in zip(self._params, self._Gs):
            if self.weight_decay > 0:
                grad_with_decay = param.grad + self.weight_decay * param.data
            else:
                grad_with_decay = param.grad
            G *= self.gamma
            G += (1-self.gamma)*grad_with_decay*grad_with_decay
            adaptive_lr = self.lr / np.sqrt(G + self.eps)
            param.data -= adaptive_lr * grad_with_decay