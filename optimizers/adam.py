from core import BaseOptimizer, Parameter
from typing import Union, Iterable
import numpy as np

class Adam(BaseOptimizer):
    def __init__(self,
                 params: Iterable[Union[np.ndarray, Parameter]],
                 lr: float = 0.001,
                 weight_decay: float = 0.0,
                 beta1: float = 0.9,
                 beta2: float = 0.99,
                 eps: float = 1e-8) -> None:

        if not 0 <= beta1 < 1:
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")

        if not 0 <= beta2 < 1:
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")

        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")



        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        super().__init__(params, lr, weight_decay)

        self._Gs = []
        self._velocities = []
        for param in self._params:
            self._Gs.append(np.zeros_like(param.data))
            self._velocities.append(np.zeros_like(param.data))

    def __repr__(self):
        return (f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, "
                f"weight_decay={self.weight_decay}, eps={self.eps:.1e}, "
                f"params={len(self._params)})")

    def _update_parameters(self):
        for param, velocity, G in zip(self._params, self._velocities, self._Gs):
            if self.weight_decay > 0:
                grad_with_decay = param.grad + self.weight_decay * param.data
            else:
                grad_with_decay = param.grad

            velocity *= self.beta1
            velocity += (1 - self.beta1) * grad_with_decay

            G *= self.beta2
            G += (1-self.beta2)*grad_with_decay*grad_with_decay

            adaptive_lr = self.lr / (np.sqrt(G) + self.eps)
            param.data -= adaptive_lr * velocity