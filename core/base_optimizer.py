import numpy as np

class BaseOptimizer:
    def __init__(self, params, lr = 0.01):

        if params is None:
            raise ValueError("parameters cannot be None")
        self._params = list(params)
        if len(self._params) == 0:
            raise ValueError("Empty parameters list")

        if (lr <= 0.0):
            raise ValueError(f"Learning rate must be positive, got {lr}")
        self.lr = lr

        self._step_count = 0

    @property
    def params(self):
        """Getter for parameters"""
        return self._params

    @property
    def step_count(self):
        """Getter for step count"""
        return self._step_count

    def step(self):
        for i, param in enumerate(self._params):
            if not hasattr(param, "grad"):
                raise AttributeError(f"Parameter {i} has no gradient attribute")

            if param.grad is None:
                raise ValueError(f"Gradient for parameter {i} is None")
        self._step_count += 1
        self._update_parameters()

    def _update_parameters(self):
        """Internal method to update parameters. Must be overridden."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _update_parameters()"
        )

    def zero_grad(self):
        for param in self._params:
            if hasattr(param, "grad") and param.grad is not None:
                param.grad.fill(0)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  lr: {self.lr}\n"
            f"  parameters: {len(self._params)} items\n"
            f"  step_count: {self._step_count}\n"
            f")"
        )