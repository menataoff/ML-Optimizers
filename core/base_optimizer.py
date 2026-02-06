import numpy as np
from typing import List, Dict, Any, Union, Iterable
from .parameter import Parameter

class BaseOptimizer:
    def __init__(self,
                 params: Iterable[Union[np.ndarray, list, tuple, Parameter]],
                 lr: float = 0.01):

        if params is None:
            raise ValueError("parameters cannot be None")

        temp_list = list(params)

        if len(temp_list) == 0:
            raise ValueError("Empty parameters list")

        self._params = []

        for i, param in enumerate(temp_list):
            if isinstance(param, Parameter):
                self._params.append(param)
            else:
                try:
                    self._params.append(Parameter(param))
                except Exception as e:
                    raise TypeError(
                        f"Cannot convert parameter {i} to Parameter. "
                        f"Type: {type(param).__name__}, Error: {e}"
                    )

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
            param.zero_grad()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  lr: {self.lr}\n"
            f"  parameters: {len(self._params)} items\n"
            f"  step_count: {self._step_count}\n"
            f")"
        )