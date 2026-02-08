import numpy as np
from typing import Union, Optional

class Parameter:
    __slots__ = ('data', 'grad', 'name')

    def __init__(self,
                 data: np.ndarray,
                 name: Optional[str] = None) -> None:

        if not isinstance(data, np.ndarray):
            raise TypeError(f"Data must be a numpy array, got {type(data).__name__}")

        if data.dtype != np.float32:
            raise TypeError(f"Data must be a numpy array with np.float32 type, got {data.dtype}")

        if data.size == 0:
            raise ValueError("Parameter data cannot be empty")

        self.data = data
        self.grad = None
        self.name = name

    def __repr__(self) -> str:
        parts = []

        parts.append(f"shape={self.data.shape}")
        parts.append(f"dtype={self.data.dtype}")
        parts.append(f"name={self.name}")

        if (self.grad is not None):
            parts.append(f"grad_shape={self.grad.shape}")
        else:
            parts.append(f"grad=None")

        return f"Parameter({", ".join(parts)})"

    def zero_grad(self) -> None:
        if self.grad is not None:
            self.grad.fill(0)

    @property
    def shape(self) -> tuple:
        return self.data.shape
    @property
    def size(self) -> int:
        return self.data.size
    @property
    def ndim(self) -> int:
        return self.data.ndim

    def __len__(self) -> int:
        return len(self.data)
