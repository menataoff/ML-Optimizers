import numpy as np
from typing import Union, Optional

class Parameter:
    def __init__(self,
                 data: Union[np.ndarray, list, tuple, 'Parameter'],
                 dtype: type = np.float32,
                 name: Optional[str] = None):
        if isinstance(data, Parameter):
            self.data = data.data.copy().astype(dtype, copy=False)
            self.grad = data.grad.copy() if data.grad is not None else None

        elif isinstance(data, np.ndarray):
            self.data = data.astype(dtype, copy=False)
            self.grad = None

        elif isinstance(data, (list, tuple)):
            self.data = np.array(data, dtype=dtype, copy=True)
            self.grad = None

        else:
            raise TypeError(
                f"Unsupported data type: {type(data)}. "
                f"Expected: np.ndarray, list, tuple or Parameter."
            )

        if self.data.size == 0:
            raise ValueError("Parameter data cannot be empty")

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

    def __len__(self) -> int:
        return len(self.data)
