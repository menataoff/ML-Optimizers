## ML Optimizers

A pure NumPy implementation of gradient-based optimization algorithms for machine learning. Clean, minimal, and focused on educational value.

## Overview

This library provides efficient implementations of popular ***optimization algorithms*** using only NumPy. Designed for clarity and understanding, it's suitable for educational purposes and as a reference implementation.

## Algorithms

- **SGD** - Stochastic Gradient Descent
- **Gradient Descent** - Gradient Descent
- **Momentum** - SGD with momentum acceleration  
- **NAG** - Nesterov Momentum
- **Adagrad** - Adaptive learning rates per parameter
- **RMSProp** - Exponential moving average of squared gradients
- **Adam** - Adaptive Moment Estimation (combines Momentum and RMSProp)

## Installation

```bash
git clone https://github.com/menataoff/ML-Optimizers.git
cd ML-Optimizers
pip install numpy
```

## Quick start

```python
import numpy as np
from optimizers import Momentum

# Create a parameter to optimize
w = np.array([5.0], dtype=np.float32)

# Initialize optimizer
opt = Momentum([w], lr=0.1)

def get_gradient(w):
    """
    For example, we want to find the minimum of f(x) = x**2, so grad(f) = 2*x.
    """
    return np.array([2*w[0]])

# Optimization loop
for _ in range(128):
    # User computes gradient (e.g., via backpropagation)
    opt.params[0].grad = get_gradient(w)
    
    # Optimizer updates parameters
    opt.step()
    opt.zero_grad()

print(f'Updated weight: {w}') #Should be close to 0.0
```

## Key features

- Pure NumPy - Zero dependencies beyond NumPy
- Weights modified directly, memory efficient
- L2 regularization - Built-in weight decay for all optimizers
- Float32 only - ML standard precision
- Educational focus - сlear code with visible mathematical formulas

## Examples

See the `examples/` directory for:
- `simple_optimization.py` - Basic usage with quadratic functions
- `comparison.ipynb` - Algorithm comparison on non-convex landscapes

## License
MIT License. See [LICENSE.md](LICENSE.md) for details.