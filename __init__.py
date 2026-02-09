"""
Optimizers - A pure NumPy implementation of optimization algorithms.
"""

from .core.parameter import Parameter
from .core.base_optimizer import BaseOptimizer

from .optimizers.sgd import SGD
from .optimizers.GD import GradientDescent
from .optimizers.momentum import Momentum
from .optimizers.nesterov_momentum import NAG
from .optimizers.adagrad import Adagrad
from .optimizers.rmsprop import RMSProp
from .optimizers.adam import Adam

__version__ = "1.0.0"
__author__ = "menataoff"

__all__ = [
    # Core
    'Parameter',
    'BaseOptimizer',

    # Optimizers
    'SGD',
    'GradientDescent',
    'Momentum',
    'NAG',
    'Adagrad',
    'RMSProp',
    'Adam',
]