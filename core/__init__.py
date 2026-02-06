"""
Module core - base classes for optimizers
"""

from .base_optimizer import BaseOptimizer
from .parameter import Parameter

__all__ = ['BaseOptimizer', 'Parameter']

# Опционально: информация о версии
__version__ = '0.1.0'