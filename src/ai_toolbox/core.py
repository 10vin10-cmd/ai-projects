"""Core utilities for ai_toolbox.

This module contains small, well-tested helper functions intended as
examples and building blocks for larger tools.
"""
from typing import Union

Number = Union[int, float]

def add(a: Number, b: Number) -> Number:
    """Return the sum of two numbers.

    This is intentionally simple so tests and packaging examples stay small.
    """
    return a + b
