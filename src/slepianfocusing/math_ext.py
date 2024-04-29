# math_ext.py

"""Additional mathematical helper functions"""

__author__ = "Kornel JAHN (kornel.jahn@gmail.com)"
__copyright__ = "Copyright (c) 2012-2024 Kornel Jahn"
__license__ = "MIT"

import numpy as np

try:
    from numba import jit
except ImportError:
    from .numba_dummy import jit


@jit(nopython=True, cache=True, nogil=True)
def minus_one_to_pow(n):
    return 1 - 2 * (n & 1)


@jit(nopython=True, cache=True, nogil=True)
def is_power_of_2(n):
    return n > 0 and ((n & (n - 1)) == 0)


@jit(nopython=True, cache=True, nogil=True)
def sign(x):
    return (x > 0) - (x < 0)


def next_power_of_2(n):
    if n > 1:
        return 1 << ((n - 1).bit_length())
    else:
        return 1


def log2_next_power_of_2(n):
    if n > 1:
        return (n - 1).bit_length()
    else:
        return 0
