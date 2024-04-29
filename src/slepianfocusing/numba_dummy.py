# numba_dummy.py

"""Dummy Numba module to replace Numba if it is not installed in order to avoid
ImportErrors.
"""

__author__ = "Kornel JAHN (kornel.jahn@gmail.com)"
__copyright__ = "Copyright (c) 2012-2024 Kornel Jahn"
__license__ = "MIT"

import types


def jit(*args, **kwargs):
    """Dummy version of numba.jit decorator, does nothing.

    Based on:
    https://stackoverflow.com/a/14412901
    """
    if (
        len(args) == 1
        and len(kwargs) == 0
        and isinstance(args[0], types.FunctionType)
    ):
        # called as @decorator
        return args[0]
    else:
        # called as @decorator(*args, **kwargs)
        return jit
