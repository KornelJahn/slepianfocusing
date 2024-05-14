# quadrature.py

"""Quadrature rules with error estimates.

Available rules:
    - Uniform trapezoidal (for periodic functions!),
    - Clenshaw--Curtis,
    - Gauss--Legendre.
"""

__author__ = "Kornel JAHN (kornel.jahn@gmail.com)"
__copyright__ = "Copyright (c) 2012-2024 Kornel Jahn"
__license__ = "MIT"

from zipfile import ZipFile
import numpy as np
from pathlib import Path


## Auxiliary functions


def _scale_rule(x_std, w_std, wn_std, a, b):
    """Scale nodes and weights to perform quadrature on [a, b]."""
    if b < a:
        raise ValueError("b >= a is required.")

    c = 0.5 * (a + b)
    h = 0.5 * (b - a)
    x = c + h * np.hstack([-x_std[::-1], x_std[1:]])
    w = h * np.hstack([w_std[::-1], w_std[1:]])
    wn = h * np.hstack([wn_std[::-1], wn_std[1:]])
    return x, w, wn


## Quadrature rules

## Clenshaw--Curtis rules


def _calculate_cc_weights(n):
    # Based on code by Greg von Winckel
    # http://www.scientificpython.net/1/post/2012/04/clenshaw-curtis-quadrature.html
    c = np.zeros((n,))
    c[::2] = 2.0 / (1.0 - np.arange(0, n + 1, 2) ** 2)
    w = np.fft.ifft(np.r_[c, c[-2:0:-1]]).real
    w[1 : (n - 1)] *= 2.0
    return w[(n // 2) : n]


class ClenshawCurtisRule:
    def __init__(self, level):
        max_level = self.get_max_level()
        if level < 2 or level > max_level:
            raise ValueError(
                "level must satisfy 2 <= level <= %d." % max_level
            )
        self.level = level
        n = 2**level + 1
        self.node_count = n
        n_nested = 2 ** (level - 1) + 1

        self.std_nodes = np.sin(np.pi / (2 * n - 2) * np.arange(0, n, 2))
        self.std_weights = _calculate_cc_weights(n)
        self.std_weights_nested = np.zeros_like(self.std_weights)
        self.std_weights_nested[::2] = _calculate_cc_weights(n_nested)

    def get_nodes_weights(self, a=-1.0, b=1.0):
        return _scale_rule(
            self.std_nodes, self.std_weights, self.std_weights_nested, a, b
        )

    def get_poly_precisions(self):
        l = self.level
        return 2**l, 2 ** (l - 1)

    @classmethod
    def get_max_level(cls):
        return 20  # arbitrary


## Gauss--Legendre rules (with error estimation)


def _load_gl():
    """Load precomputed Gauss--Legendre quadrature rules.

    Quadrature error is estimated as proposed by Berntsen and Espelid: the
    central node is dropped (weight set to 0) and a new set of weights are
    applied to the rest of the existing set of nodes to obtain an interpolatory
    rule of order (2n - 1). The error can, e.g., be estimated as the absolute
    difference between the weighted sum using these new weights and the
    original one obtained using the Gauss--Legendre weights.

    References:

    Berntsen, J., & Espelid, T. O. (1984).
        On the use of Gauss quadrature in adaptive automatic integration
        schemes. BIT Numerical Mathematics, 24(2), 239-242.
        https://doi.org/10.1007/BF01937489
    """
    path = Path(__file__).parent / "gauss_legendre_rules.zip"
    with ZipFile(path, "r") as zipf:
        all_std_nodes = {}
        all_std_weights = {}
        all_std_weights_nested = {}
        names = list(sorted(zipf.namelist()))
        zip_ = zip(names[::3], names[1::3], names[2::3])
        for i, (nodes_fn, weights_fn, weights_nested_fn) in enumerate(zip_):
            level = i + 2
            with zipf.open(nodes_fn, "r") as f:
                all_std_nodes[level] = np.loadtxt(f)
            with zipf.open(weights_fn, "r") as f:
                all_std_weights[level] = np.loadtxt(f)
            with zipf.open(weights_nested_fn, "r") as f:
                all_std_weights_nested[level] = np.loadtxt(f)
        return all_std_nodes, all_std_weights, all_std_weights_nested


class GaussLegendreRule:
    # Load precomputed rules
    _all_std_nodes, _all_std_weights, _all_std_weights_nested = _load_gl()

    def __init__(self, level):
        max_level = self.get_max_level()
        if level < 2 or level > max_level:
            raise ValueError(f"level must satisfy 2 <= level <= {max_level}.")
        self.level = level
        n = 2**level - 1
        self.node_count = n
        # n_nested = (n - 1) // 2 + 1
        n_nested = 2 ** (level - 1)

        self.std_nodes = self._all_std_nodes[level]
        self.std_weights = self._all_std_weights[level]
        self.std_weights_nested = self._all_std_weights_nested[level]

    def get_nodes_weights(self, a=-1.0, b=1.0):
        return _scale_rule(
            self.std_nodes, self.std_weights, self.std_weights_nested, a, b
        )

    def get_poly_precisions(self):
        n = self.node_count
        return 2 * n - 1, n - 2

    @classmethod
    def get_max_level(cls):
        return max(*cls._all_std_weights.keys())


def showcase():
    test_cases = [
        # (formula, function, a, b, exact value)
        ["exp(x)", np.exp, -1, 1, np.exp(1.0) - np.exp(-1.0)],
        ["cos(128*x)", lambda x: np.cos(128 * x), 0, 1, np.sin(128.0) / 128.0],
        [
            "sqrt(cos(x))",
            lambda x: np.sqrt(np.cos(x)),
            0,
            np.pi / 2,
            1.1981402347355922e00,
        ],
        ["sqrt(1 - x^2)", lambda x: np.sqrt(1 - x**2), 0, 1, np.pi / 4],
        [
            "1 / (1 + 25*x^2)",
            lambda x: 1.0 / (1 + 25 * x**2),
            -1,
            1,
            5.4936030677800634e-01,
        ],
    ]

    rules = [(" CC", ClenshawCurtisRule), (" GL", GaussLegendreRule)]
    max_level = min(*[rule[1].get_max_level() for rule in rules])

    for test_case in test_cases:
        formula, f, a, b, I_exact = test_case
        print(f"\nFunction: {formula} on interval [{a}, {b}]\n")
        print("level\t | ", end="")
        for rule in rules:
            print(f"{rule[0]} error est. ", end="")
            print(f"{rule[0]} true error | ", end="")
        print()
        print("-" * 80)
        for level in range(2, max_level + 1):
            print(f"{level}\t | ", end="")
            for rule in rules:
                x, w, wn = rule[1](level).get_nodes_weights(a, b)
                fx = f(x)
                I_approx = np.dot(w, fx)
                I_nested = np.dot(wn, fx)
                E_exact = abs(I_approx - I_exact)
                E_approx = abs(I_approx - I_nested)
                print(f"    {E_approx:.3e}       {E_exact:.3e} | ", end="")
            print()
        print("-" * 80)


if __name__ == "__main__":
    showcase()
