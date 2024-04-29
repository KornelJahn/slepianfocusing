# quadrature_test.py

import numpy as np
from slepianfocusing.quadrature import ClenshawCurtisRule, GaussLegendreRule

EPS = np.finfo(np.float64).eps


def test_clenshaw_curtis_rules():
    _test_rule(ClenshawCurtisRule, 9)


def test_gauss_legendre_rules():
    _test_rule(GaussLegendreRule, 9)


def _test_rule(rule, max_level):
    for level in range(2, max_level + 1):
        current_rule = rule(level)
        assert _are_all_nodes_internal(current_rule)
        assert _are_all_weights_nonnegative(current_rule)
        assert _is_sum_of_weights_2(current_rule)
        assert _is_polynomial_accuracy_reached(current_rule)
    _test_integration_correctness(current_rule, np.exp, np.exp, -3.0, 4.0)
    _test_integration_correctness(current_rule, np.cos, np.sin, -90.0, 100.0)


def _are_all_nodes_internal(rule):
    x_std, _, _ = rule.get_nodes_weights()
    return np.all(np.abs(x_std) <= 1.0)


def _are_all_weights_nonnegative(rule):
    _, w_std, wn_std = rule.get_nodes_weights()
    return np.all(w_std >= 0.0) and np.all(wn_std >= 0.0)


def _is_sum_of_weights_2(rule):
    _, w_std, wn_std = rule.get_nodes_weights()
    w_error = abs(w_std.sum() - 2.0)
    wn_error = abs(wn_std.sum() - 2.0)
    abs_tol = 100 * EPS
    return w_error < abs_tol and wn_error < abs_tol


def _is_polynomial_accuracy_reached(rule):
    x_std, w_std, wn_std = rule.get_nodes_weights()

    def calculate_moment_errors(p, w):
        e = np.arange(p + 1)
        exact_moments = ((e + 1) % 2) * 2.0 / (e + 1)
        integrated_moments = np.dot(x_std ** (e[:, np.newaxis]), w)
        return abs(integrated_moments - exact_moments)

    abs_tol = 100 * EPS
    prec, nested_prec = rule.get_poly_precisions()
    moment_errors = calculate_moment_errors(prec, w_std)
    are_errors_small_for_w = np.all(moment_errors < abs_tol)
    moment_errors = calculate_moment_errors(nested_prec, wn_std)
    are_errors_small_for_wn = np.all(moment_errors < abs_tol)
    return are_errors_small_for_w and are_errors_small_for_wn


# def _is_integral_correct(rule, func, antiderivative, a, b):
#     x, w, wn = rule.get_nodes_weights(a, b)
#     fx = func(x)
#     I = np.dot(w, fx)
#     In = np.dot(wn, fx)
#     error_estimate = abs(I - In)
#     Iref = antiderivative(b) - antiderivative(a)
#     true_error = abs(I - Iref)
#     is_value_ok = error_estimate < 100 * EPS
#     is_error_estimate_ok = true_error <= (10.0 * error_estimate)
#     return is_value_ok and is_error_estimate_ok

def _test_integration_correctness(rule, func, antiderivative, a, b):
    x, w, wn = rule.get_nodes_weights(a, b)
    fx = func(x)
    I = np.dot(w, fx)
    In = np.dot(wn, fx)
    error_estimate = abs(I - In)
    Iref = antiderivative(b) - antiderivative(a)
    true_error = abs(I - Iref)
    atol = 100 * EPS
    rtol = 100 * EPS
    tol = max(atol, rtol * abs(I))
    # Test value
    assert error_estimate < tol
    # Test error estimate
    assert true_error <= max(10 * error_estimate, tol)
