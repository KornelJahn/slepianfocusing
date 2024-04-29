# math_ext_test.py

from slepianfocusing.math_ext import (
    minus_one_to_pow,
    is_power_of_2,
    next_power_of_2,
    log2_next_power_of_2,
)


def test_minus_one_to_pow():
    assert minus_one_to_pow(-3) == -1
    assert minus_one_to_pow(-2) == 1
    assert minus_one_to_pow(-1) == -1
    assert minus_one_to_pow(0) == 1
    assert minus_one_to_pow(1) == -1
    assert minus_one_to_pow(2) == 1
    assert minus_one_to_pow(3) == -1


def test_is_power_of_2():
    assert is_power_of_2(1)
    assert is_power_of_2(2)
    assert is_power_of_2(4)
    assert is_power_of_2(8)
    assert is_power_of_2(1024)
    assert not is_power_of_2(-1)
    assert not is_power_of_2(0)
    assert not is_power_of_2(3)
    assert not is_power_of_2(1023)


def test_next_power_of_2():
    assert next_power_of_2(0) == 1
    assert next_power_of_2(1) == 1
    assert next_power_of_2(2) == 2
    assert next_power_of_2(3) == 4
    assert next_power_of_2(9) == 16


def test_log2_next_power_of_2():
    assert log2_next_power_of_2(0) == 0
    assert log2_next_power_of_2(1) == 0
    assert log2_next_power_of_2(2) == 1
    assert log2_next_power_of_2(3) == 2
    assert log2_next_power_of_2(9) == 4
