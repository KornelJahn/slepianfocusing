# sphcap_conc_problems_test.py

import numpy as np
from slepianfocusing.sphcap_conc_problems import (
    ScalarConcProblem,
    VectorRelatedScalarConcProblem,
    TangentialVectorConcProblem,
)

EPS = np.finfo(np.float64).eps


def test_eigenvectors_scalar():
    tol = dict(atol=1e2 * EPS, rtol=1e2 * EPS)
    L = 100
    m_values = [0, 1]
    lmin_values = [0, 1]
    conc = ScalarConcProblem(L, m_values, np.pi / 3.0)
    for m, lmin in zip(m_values, lmin_values):
        ev = conc.eigenvectors[m][lmin:]
        assert np.allclose(np.eye(ev.shape[0]), np.dot(ev, ev.T), **tol)


def test_eigenvectors_vector_related_scalar():
    tol = dict(atol=1e2 * EPS, rtol=1e2 * EPS)
    L = 100
    m_values = [0, 1]
    lmin_values = [1, 1]
    conc = VectorRelatedScalarConcProblem(L, m_values, np.pi / 3.0)
    for m, lmin in zip(m_values, lmin_values):
        ev = conc.eigenvectors[m][lmin:]
        assert np.allclose(np.eye(ev.shape[0]), np.dot(ev, ev.T), **tol)


def test_eigenvectors_vector():
    tol = dict(atol=1e2 * EPS, rtol=1e2 * EPS)
    L = 100
    m_values = [0, 1]
    lmin_values = [1, 1]
    conc = TangentialVectorConcProblem(L, m_values, np.pi / 3.0)
    for m, lmin in zip(m_values, lmin_values):
        ev = conc.eigenvectors[m][lmin:]
        assert np.allclose(np.eye(ev.shape[0]), np.dot(ev, ev.T), **tol)


def test_vector_m0_scalar_m1_conc_problem_equivalence():
    tol = dict(atol=1e2 * EPS, rtol=1e2 * EPS)
    L = 100
    Theta = np.pi / 3.0
    conc_scalar = ScalarConcProblem(L, 1, Theta)
    conc_vector = VectorRelatedScalarConcProblem(L, 0, Theta)
    ev_scalar = conc_scalar.eigenvectors[1]
    ev_vector = conc_vector.eigenvectors[0]
    assert np.allclose(ev_vector, ev_scalar, **tol)
