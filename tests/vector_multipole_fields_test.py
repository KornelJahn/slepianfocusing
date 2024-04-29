# vector_multipole_fields_test.py

import numpy as np
from slepianfocusing.vector_spherical_harmonics import vsh_coeffs_YZ_to_Qpm
from slepianfocusing.vector_multipole_fields import (
    vmf_series,
    vmf,
    vmf_M,
    vmf_N,
)

EPS = np.finfo(np.float64).eps


def test_vmf_MN():
    rlambda, theta, phi = 0.5, 1.0, 1.0
    coords = rlambda, np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    tols = dict(atol=10 * EPS, rtol=10 * EPS)

    expected = np.array(
        [
            [0.97854089363875461, -0.62831388219522561, 0],
            [1.1289434414897966j, -0.72488624758808951j, 0],
            [-0.21869821118203241, 0.14042450651802988, 0],
        ]
    )
    assert np.allclose(vmf_M(1, 0, *coords), expected[0], **tols)
    assert np.allclose(vmf_M(2, 0, *coords), expected[1], **tols)
    assert np.allclose(vmf_M(3, 0, *coords), expected[2], **tols)

    expected = np.array(
        [
            [
                0.32417898511234222j,
                0.50487885558474833j,
                -0.054644094277919512j,
            ],
            [0.21385475530225851, 0.33305904786192132, -0.40047615369013103],
            [0.39067265023328788j, 0.60843660328472534j, 0.19181938851271657j],
        ]
    )
    assert np.allclose(vmf_N(1, 0, *coords), expected[0], **tols)
    assert np.allclose(vmf_N(2, 0, *coords), expected[1], **tols)
    assert np.allclose(vmf_N(3, 0, *coords), expected[2], **tols)

    expected = np.array(
        [
            [
                0.52798612766822563j,
                0.52798612766822563,
                -0.69193290156030753 - 0.44428500681388960j,
            ],
            [
                0.25231550946143974 + 0.38782491427110416j,
                -0.38782491427110416 + 0.10266649410916230j,
                0.29593357135860209 - 0.46088923001859962j,
            ],
            [
                -0.33740870235872113 + 0.47730887176774450j,
                0.16847358868500471 - 0.33740870235872113j,
                0.063132735548618017 + 0.040537063319502455j,
            ],
        ]
    )
    assert np.allclose(vmf_M(1, -1, *coords), expected[0], **tols)
    assert np.allclose(vmf_M(2, -1, *coords), expected[1], **tols)
    assert np.allclose(vmf_M(3, -1, *coords), expected[2], **tols)

    expected = np.array(
        [
            [
                0.30040788684326112 - 0.11816433399537140j,
                0.15680354361108530 + 0.30040788684326112j,
                0.35700326246167916 + 0.22922915869111002j,
            ],
            [
                -0.41372807331058903 + 0.22812992030472066j,
                -0.22812992030472066 + 0.62253883611671974j,
                -0.39895780093385148 + 0.62133996098571307j,
            ],
            [
                0.076493770449431857 - 0.029534632174251556j,
                0.040481271641116605 + 0.076493770449431857j,
                -0.56374142734766779 - 0.36197420779621745j,
            ],
        ]
    )
    assert np.allclose(vmf_N(1, -1, *coords), expected[0], **tols)
    assert np.allclose(vmf_N(2, -1, *coords), expected[1], **tols)
    assert np.allclose(vmf_N(3, -1, *coords), expected[2], **tols)

    expected = np.array(
        [
            [
                0.52798612766822563j,
                -0.52798612766822563,
                0.69193290156030753 - 0.44428500681388960j,
            ],
            [
                0.25231550946143974 - 0.38782491427110416j,
                -0.38782491427110416 - 0.10266649410916230j,
                0.29593357135860209 + 0.46088923001859962j,
            ],
            [
                0.33740870235872113 + 0.47730887176774450j,
                -0.16847358868500471 - 0.33740870235872113j,
                -0.063132735548618017 + 0.040537063319502455j,
            ],
        ]
    )
    assert np.allclose(vmf_M(1, 1, *coords), expected[0], **tols)
    assert np.allclose(vmf_M(2, 1, *coords), expected[1], **tols)
    assert np.allclose(vmf_M(3, 1, *coords), expected[2], **tols)

    expected = np.array(
        [
            [
                0.30040788684326112 + 0.11816433399537140j,
                0.15680354361108530 - 0.30040788684326112j,
                0.35700326246167916 - 0.22922915869111002j,
            ],
            [
                0.41372807331058903 + 0.22812992030472066j,
                0.22812992030472066 + 0.62253883611671974j,
                0.39895780093385148 + 0.62133996098571307j,
            ],
            [
                0.076493770449431857 + 0.029534632174251556j,
                0.040481271641116605 - 0.076493770449431857j,
                -0.56374142734766779 + 0.36197420779621745j,
            ],
        ]
    )
    assert np.allclose(vmf_N(1, 1, *coords), expected[0], **tols)
    assert np.allclose(vmf_N(2, 1, *coords), expected[1], **tols)
    assert np.allclose(vmf_N(3, 1, *coords), expected[2], **tols)


def test_vmf_MN_on_axis():
    rlambda, theta, phi = 0.75, 0.0, 0.0
    coords = rlambda, np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    tols = dict(atol=10 * EPS, rtol=10 * EPS)

    expected = np.zeros((3, 3), dtype=np.complex128)
    assert np.allclose(vmf_M(1, 0, *coords), expected[0], **tols)
    assert np.allclose(vmf_M(2, 0, *coords), expected[1], **tols)
    assert np.allclose(vmf_M(3, 0, *coords), expected[2], **tols)

    expected = np.array(
        [
            [0, 0, -0.082976891652594909j],
            [0, 0, -0.75622624933067354],
            [0, 0, -1.6531112688572659j],
        ]
    )
    assert np.allclose(vmf_N(1, 0, *coords), expected[0], **tols)
    assert np.allclose(vmf_N(2, 0, *coords), expected[1], **tols)
    assert np.allclose(vmf_N(3, 0, *coords), expected[2], **tols)

    expected = np.array(
        [
            [-0.13824623106927348j, 0.13824623106927348, 0],
            [-0.72742338573224913, -0.72742338573224913j, 0],
            [-1.1244045631731488j, 1.1244045631731488, 0],
        ]
    )
    assert np.allclose(vmf_M(1, 1, *coords), expected[0], **tols)
    assert np.allclose(vmf_M(2, 1, *coords), expected[1], **tols)
    assert np.allclose(vmf_M(3, 1, *coords), expected[2], **tols)

    expected = np.array(
        [
            [0.62213325448589425j, -0.62213325448589425, 0],
            [-0.48720319036222603, -0.48720319036222603j, 0],
            [0.14488078016052685j, -0.14488078016052685, 0],
        ]
    )
    assert np.allclose(vmf_N(1, 1, *coords), expected[0], **tols)
    assert np.allclose(vmf_N(2, 1, *coords), expected[1], **tols)
    assert np.allclose(vmf_N(3, 1, *coords), expected[2], **tols)

    rlambda, theta, phi = 0.75, np.pi, 0.0
    coords = rlambda, np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)

    expected = np.array(
        [
            [0.13824623106927348j, 0.13824623106927348, 0],
            [-0.72742338573224913, 0.72742338573224913j, 0],
            [1.1244045631731488j, 1.1244045631731488, 0],
        ]
    )
    assert np.allclose(vmf_M(1, -1, *coords), expected[0], **tols)
    assert np.allclose(vmf_M(2, -1, *coords), expected[1], **tols)
    assert np.allclose(vmf_M(3, -1, *coords), expected[2], **tols)

    expected = np.array(
        [
            [-0.62213325448589425j, -0.62213325448589425, 0],
            [-0.48720319036222603, 0.48720319036222603j, 0],
            [-0.14488078016052685j, -0.14488078016052685, 0],
        ]
    )
    assert np.allclose(vmf_N(1, -1, *coords), expected[0], **tols)
    assert np.allclose(vmf_N(2, -1, *coords), expected[1], **tols)
    assert np.allclose(vmf_N(3, -1, *coords), expected[2], **tols)


def test_vmf_MN_origin():
    rlambda, theta, phi = 0.0, 0.0, 0.0
    coords = rlambda, np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    tols = dict(atol=10 * EPS, rtol=10 * EPS)

    expected = np.zeros((3, 3), dtype=np.complex128)
    assert np.allclose(vmf_M(1, 0, *coords), expected[0], **tols)
    assert np.allclose(vmf_M(2, 0, *coords), expected[1], **tols)
    assert np.allclose(vmf_M(3, 0, *coords), expected[2], **tols)

    expected = np.array([[0, 0, 2.8944050182330706j], [0, 0, 0], [0, 0, 0]])
    assert np.allclose(vmf_N(1, 0, *coords), expected[0], **tols)
    assert np.allclose(vmf_N(2, 0, *coords), expected[1], **tols)
    assert np.allclose(vmf_N(3, 0, *coords), expected[2], **tols)

    expected = np.zeros((3, 3), dtype=np.complex128)
    assert np.allclose(vmf_M(1, 1, *coords), expected[0], **tols)
    assert np.allclose(vmf_M(2, 1, *coords), expected[1], **tols)
    assert np.allclose(vmf_M(3, 1, *coords), expected[2], **tols)

    expected = np.array(
        [[-2.0466534158929770j, 2.0466534158929770, 0], [0, 0, 0], [0, 0, 0]]
    )
    assert np.allclose(vmf_N(1, 1, *coords), expected[0], **tols)
    assert np.allclose(vmf_N(2, 1, *coords), expected[1], **tols)
    assert np.allclose(vmf_N(3, 1, *coords), expected[2], **tols)

    expected = np.zeros((3, 3), dtype=np.complex128)
    assert np.allclose(vmf_M(1, -1, *coords), expected[0], **tols)
    assert np.allclose(vmf_M(2, -1, *coords), expected[1], **tols)
    assert np.allclose(vmf_M(3, -1, *coords), expected[2], **tols)

    expected = np.array(
        [[2.0466534158929770j, 2.0466534158929770, 0], [0, 0, 0], [0, 0, 0]]
    )
    assert np.allclose(vmf_N(1, -1, *coords), expected[0], **tols)
    assert np.allclose(vmf_N(2, -1, *coords), expected[1], **tols)
    assert np.allclose(vmf_N(3, -1, *coords), expected[2], **tols)


def test_vmf_T_series():
    rlambda, theta, phi = 0.5, 1.0, 1.0
    coords = rlambda, np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)

    c = np.array([0.0, 1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex128)
    S = 0.0
    coeffs_Tp, coeffs_Tm = vsh_coeffs_YZ_to_Qpm(-1, c, -c)
    S += vmf_series(1, -1, coeffs_Tp, *coords)
    S += vmf_series(-1, -1, coeffs_Tm, *coords)
    coeffs_Tp, coeffs_Tm = vsh_coeffs_YZ_to_Qpm(0, c.conj(), -c.conj())
    S += vmf_series(1, 0, coeffs_Tp, *coords)
    S += vmf_series(-1, 0, coeffs_Tm, *coords)
    coeffs_Tp, coeffs_Tm = vsh_coeffs_YZ_to_Qpm(1, -c, c.conj())
    S += vmf_series(1, 1, coeffs_Tp, *coords)
    S += vmf_series(-1, 1, coeffs_Tm, *coords)

    expected = np.array(
        [
            -0.35189320011997697 - 0.25244129473552440j,
            0.4057032177155100 - 2.0455960518189797j,
            5.1998168140437816 + 0.2114362800139354j,
        ]
    )
    assert np.allclose(S, expected, atol=10 * EPS, rtol=10 * EPS)
