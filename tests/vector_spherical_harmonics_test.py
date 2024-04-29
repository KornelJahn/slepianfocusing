# vector_spherical_harmonics_test.py

import numpy as np
from slepianfocusing.vector_spherical_harmonics import (
    vsh_series,
    vsh,
    vsh_Y,
    vsh_Z,
    vsh_coeffs_YZ_to_Qpm,
)

EPS = np.finfo(np.float64).eps


def test_vsh_Qpm():
    theta, phi = 1.0, 1.0
    angular_coords = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    tols = dict(atol=10 * EPS, rtol=10 * EPS)

    expected = np.array(
        [
            [
                0.06001205341104668 - 0.17298322539007688j,
                0.09346323555476667 + 0.11107125170347240j,
                -0.17298322539007688,
            ],
            [
                0.07250372342017391 - 0.20899014809753355j,
                0.11291785892082139 + 0.13419103089644850j,
                -0.20899014809753355,
            ],
            [
                0.025802015347352044 - 0.074373656334972862j,
                0.040184258013630414 + 0.047754775552723631j,
                -0.074373656334972862,
            ],
        ]
    )
    assert np.allclose(vsh(1, 1, 0, *angular_coords), expected[0], **tols)
    assert np.allclose(vsh(1, 2, 0, *angular_coords), expected[1], **tols)
    assert np.allclose(vsh(1, 3, 0, *angular_coords), expected[2], **tols)

    expected = np.array(
        [
            [
                0.06001205341104668 + 0.17298322539007688j,
                0.09346323555476667 - 0.11107125170347240j,
                -0.17298322539007688,
            ],
            [
                0.07250372342017391 + 0.20899014809753355j,
                0.11291785892082139 - 0.13419103089644850j,
                -0.20899014809753355,
            ],
            [
                0.025802015347352044 + 0.074373656334972862j,
                0.040184258013630414 - 0.047754775552723631j,
                -0.074373656334972862,
            ],
        ]
    )
    assert np.allclose(vsh(-1, 1, 0, *angular_coords), expected[0], **tols)
    assert np.allclose(vsh(-1, 2, 0, *angular_coords), expected[1], **tols)
    assert np.allclose(vsh(-1, 3, 0, *angular_coords), expected[2], **tols)

    expected = np.array(
        [
            [
                -0.043703669654074776 - 0.055611544789338915j,
                0.055611544789338915 - 0.007198417510897907j,
                -0.036104305354522815 + 0.056229124052433180j,
            ],
            [
                -0.11739019833523075 - 0.14937533447927278j,
                0.14937533447927278 - 0.01933530218383655j,
                -0.09697793342916694 + 0.15103418264365346j,
            ],
            [
                -0.14656688940048570 - 0.18650175600916022j,
                0.18650175600916022 - 0.02414098567761629j,
                -0.12108126781258356 + 0.18857290180232661j,
            ],
        ]
    )
    assert np.allclose(vsh(1, 1, -1, *angular_coords), expected[0], **tols)
    assert np.allclose(vsh(1, 2, -1, *angular_coords), expected[1], **tols)
    assert np.allclose(vsh(1, 3, -1, *angular_coords), expected[2], **tols)

    expected = np.array(
        [
            [
                0.23037495527738906 + 0.05561154478933892j,
                -0.05561154478933892 - 0.17947286811241638j,
                -0.12097416519428359 + 0.18840609935725547j,
            ],
            [
                0.023972842333650212 + 0.005786943261954469j,
                -0.005786943261954469 - 0.018675965732674579j,
                -0.012588584488950112 + 0.019605558725561789j,
            ],
            [
                -0.17807413809232965 - 0.04298634760198031j,
                0.04298634760198031 + 0.13872808466352277j,
                0.09351003529212841 - 0.14563325129671331j,
            ],
        ]
    )
    assert np.allclose(vsh(-1, 1, -1, *angular_coords), expected[0], **tols)
    assert np.allclose(vsh(-1, 2, -1, *angular_coords), expected[1], **tols)
    assert np.allclose(vsh(-1, 3, -1, *angular_coords), expected[2], **tols)


def test_vsh_YZ():
    theta, phi = 1.0, 1.0
    angular_coords = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    tols = dict(atol=10 * EPS, rtol=10 * EPS)

    expected = np.array(
        [
            [-0.24463522340968865 * 1j, 0.15707847054880640 * 1j, 0],
            [-0.29555670184189363 * 1j, 0.18977477584258450 * 1j, 0],
            [-0.10518023347219428 * 1j, 0.067535451254744876 * 1j, 0],
        ]
    )
    assert np.allclose(vsh_Y(1, 0, *angular_coords), expected[0], **tols)
    assert np.allclose(vsh_Y(2, 0, *angular_coords), expected[1], **tols)
    assert np.allclose(vsh_Y(3, 0, *angular_coords), expected[2], **tols)

    expected = np.array(
        [
            [
                -0.084869859839760774 * 1j,
                -0.13217697530482229 * 1j,
                0.24463522340968865 * 1j,
            ],
            [
                -0.10253574898335774 * 1j,
                -0.15968996751995739 * 1j,
                0.29555670184189363 * 1j,
            ],
            [
                -0.036489560040784006 * 1j,
                -0.056829122676775861 * 1j,
                0.10518023347219428 * 1j,
            ],
        ]
    )
    assert np.allclose(vsh_Z(1, 0, *angular_coords), expected[0], **tols)
    assert np.allclose(vsh_Z(2, 0, *angular_coords), expected[1], **tols)
    assert np.allclose(vsh_Z(3, 0, *angular_coords), expected[2], **tols)

    expected = np.array(
        [
            [
                0.131996531917056408,
                -0.131996531917056408 * 1j,
                -0.11107125170347240 + 0.17298322539007688 * 1j,
            ],
            [
                -0.066056045909235422 - 0.101532325129432683 * 1j,
                0.101532325129432683 - 0.026878025305263672 * 1j,
                -0.07747522781089793 + 0.12066051826209075 * 1j,
            ],
            [
                -0.22955587199153814 - 0.16227259426507849 * 1j,
                0.16227259426507849 + 0.08102531472942863 * 1j,
                -0.019495805480884902 + 0.030362918054299527 * 1j,
            ],
        ]
    )
    assert np.allclose(vsh_Y(1, -1, *angular_coords), expected[0], **tols)
    assert np.allclose(vsh_Y(2, -1, *angular_coords), expected[1], **tols)
    assert np.allclose(vsh_Y(3, -1, *angular_coords), expected[2], **tols)

    expected = np.array(
        [
            [
                -0.07864660086560192 + 0.19380285426732243 * 1j,
                0.12181643224552061 - 0.07864660086560192 * 1j,
                -0.093463235554766672 - 0.060012053411046679 * 1j,
            ],
            [
                -0.10971629877517229 + 0.09995876466611543 * 1j,
                -0.000466221275700106 - 0.109716298775172292 * 1j,
                0.092934071214499093 + 0.059672280895545105 * 1j,
            ],
            [
                -0.101480718489473771 - 0.022278989206533839 * 1j,
                -0.11516582408376824 - 0.10148071848947377 * 1j,
                0.23631943717060063 + 0.15173896560899966 * 1j,
            ],
        ]
    )
    assert np.allclose(vsh_Z(1, -1, *angular_coords), expected[0], **tols)
    assert np.allclose(vsh_Z(2, -1, *angular_coords), expected[1], **tols)
    assert np.allclose(vsh_Z(3, -1, *angular_coords), expected[2], **tols)

    expected = np.array(
        [
            [
                0.131996531917056408,
                0.131996531917056408 * 1j,
                -0.11107125170347240 - 0.17298322539007688 * 1j,
            ],
            [
                -0.066056045909235422 + 0.101532325129432683 * 1j,
                0.101532325129432683 + 0.026878025305263672 * 1j,
                -0.07747522781089793 - 0.12066051826209075 * 1j,
            ],
            [
                -0.22955587199153814 + 0.16227259426507849 * 1j,
                0.16227259426507849 - 0.08102531472942863 * 1j,
                -0.019495805480884902 - 0.030362918054299527 * 1j,
            ],
        ]
    )
    assert np.allclose(vsh_Y(1, 1, *angular_coords), expected[0], **tols)
    assert np.allclose(vsh_Y(2, 1, *angular_coords), expected[1], **tols)
    assert np.allclose(vsh_Y(3, 1, *angular_coords), expected[2], **tols)

    expected = np.array(
        [
            [
                -0.07864660086560192 - 0.19380285426732243 * 1j,
                0.12181643224552061 + 0.07864660086560192 * 1j,
                -0.093463235554766672 + 0.060012053411046679 * 1j,
            ],
            [
                -0.10971629877517229 - 0.09995876466611543 * 1j,
                -0.000466221275700106 + 0.109716298775172292 * 1j,
                0.092934071214499093 - 0.059672280895545105 * 1j,
            ],
            [
                -0.101480718489473771 + 0.022278989206533839 * 1j,
                -0.11516582408376824 + 0.10148071848947377 * 1j,
                0.23631943717060063 - 0.15173896560899966 * 1j,
            ],
        ]
    )
    assert np.allclose(vsh_Z(1, 1, *angular_coords), expected[0], **tols)
    assert np.allclose(vsh_Z(2, 1, *angular_coords), expected[1], **tols)
    assert np.allclose(vsh_Z(3, 1, *angular_coords), expected[2], **tols)


def test_vsh_Q_series():
    theta, phi = 1.0, 1.0
    angular_coords = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)

    c = np.array([0.0, 1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex128)
    S = 0.0
    coeffs_Qp, coeffs_Qm = vsh_coeffs_YZ_to_Qpm(-1, c, -c)
    S += vsh_series(1, -1, coeffs_Qp, *angular_coords)
    S += vsh_series(-1, -1, coeffs_Qm, *angular_coords)
    coeffs_Qp, coeffs_Qm = vsh_coeffs_YZ_to_Qpm(0, c.conj(), -c.conj())
    S += vsh_series(1, 0, coeffs_Qp, *angular_coords)
    S += vsh_series(-1, 0, coeffs_Qm, *angular_coords)
    coeffs_Qp, coeffs_Qm = vsh_coeffs_YZ_to_Qpm(1, -c, c.conj())
    S += vsh_series(1, 1, coeffs_Qp, *angular_coords)
    S += vsh_series(-1, 1, coeffs_Qm, *angular_coords)

    expected = np.array(
        [
            0.6278855765269712 - 1.5803682798806001 * 1j,
            1.2466319310518704 + 3.1301950505777832 * 1j,
            -2.1620753596643727 - 2.7723285425418966 * 1j,
        ]
    )
    assert np.allclose(S, expected, atol=10 * EPS, rtol=10 * EPS)
