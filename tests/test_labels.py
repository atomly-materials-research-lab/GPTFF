import numpy as np

from gptff.utils.labels import (
    convert_vasp_stress_to_gpa,
    stress_matrix_to_ase_voigt,
)


def test_convert_vasp_stress_to_gpa_uses_vasp_sign_and_kbar_scale():
    stress = np.asarray([[10.0, 2.0, 0.0], [2.0, -5.0, 1.0], [0.0, 1.0, 3.0]])

    converted = convert_vasp_stress_to_gpa(stress)

    assert np.allclose(converted, -0.1 * stress)


def test_stress_matrix_to_ase_voigt_order():
    stress = np.asarray([[1.0, 6.0, 5.0], [6.0, 2.0, 4.0], [5.0, 4.0, 3.0]])

    converted = stress_matrix_to_ase_voigt(stress)

    assert np.allclose(converted, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
