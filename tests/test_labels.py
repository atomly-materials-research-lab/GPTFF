import numpy as np

from gptff.utils.labels import (
    EV_PER_ANG3_TO_GPA,
    convert_stress_gpa_to_ev_per_ang3,
    convert_vasp_stress_to_gpa,
    stress_gpa_to_ase_voigt,
)


def test_convert_vasp_stress_to_gpa_uses_vasp_sign_and_kbar_scale():
    stress = np.asarray([[10.0, 2.0, 0.0], [2.0, -5.0, 1.0], [0.0, 1.0, 3.0]])

    converted = convert_vasp_stress_to_gpa(stress)

    assert np.allclose(converted, -0.1 * stress)


def test_convert_stress_gpa_to_ev_per_ang3():
    stress = np.eye(3) * EV_PER_ANG3_TO_GPA

    assert np.allclose(convert_stress_gpa_to_ev_per_ang3(stress), np.eye(3))


def test_stress_gpa_to_ase_voigt_order():
    stress = np.asarray([[1.0, 6.0, 5.0], [6.0, 2.0, 4.0], [5.0, 4.0, 3.0]])

    converted = stress_gpa_to_ase_voigt(stress * EV_PER_ANG3_TO_GPA)

    assert np.allclose(converted, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
