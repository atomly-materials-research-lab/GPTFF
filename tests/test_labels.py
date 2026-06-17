import numpy as np
import pytest

from gptff.utils_.labels import (
    EV_PER_ANG3_TO_GPA,
    LabelConfig,
    convert_forces_to_ev_per_ang,
    convert_stress_to_gpa,
    normalize_stress_unit,
)


def test_convert_kbar_stress_to_gpa_with_sign():
    stress = np.eye(3, dtype=np.float32) * 10.0

    converted = convert_stress_to_gpa(stress, unit="kbar", sign=-1.0)

    assert np.allclose(converted, -np.eye(3, dtype=np.float32))


def test_convert_ev_per_ang3_stress_to_gpa():
    stress = np.eye(3, dtype=np.float32)

    converted = convert_stress_to_gpa(stress, unit="ev_per_ang3")

    assert np.allclose(converted, np.eye(3, dtype=np.float32) * EV_PER_ANG3_TO_GPA)


def test_label_config_validates_units():
    assert LabelConfig(stress_unit="ev/A^3").stress_sign == -1.0
    assert normalize_stress_unit("eV/angstrom^3") == "ev_per_ang3"
    assert convert_forces_to_ev_per_ang([[1, 2, 3]], unit="ev/A").shape == (1, 3)

    with pytest.raises(ValueError, match="stress_unit"):
        LabelConfig(stress_unit="bar")
