from __future__ import annotations

import numpy as np

EV_PER_ANG3_TO_GPA = 160.21766208
VASP_KBAR_TO_GPA = -0.1


def convert_vasp_stress_to_gpa(stress, *, dtype=np.float32) -> np.ndarray:
    return np.asarray(stress, dtype=dtype) * VASP_KBAR_TO_GPA


def convert_stress_gpa_to_ev_per_ang3(stress, *, dtype=np.float64) -> np.ndarray:
    return np.asarray(stress, dtype=dtype) / EV_PER_ANG3_TO_GPA


def stress_matrix_to_ase_voigt(stress_matrix, *, dtype=np.float64) -> np.ndarray:
    stress_matrix = np.asarray(stress_matrix, dtype=dtype)
    if stress_matrix.shape != (3, 3):
        raise ValueError(f"stress_matrix must have shape (3, 3), got {stress_matrix.shape}.")
    return np.asarray(
        [
            stress_matrix[0, 0],
            stress_matrix[1, 1],
            stress_matrix[2, 2],
            stress_matrix[1, 2],
            stress_matrix[0, 2],
            stress_matrix[0, 1],
        ],
        dtype=dtype,
    )


def stress_gpa_to_ase_voigt(stress_matrix, *, dtype=np.float64) -> np.ndarray:
    return stress_matrix_to_ase_voigt(
        convert_stress_gpa_to_ev_per_ang3(stress_matrix, dtype=dtype),
        dtype=dtype,
    )
