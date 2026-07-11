from __future__ import annotations

import numpy as np

EV_PER_ANG3_TO_GPA = 160.21766208
KBAR_TO_GPA = 0.1
VASP_COMPRESSIVE_TO_TENSILE_SIGN = -1.0


def convert_vasp_stress_to_gpa(stress, *, dtype=np.float32) -> np.ndarray:
    """Convert VASP's compressive-positive kBar stress to tensile-positive GPa."""

    return (
        np.asarray(stress, dtype=dtype)
        * VASP_COMPRESSIVE_TO_TENSILE_SIGN
        * KBAR_TO_GPA
    )


def stress_matrix_to_ase_voigt(stress_matrix, *, dtype=np.float64) -> np.ndarray:
    """Flatten a stress matrix in ASE Voigt order without changing its units."""

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
