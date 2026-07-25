from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np


INORGANIC_ATOM_REFS = np.array([
    0.00000000e00, -3.46535853e00, -7.56101906e-01, -3.46224791e00,
    -4.77600176e00, -8.03619240e00, -8.40374071e00, -7.76814618e00,
    -7.38918302e00, -4.94725878e00, -2.92883670e-02, -2.47830716e00,
    -2.02015956e00, -5.15479820e00, -7.91209653e00, -6.91345095e00,
    -4.62278149e00, -3.01552069e00, -6.27971322e-02, -2.31732442e00,
    -4.75968073e00, -8.17421803e00, -1.14207788e01, -8.92294483e00,
    -8.48981509e00, -8.16635547e00, -6.58248850e00, -5.26139665e00,
    -4.48412068e00, -3.27367370e00, -1.34976438e00, -3.62637456e00,
    -4.67270042e00, -4.13166577e00, -3.67546394e00, -2.80302539e00,
    6.47272418e00, -2.24681188e00, -4.25110577e00, -1.02452951e01,
    -1.16658385e01, -1.18015760e01, -8.65537518e00, -9.36409198e00,
    -7.57165084e00, -5.69907599e00, -4.97159232e00, -1.88700594e00,
    -6.79483530e-01, -2.74880153e00, -3.79441765e00, -3.38825264e00,
    -2.55867271e00, -1.96213610e00, 9.97909972e00, -2.55677995e00,
    -4.88030347e00, -8.86033743e00, -9.05368602e00, -7.94309693e00,
    -8.12585485e00, -6.31826210e00, -8.30242223e00, -1.22893251e01,
    -1.73097460e01, -7.55105974e00, -8.19580521e00, -8.34926874e00,
    -7.25911206e00, -8.41697224e00, -3.38725429e00, -7.68222088e00,
    -1.26297007e01, -1.36257602e01, -9.52985029e00, -1.18396814e01,
    -9.79914325e00, -7.55608603e00, -5.46902454e00, -2.65092136e00,
    4.17472161e-01, -2.32548971e00, -3.48299933e00, -3.18067109e00,
    3.57605604e-15, 9.96350211e-16, 1.18278079e-15, -1.44201673e-15,
    -6.73760309e-18, -5.48347781e00, -1.03346396e01, -1.11296117e01,
    -1.43116273e01, -1.47003999e01, -1.54726487e01,
], dtype=np.float64)

MOLECULAR_ATOM_REFS = np.array([
    0.00000000e00, -1.67038438e01, -5.68434189e-14, 2.04636308e-12,
    2.27373675e-13, 1.81898940e-12, -1.03529378e03, -1.48797876e03,
    -2.04465848e03, 4.59177481e-41, 4.59177481e-41, 1.01957882e-56,
    1.01957882e-56, -2.26391977e-72, -5.65979942e-73, 0.00000000e00,
    -1.08313801e04, -1.25197421e04, 0.00000000e00, 0.00000000e00,
], dtype=np.float64)

REFERENCE_ENERGIES = {
    "inorganic": INORGANIC_ATOM_REFS,
    "molecular": MOLECULAR_ATOM_REFS,
}

ReferenceEnergies: TypeAlias = str | Sequence[float] | np.ndarray


def resolve_reference_energies(
    reference_energies: ReferenceEnergies = "inorganic",
) -> np.ndarray:
    if isinstance(reference_energies, str):
        profile = reference_energies.lower()
        try:
            references = REFERENCE_ENERGIES[profile]
        except KeyError as exc:
            available = ", ".join(sorted(REFERENCE_ENERGIES))
            raise ValueError(
                f"Unknown reference energy profile {reference_energies!r}. "
                f"Available profiles: {available}."
            ) from exc
    else:
        references = np.asarray(reference_energies, dtype=np.float64)

    references = np.asarray(references, dtype=np.float64)
    if references.ndim != 1 or references.size == 0:
        raise ValueError("reference_energies must be a non-empty one-dimensional array.")
    if not np.isfinite(references).all():
        raise ValueError("reference_energies must contain only finite values.")
    return references.copy()


__all__ = [
    "INORGANIC_ATOM_REFS",
    "MOLECULAR_ATOM_REFS",
    "REFERENCE_ENERGIES",
    "ReferenceEnergies",
    "resolve_reference_energies",
]
