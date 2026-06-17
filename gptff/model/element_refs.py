from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


_ATOMLY_ELEMENT_REFS = (
    0.00000000e00,
    -3.46535853e00,
    -7.56101906e-01,
    -3.46224791e00,
    -4.77600176e00,
    -8.03619240e00,
    -8.40374071e00,
    -7.76814618e00,
    -7.38918302e00,
    -4.94725878e00,
    -2.92883670e-02,
    -2.47830716e00,
    -2.02015956e00,
    -5.15479820e00,
    -7.91209653e00,
    -6.91345095e00,
    -4.62278149e00,
    -3.01552069e00,
    -6.27971322e-02,
    -2.31732442e00,
    -4.75968073e00,
    -8.17421803e00,
    -1.14207788e01,
    -8.92294483e00,
    -8.48981509e00,
    -8.16635547e00,
    -6.58248850e00,
    -5.26139665e00,
    -4.48412068e00,
    -3.27367370e00,
    -1.34976438e00,
    -3.62637456e00,
    -4.67270042e00,
    -4.13166577e00,
    -3.67546394e00,
    -2.80302539e00,
    6.47272418e00,
    -2.24681188e00,
    -4.25110577e00,
    -1.02452951e01,
    -1.16658385e01,
    -1.18015760e01,
    -8.65537518e00,
    -9.36409198e00,
    -7.57165084e00,
    -5.69907599e00,
    -4.97159232e00,
    -1.88700594e00,
    -6.79483530e-01,
    -2.74880153e00,
    -3.79441765e00,
    -3.38825264e00,
    -2.55867271e00,
    -1.96213610e00,
    9.97909972e00,
    -2.55677995e00,
    -4.88030347e00,
    -8.86033743e00,
    -9.05368602e00,
    -7.94309693e00,
    -8.12585485e00,
    -6.31826210e00,
    -8.30242223e00,
    -1.22893251e01,
    -1.73097460e01,
    -7.55105974e00,
    -8.19580521e00,
    -8.34926874e00,
    -7.25911206e00,
    -8.41697224e00,
    -3.38725429e00,
    -7.68222088e00,
    -1.26297007e01,
    -1.36257602e01,
    -9.52985029e00,
    -1.18396814e01,
    -9.79914325e00,
    -7.55608603e00,
    -5.46902454e00,
    -2.65092136e00,
    4.17472161e-01,
    -2.32548971e00,
    -3.48299933e00,
    -3.18067109e00,
    3.57605604e-15,
    9.96350211e-16,
    1.18278079e-15,
    -1.44201673e-15,
    -6.73760309e-18,
    -5.48347781e00,
    -1.03346396e01,
    -1.11296117e01,
    -1.43116273e01,
    -1.47003999e01,
    -1.54726487e01,
)


ELEMENT_REF_PRESETS = {
    "atomly": _ATOMLY_ELEMENT_REFS,
}


def available_element_ref_presets() -> tuple[str, ...]:
    return tuple(sorted(ELEMENT_REF_PRESETS))


def build_element_ref_tensor(
    element_refs: str | Mapping[int | str, float] | Sequence[float] | None,
    max_atomic_number: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor | None:
    if max_atomic_number < 1:
        raise ValueError("max_atomic_number must be positive.")

    max_atomic_number = int(max_atomic_number)
    if element_refs is None:
        return None

    if isinstance(element_refs, str):
        return _build_from_preset(element_refs, max_atomic_number, dtype=dtype)

    if isinstance(element_refs, Mapping):
        return _build_from_mapping(element_refs, max_atomic_number, dtype=dtype)

    return _build_from_sequence(element_refs, max_atomic_number, dtype=dtype)


def _build_from_preset(
    preset_name: str,
    max_atomic_number: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = preset_name.lower()
    if key not in ELEMENT_REF_PRESETS:
        available = ", ".join(available_element_ref_presets())
        raise ValueError(
            f"Unknown element_refs preset '{preset_name}'. "
            f"Available presets: {available}."
        )

    refs = torch.as_tensor(ELEMENT_REF_PRESETS[key], dtype=dtype)
    expected = max_atomic_number + 1
    if refs.numel() < expected:
        raise ValueError(
            f"Element reference preset '{preset_name}' only covers atomic numbers "
            f"up to {refs.numel() - 1}, but max_atomic_number is {max_atomic_number}."
        )
    return refs[:expected].clone()


def _build_from_mapping(
    element_refs: Mapping[int | str, float],
    max_atomic_number: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    refs = torch.zeros(max_atomic_number + 1, dtype=dtype)
    for atomic_number, value in element_refs.items():
        atomic_number = int(atomic_number)
        if atomic_number < 1 or atomic_number > max_atomic_number:
            raise ValueError(
                "element_refs keys must be in the range "
                f"[1, {max_atomic_number}], got {atomic_number}."
            )
        refs[atomic_number] = float(value)
    return refs


def _build_from_sequence(
    element_refs: Sequence[float],
    max_atomic_number: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    refs = torch.zeros(max_atomic_number + 1, dtype=dtype)
    refs_tensor = torch.as_tensor(element_refs, dtype=dtype)
    if refs_tensor.ndim != 1:
        raise ValueError("element_refs must be a 1D sequence, mapping, preset name, or None.")
    if refs_tensor.numel() == max_atomic_number:
        refs[1:] = refs_tensor
        return refs
    if refs_tensor.numel() == max_atomic_number + 1:
        return refs_tensor.clone()
    raise ValueError(
        "element_refs must have length max_atomic_number or "
        "max_atomic_number + 1."
    )
