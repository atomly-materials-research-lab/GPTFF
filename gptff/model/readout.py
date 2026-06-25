from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class EnergyReadoutOutput:
    energy: torch.Tensor
    site_energy: torch.Tensor
    residual_site_energy: torch.Tensor
    reference_site_energy: torch.Tensor


def build_element_ref_tensor(
    element_refs: Mapping[int | str, float] | Sequence[float] | None,
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
        raise TypeError(
            "element_refs no longer accepts named presets. Provide a mapping, "
            "sequence, None, or configure element_references.source as a YAML/JSON file."
        )

    if isinstance(element_refs, Mapping):
        return _build_from_mapping(element_refs, max_atomic_number, dtype=dtype)

    return _build_from_sequence(element_refs, max_atomic_number, dtype=dtype)


def fit_element_refs_from_samples(
    samples,
    max_atomic_number: int,
    *,
    ridge: float = 0.0,
) -> dict[str, float]:
    if max_atomic_number < 1:
        raise ValueError("max_atomic_number must be positive.")
    if ridge < 0:
        raise ValueError("ridge must be non-negative.")

    compositions = []
    energies = []
    observed = torch.zeros(max_atomic_number, dtype=torch.bool)

    for sample in _iter_samples(samples):
        atom_types = torch.as_tensor(sample.graph.atom_types, dtype=torch.long)
        if atom_types.numel() == 0:
            raise ValueError("Cannot fit element_refs from an empty structure.")
        if torch.any((atom_types < 1) | (atom_types > max_atomic_number)):
            raise ValueError(
                "Atomic numbers must be in the range "
                f"[1, {max_atomic_number}] when fitting element_refs."
            )

        composition = torch.bincount(
            atom_types,
            minlength=max_atomic_number + 1,
        )[1:].to(torch.float64)
        compositions.append(composition)
        energies.append(float(sample.energy))
        observed |= composition > 0

    if not compositions:
        raise ValueError("Cannot fit element_refs from an empty sample collection.")

    composition_matrix = torch.stack(compositions, dim=0)
    target_energy = torch.tensor(energies, dtype=torch.float64)
    active_matrix = composition_matrix[:, observed]
    if active_matrix.shape[1] == 0:
        raise ValueError("No elements were observed while fitting element_refs.")

    if ridge == 0:
        solution = torch.linalg.lstsq(
            active_matrix,
            target_energy.unsqueeze(-1),
        ).solution.squeeze(-1)
    else:
        gram = active_matrix.T @ active_matrix
        regularizer = ridge * torch.eye(gram.shape[0], dtype=gram.dtype)
        solution = torch.linalg.solve(
            gram + regularizer,
            active_matrix.T @ target_energy,
        )

    refs = torch.zeros(max_atomic_number, dtype=torch.float64)
    refs[observed] = solution
    return {
        str(atomic_number): float(refs[atomic_number - 1].item())
        for atomic_number in torch.nonzero(observed, as_tuple=False).flatten().add(1).tolist()
    }


def _iter_samples(samples):
    if hasattr(samples, "__len__") and hasattr(samples, "__getitem__"):
        for idx in range(len(samples)):
            yield samples[idx]
    else:
        yield from samples


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
        raise ValueError("element_refs must be a 1D sequence, mapping, or None.")
    if refs_tensor.numel() == max_atomic_number:
        refs[1:] = refs_tensor
        return refs
    if refs_tensor.numel() == max_atomic_number + 1:
        return refs_tensor.clone()
    raise ValueError("element_refs must have length max_atomic_number or max_atomic_number + 1.")


class EnergyHead(nn.Module):
    def __init__(
        self,
        atom_feature_dim,
        max_atomic_number=94,
        element_refs=None,
        num_readout_layers=3,
    ):
        super().__init__()
        self.max_atomic_number = int(max_atomic_number)
        self.num_readout_layers = int(num_readout_layers)
        if self.num_readout_layers <= 0:
            raise ValueError("num_readout_layers must be positive.")

        hidden_layers = []
        for _ in range(self.num_readout_layers - 1):
            hidden_layers.extend(
                [
                    nn.Linear(atom_feature_dim, atom_feature_dim),
                    nn.SiLU(),
                ]
            )
        self.hidden_mlp = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(atom_feature_dim, 1)
        self.register_buffer(
            "element_refs",
            build_element_ref_tensor(
                element_refs,
                max_atomic_number=self.max_atomic_number,
            ),
        )

    def residual_site_energy(self, atom_features):
        return self.output_layer(self.hidden_mlp(atom_features))

    def reference_site_energy(self, atom_types, reference):
        if self.element_refs is not None:
            return (
                self.element_refs[atom_types]
                .unsqueeze(-1)
                .to(
                    dtype=reference.dtype,
                    device=reference.device,
                )
            )
        return torch.zeros_like(reference)

    def forward_with_site_energies(self, atom_features, atom_types, atom_batch, num_graphs):
        residual_site_energy = self.residual_site_energy(atom_features)
        reference_site_energy = self.reference_site_energy(atom_types, residual_site_energy)
        site_energy = residual_site_energy + reference_site_energy

        energy = torch.zeros(
            (num_graphs, 1),
            dtype=site_energy.dtype,
            device=site_energy.device,
        )
        energy = torch.index_add(energy, 0, atom_batch, site_energy)
        return EnergyReadoutOutput(
            energy=energy,
            site_energy=site_energy,
            residual_site_energy=residual_site_energy,
            reference_site_energy=reference_site_energy,
        )

    def forward(self, atom_features, atom_types, atom_batch, num_graphs):
        return self.forward_with_site_energies(
            atom_features,
            atom_types,
            atom_batch,
            num_graphs,
        ).energy
