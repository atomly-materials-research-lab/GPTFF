from __future__ import annotations

import torch


def predict_energy_forces_stress(
    model,
    batch,
    create_graph=False,
    *,
    compute_stress=True,
):
    """Return energy, forces, and stress in eV, eV/Angstrom, and eV/Angstrom^3."""

    graph = batch.with_geometry(
        positions_requires_grad=True,
        strain_requires_grad=compute_stress,
    )
    energy = model(graph).squeeze(-1)

    zero_energy = energy.sum() * 0.0 if energy.requires_grad and create_graph else None

    grad_inputs = [graph.positions]
    if compute_stress:
        grad_inputs.append(graph.strain)

    if energy.requires_grad:
        gradients = torch.autograd.grad(
            energy,
            grad_inputs,
            torch.ones_like(energy),
            retain_graph=create_graph,
            create_graph=create_graph,
            allow_unused=True,
        )
    else:
        gradients = tuple(None for _ in grad_inputs)

    grad_iter = iter(gradients)
    forces = -_optional_gradient(next(grad_iter), graph.positions, zero_energy)
    if compute_stress:
        stress_grad = _optional_gradient(next(grad_iter), graph.strain, zero_energy)
        volumes = graph.volumes if create_graph else graph.volumes.detach()
        stress = stress_grad / volumes[:, None, None]
    else:
        stress = None
    return energy, forces, stress


def _optional_gradient(gradient, reference, zero_energy):
    if gradient is not None:
        return gradient
    zeros = torch.zeros_like(reference)
    if zero_energy is None:
        return zeros
    return zeros + zero_energy
