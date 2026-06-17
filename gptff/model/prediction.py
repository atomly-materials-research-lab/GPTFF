from __future__ import annotations

import torch


def predict_energy_forces_stress(
    model,
    batch,
    unit_trans=160.21766208,
    create_graph=True,
    *,
    compute_forces=True,
    compute_stress=True,
):
    graph = batch.with_geometry(
        positions_requires_grad=compute_forces,
        strain_requires_grad=compute_stress,
    )
    energy = model(graph).squeeze(-1)

    if not compute_forces and not compute_stress:
        return energy, None, None

    grad_inputs = []
    if compute_forces:
        grad_inputs.append(graph.positions)
    if compute_stress:
        grad_inputs.append(graph.strain)

    gradients = torch.autograd.grad(
        energy,
        grad_inputs,
        torch.ones_like(energy),
        retain_graph=True,
        create_graph=create_graph,
    )

    grad_iter = iter(gradients)
    forces = -next(grad_iter) if compute_forces else None
    if compute_stress:
        stress_grad = next(grad_iter)
        stress = stress_grad / graph.volumes[:, None, None] * unit_trans
    else:
        stress = None
    return energy, forces, stress
