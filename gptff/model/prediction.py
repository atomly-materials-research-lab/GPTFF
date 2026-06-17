from __future__ import annotations

import torch


def predict_energy_forces_stress(model, batch, unit_trans=160.21766208, create_graph=True):
    graph = batch.with_geometry()
    energy = model(graph).squeeze(-1)

    forces, stress_grad = torch.autograd.grad(
        energy,
        [graph.positions, graph.strain],
        torch.ones_like(energy),
        retain_graph=True,
        create_graph=create_graph,
    )
    forces = -forces
    stress = stress_grad / graph.volumes[:, None, None] * unit_trans
    return energy, forces, stress
