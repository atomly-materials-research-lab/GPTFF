from dataclasses import replace

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure

from gptff.graph import CrystalGraphBatch, CrystalGraphConverter
from gptff.inference import predict_energy_forces_stress
from gptff.model import GPTFF, GPTFFConfig
from gptff.utils.labels import EV_PER_ANG3_TO_GPA

RADIAL_CUTOFF = 3.0
ANGLE_CUTOFF = 2.5


@pytest.mark.parametrize("atom_attention", [False, True], ids=["message-passing", "attention"])
def test_gptff_rigid_translation_and_rotation_consistency(atom_attention):
    model = _build_model(atom_attention=atom_attention)
    structure = _reference_structure()
    energy, forces, stress = _predict(model, structure)

    translated = Structure(
        structure.lattice,
        structure.species,
        structure.cart_coords + np.array([0.37, -0.29, 0.41]),
        coords_are_cartesian=True,
    )
    translated_energy, translated_forces, translated_stress = _predict(model, translated)

    rotation = _rotation_matrix(axis=np.array([1.0, -2.0, 0.5]), angle=0.61)
    rotated = Structure(
        Lattice(structure.lattice.matrix @ rotation.T),
        structure.species,
        structure.cart_coords @ rotation.T,
        coords_are_cartesian=True,
    )
    rotated_energy, rotated_forces, rotated_stress = _predict(model, rotated)
    rotation_tensor = torch.tensor(rotation, dtype=forces.dtype)

    assert torch.allclose(translated_energy, energy, rtol=1e-6, atol=1e-7)
    assert torch.allclose(translated_forces, forces, rtol=1e-5, atol=1e-6)
    assert torch.allclose(translated_stress, stress, rtol=1e-5, atol=1e-6)
    assert torch.allclose(rotated_energy, energy, rtol=1e-6, atol=1e-7)
    assert torch.allclose(
        rotated_forces,
        forces @ rotation_tensor.T,
        rtol=1e-5,
        atol=1e-6,
    )
    assert torch.allclose(
        rotated_stress,
        rotation_tensor @ stress @ rotation_tensor.T,
        rtol=1e-5,
        atol=1e-6,
    )


def test_gptff_atom_permutation_invariance():
    model = _build_model()
    structure = _reference_structure()
    permutation = np.array([2, 0, 3, 1])

    permuted = Structure(
        structure.lattice,
        [structure[index].specie for index in permutation],
        structure.cart_coords[permutation],
        coords_are_cartesian=True,
    )

    energy, forces, stress = _predict(model, structure)
    permuted_energy, permuted_forces, permuted_stress = _predict(model, permuted)

    assert torch.allclose(permuted_energy, energy, rtol=1e-6, atol=1e-7)
    assert torch.allclose(
        permuted_forces,
        forces[torch.tensor(permutation)],
        rtol=1e-5,
        atol=1e-6,
    )
    assert torch.allclose(permuted_stress, stress, rtol=1e-5, atol=1e-6)


def test_gptff_periodic_site_representation_invariance():
    model = _build_model()
    structure = _reference_structure()
    shifted_fractional_coords = structure.frac_coords.copy()
    shifted_fractional_coords[1] += np.array([1.0, -1.0, 0.0])
    shifted = Structure(
        structure.lattice,
        structure.species,
        shifted_fractional_coords,
    )

    energy, forces, stress = _predict(model, structure)
    shifted_energy, shifted_forces, shifted_stress = _predict(model, shifted)

    assert torch.allclose(shifted_energy, energy, rtol=1e-6, atol=1e-7)
    assert torch.allclose(shifted_forces, forces, rtol=1e-5, atol=1e-6)
    assert torch.allclose(shifted_stress, stress, rtol=1e-5, atol=1e-6)


def test_gptff_net_force_is_zero_per_structure_in_batch():
    model = _build_model()
    structure = _reference_structure()
    translated = Structure(
        structure.lattice,
        structure.species,
        structure.cart_coords + np.array([-0.2, 0.3, 0.1]),
        coords_are_cartesian=True,
    )
    batch = _batch([structure, translated])

    _, forces, _ = predict_energy_forces_stress(
        model,
        batch,
        create_graph=False,
        compute_stress=False,
    )

    atom_offset = 0
    for atom_count in batch.num_atoms.tolist():
        structure_forces = forces[atom_offset : atom_offset + atom_count]
        assert torch.allclose(
            structure_forces.sum(dim=0),
            torch.zeros(3, dtype=forces.dtype),
            atol=1e-9,
        )
        atom_offset += atom_count


def test_full_gptff_force_matches_position_finite_difference():
    model = _build_model()
    batch = _batch([_reference_structure()])
    _, forces, _ = predict_energy_forces_stress(
        model,
        batch,
        create_graph=False,
        compute_stress=False,
    )

    atom_index = 1
    coordinate = 0
    step = 1e-5
    positive = _displace_batch_atom(batch, atom_index, coordinate, step)
    negative = _displace_batch_atom(batch, atom_index, coordinate, -step)
    finite_difference = -(_energy(model, positive) - _energy(model, negative)) / (2 * step)

    assert forces[atom_index, coordinate].item() == pytest.approx(
        finite_difference,
        rel=2e-4,
        abs=2e-6,
    )


@pytest.mark.parametrize("component", [(0, 0), (0, 1), (1, 2)])
def test_full_gptff_stress_matches_strain_finite_difference(component):
    model = _build_model()
    batch = _batch([_reference_structure()])
    _, _, stress = predict_energy_forces_stress(
        model,
        batch,
        create_graph=False,
        compute_stress=True,
    )

    row, column = component
    step = 1e-5
    positive = _strain_batch(batch, row, column, step)
    negative = _strain_batch(batch, row, column, -step)
    volume = torch.abs(torch.linalg.det(batch.lattice))[0].item()
    finite_difference = (
        (_energy(model, positive) - _energy(model, negative))
        / (2 * step)
        / volume
        * EV_PER_ANG3_TO_GPA
    )

    assert stress[0, row, column].item() == pytest.approx(
        finite_difference,
        rel=5e-4,
        abs=2e-5,
    )


def _build_model(*, atom_attention=False):
    torch.manual_seed(19)
    return (
        GPTFF(
            GPTFFConfig(
                atom_feature_dim=8,
                edge_feature_dim=8,
                num_interaction_blocks=1,
                num_radial=6,
                num_angular=5,
                radial_cutoff=RADIAL_CUTOFF,
                angle_cutoff=ANGLE_CUTOFF,
                num_readout_layers=2,
                atom_attention={
                    "enabled": atom_attention,
                    "num_heads": 2,
                    "dropout": 0.0,
                    "use_ffn": True,
                },
            )
        )
        .double()
        .eval()
    )


def _reference_structure():
    lattice = Lattice(
        [
            [7.0, 0.2, 0.1],
            [0.1, 7.5, 0.3],
            [0.2, 0.4, 8.0],
        ]
    )
    return Structure(
        lattice,
        ["Si", "O", "Na", "Cl"],
        [
            [0.18, 0.20, 0.22],
            [0.31, 0.22, 0.24],
            [0.20, 0.35, 0.27],
            [0.34, 0.36, 0.31],
        ],
    )


def _rotation_matrix(*, axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross_product = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )
    identity = np.eye(3)
    return (
        np.cos(angle) * identity
        + np.sin(angle) * cross_product
        + (1 - np.cos(angle)) * np.outer(axis, axis)
    )


def _predict(model, structure):
    return predict_energy_forces_stress(
        model,
        _batch([structure]),
        create_graph=False,
        compute_stress=True,
    )


def _batch(structures):
    converter = CrystalGraphConverter(
        radial_cutoff=RADIAL_CUTOFF,
        angle_cutoff=ANGLE_CUTOFF,
    )
    graphs = [converter.convert(structure) for structure in structures]
    batch = CrystalGraphBatch.from_graphs(graphs)
    float_fields = {
        name: value.double()
        for name, value in batch.__dict__.items()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    }
    return replace(batch, **float_fields)


def _displace_batch_atom(batch, atom_index, coordinate, displacement):
    positions = batch.positions.clone()
    positions[atom_index, coordinate] += displacement
    return replace(batch, positions=positions)


def _strain_batch(batch, row, column, strain):
    transform = torch.eye(3, dtype=batch.positions.dtype)
    transform[row, column] += strain
    return replace(
        batch,
        positions=batch.positions @ transform,
        lattice=batch.lattice @ transform,
    )


def _energy(model, batch):
    graph = batch.with_geometry(
        positions_requires_grad=False,
        strain_requires_grad=False,
    )
    return model(graph).sum().item()
