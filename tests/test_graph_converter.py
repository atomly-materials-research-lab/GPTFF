from dataclasses import replace

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure

from gptff.graph import CrystalGraphBatch, CrystalGraphConverter, GraphSample, batch_samples


def test_periodic_self_images_are_kept():
    structure = Structure(Lattice.cubic(2.0), ["Na"], [[0.0, 0.0, 0.0]])
    graph = CrystalGraphConverter(radial_cutoff=2.1, angle_cutoff=2.1).convert(structure)

    assert graph.num_edges == 6
    assert np.all(graph.edge_index[0] == 0)
    assert np.all(graph.edge_index[1] == 0)
    assert not np.any(np.all(graph.edge_offsets == 0, axis=1))
    assert {tuple(offset.astype(int)) for offset in graph.edge_offsets} == {
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
        (0, 0, 1),
        (0, 1, 0),
        (1, 0, 0),
    }


def test_offset_convention_matches_edge_distance():
    structure = Structure(
        Lattice.cubic(3.0),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    graph = CrystalGraphConverter(radial_cutoff=3.0, angle_cutoff=3.0).convert(structure)

    vectors = (
        graph.positions[graph.edge_index[1]]
        + graph.edge_offsets @ graph.lattice
        - graph.positions[graph.edge_index[0]]
    )
    assert np.allclose(np.linalg.norm(vectors, axis=1), graph.edge_distances)


def test_triplets_are_ordered_edge_pairs_with_same_center():
    structure = Structure(Lattice.cubic(2.0), ["Na"], [[0.0, 0.0, 0.0]])
    graph = CrystalGraphConverter(radial_cutoff=2.1, angle_cutoff=2.1).convert(structure)

    assert graph.num_triplets == 30
    assert graph.triplets_per_atom.tolist() == [30]
    assert graph.triplets_per_edge.tolist() == [5, 5, 5, 5, 5, 5]
    assert np.all(graph.triplet_edge_index[0] != graph.triplet_edge_index[1])
    assert np.all(
        graph.edge_index[0][graph.triplet_edge_index[0]]
        == graph.edge_index[0][graph.triplet_edge_index[1]]
    )


def test_collinear_triplet_cosine_reaches_physical_boundary():
    structure = Structure(
        Lattice.cubic(10.0),
        ["Si", "O", "O"],
        [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        coords_are_cartesian=True,
    )
    graph = CrystalGraphConverter(radial_cutoff=1.5, angle_cutoff=1.5).convert(structure)

    differentiable_graph = CrystalGraphBatch.from_graphs([graph]).with_geometry(
        positions_requires_grad=False,
        strain_requires_grad=False,
    )

    assert torch.count_nonzero(differentiable_graph.triplet_cosine == -1.0).item() == 2


def test_batch_offsets_atom_and_triplet_indices():
    structure = Structure(Lattice.cubic(2.0), ["Na"], [[0.0, 0.0, 0.0]])
    converter = CrystalGraphConverter(radial_cutoff=2.1, angle_cutoff=2.1)
    graph_a = converter.convert(structure)
    graph_b = converter.convert(structure)

    batch = CrystalGraphBatch.from_graphs([graph_a, graph_b])

    assert batch.num_atoms.tolist() == [1, 1]
    assert batch.num_edges.tolist() == [6, 6]
    assert batch.edge_index[:, :6].max().item() == 0
    assert batch.edge_index[:, 6:].min().item() == 1
    assert batch.triplet_edge_index[:, :30].max().item() < 6
    assert batch.triplet_edge_index[:, 30:].min().item() >= 6


def test_converter_and_batch_keep_cutoff_metadata():
    structure = Structure(Lattice.cubic(2.0), ["Na"], [[0.0, 0.0, 0.0]])
    graph = CrystalGraphConverter(radial_cutoff=2.1, angle_cutoff=1.9).convert(structure)
    batch = CrystalGraphBatch.from_graphs([graph])

    assert graph.radial_cutoff == pytest.approx(2.1)
    assert graph.angle_cutoff == pytest.approx(1.9)
    assert batch.radial_cutoff == pytest.approx(2.1)
    assert batch.angle_cutoff == pytest.approx(1.9)


def test_batch_pin_memory_is_noop_without_cuda(monkeypatch):
    structure = Structure(Lattice.cubic(2.0), ["Na"], [[0.0, 0.0, 0.0]])
    graph = CrystalGraphConverter(radial_cutoff=2.1, angle_cutoff=2.1).convert(structure)
    batch = CrystalGraphBatch.from_graphs([graph])

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    pinned = batch.pin_memory()

    assert pinned.atom_types is batch.atom_types
    assert pinned.positions is batch.positions


def test_batch_rejects_mismatched_cutoff_metadata():
    structure = Structure(Lattice.cubic(2.0), ["Na"], [[0.0, 0.0, 0.0]])
    graph = CrystalGraphConverter(radial_cutoff=2.1, angle_cutoff=2.1).convert(structure)

    with pytest.raises(ValueError, match="different radial_cutoff"):
        CrystalGraphBatch.from_graphs([graph, replace(graph, radial_cutoff=2.2)])

    with pytest.raises(ValueError, match="different angle_cutoff"):
        CrystalGraphBatch.from_graphs([graph, replace(graph, angle_cutoff=2.2)])


def test_batch_samples_collates_labels():
    structure = Structure(Lattice.cubic(2.0), ["Na"], [[0.0, 0.0, 0.0]])
    graph = CrystalGraphConverter(radial_cutoff=2.1, angle_cutoff=2.1).convert(structure)
    sample = GraphSample(
        graph=graph,
        energy=-1.0,
        forces=np.zeros((1, 3), dtype=np.float32),
        stress=np.zeros((3, 3), dtype=np.float32),
    )

    batch = batch_samples([sample])

    assert batch.energy.tolist() == [-1.0]
    assert batch.forces.shape == (1, 3)
    assert batch.stress.shape == (1, 3, 3)


def test_batch_samples_allows_missing_optional_labels():
    structure = Structure(Lattice.cubic(2.0), ["Na"], [[0.0, 0.0, 0.0]])
    graph = CrystalGraphConverter(radial_cutoff=2.1, angle_cutoff=2.1).convert(structure)
    sample = GraphSample(graph=graph, energy=-1.0)

    batch = batch_samples([sample])

    assert batch.energy.tolist() == [-1.0]
    assert batch.forces is None
    assert batch.stress is None


def test_batch_samples_rejects_partially_missing_labels():
    structure = Structure(Lattice.cubic(2.0), ["Na"], [[0.0, 0.0, 0.0]])
    graph = CrystalGraphConverter(radial_cutoff=2.1, angle_cutoff=2.1).convert(structure)
    samples = [
        GraphSample(graph=graph, energy=-1.0, stress=np.zeros((3, 3), dtype=np.float32)),
        GraphSample(graph=graph, energy=-1.0),
    ]

    with pytest.raises(ValueError, match="partially missing stress"):
        batch_samples(samples)
