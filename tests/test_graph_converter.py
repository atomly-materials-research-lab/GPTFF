import numpy as np
from pymatgen.core import Lattice, Structure

from gptff.graph import CrystalGraphBatch, CrystalGraphConverter, GraphSample, batch_samples


def test_periodic_self_images_are_kept():
    structure = Structure(Lattice.cubic(2.0), ["Na"], [[0.0, 0.0, 0.0]])
    graph = CrystalGraphConverter(r_cut=2.1, a_cut=2.1).convert(structure)

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
    graph = CrystalGraphConverter(r_cut=3.0, a_cut=3.0).convert(structure)

    vectors = (
        graph.positions[graph.edge_index[1]]
        + graph.edge_offsets @ graph.lattice
        - graph.positions[graph.edge_index[0]]
    )
    assert np.allclose(np.linalg.norm(vectors, axis=1), graph.edge_distances)


def test_triplets_are_ordered_edge_pairs_with_same_center():
    structure = Structure(Lattice.cubic(2.0), ["Na"], [[0.0, 0.0, 0.0]])
    graph = CrystalGraphConverter(r_cut=2.1, a_cut=2.1).convert(structure)

    assert graph.num_triplets == 30
    assert graph.triplets_per_atom.tolist() == [30]
    assert graph.triplets_per_edge.tolist() == [5, 5, 5, 5, 5, 5]
    assert np.all(graph.triplet_edge_index[0] != graph.triplet_edge_index[1])
    assert np.all(
        graph.edge_index[0][graph.triplet_edge_index[0]]
        == graph.edge_index[0][graph.triplet_edge_index[1]]
    )


def test_batch_offsets_atom_and_triplet_indices():
    structure = Structure(Lattice.cubic(2.0), ["Na"], [[0.0, 0.0, 0.0]])
    converter = CrystalGraphConverter(r_cut=2.1, a_cut=2.1)
    graph_a = converter.convert(structure)
    graph_b = converter.convert(structure)

    batch = CrystalGraphBatch.from_graphs([graph_a, graph_b])

    assert batch.num_atoms.tolist() == [1, 1]
    assert batch.num_edges.tolist() == [6, 6]
    assert batch.edge_index[:, :6].max().item() == 0
    assert batch.edge_index[:, 6:].min().item() == 1
    assert batch.triplet_edge_index[:, :30].max().item() < 6
    assert batch.triplet_edge_index[:, 30:].min().item() >= 6


def test_batch_samples_accepts_missing_ref_energy():
    structure = Structure(Lattice.cubic(2.0), ["Na"], [[0.0, 0.0, 0.0]])
    graph = CrystalGraphConverter(r_cut=2.1, a_cut=2.1).convert(structure)
    sample = GraphSample(
        graph=graph,
        energy=-1.0,
        forces=np.zeros((1, 3), dtype=np.float32),
        stress=np.zeros((3, 3), dtype=np.float32),
    )

    batch = batch_samples([sample])

    assert batch.energy.tolist() == [-1.0]
    assert batch.ref_energy is None
