import numpy as np
import torch
from pymatgen.core import Lattice, Structure

from gptff.graph import CrystalGraph, CrystalGraphBatch, CrystalGraphConverter
from gptff.model import GPTFF, GPTFFConfig


def test_atomwise_readout_is_extensive_for_disconnected_copies():
    cfg = GPTFFConfig(
        node_feature_len=8,
        edge_feature_len=8,
        n_layers=1,
        num_radial=8,
        num_angular=4,
        radial_cutoff=2.1,
        angle_cutoff=2.1,
        cutoff_coeff=5,
    )
    structure = Structure(Lattice.cubic(2.0), ["Na"], [[0.0, 0.0, 0.0]])
    graph = CrystalGraphConverter(r_cut=2.1, a_cut=2.1).convert(structure)
    doubled_graph = _make_disconnected_double_graph(graph)
    model = GPTFF(cfg)

    single_energy = model(CrystalGraphBatch.from_graphs([graph]).with_geometry())
    doubled_energy = model(CrystalGraphBatch.from_graphs([doubled_graph]).with_geometry())

    assert torch.allclose(doubled_energy, 2 * single_energy, atol=1e-5, rtol=1e-5)


def _make_disconnected_double_graph(graph: CrystalGraph) -> CrystalGraph:
    atom_offset = graph.num_atoms
    edge_offset = graph.num_edges
    displacement = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)

    return CrystalGraph(
        atom_types=np.concatenate([graph.atom_types, graph.atom_types], axis=0),
        positions=np.concatenate([graph.positions, graph.positions + displacement], axis=0),
        lattice=graph.lattice.copy(),
        radial_cutoff=graph.radial_cutoff,
        angle_cutoff=graph.angle_cutoff,
        edge_index=np.concatenate([graph.edge_index, graph.edge_index + atom_offset], axis=1),
        edge_offsets=np.concatenate([graph.edge_offsets, graph.edge_offsets], axis=0),
        edge_distances=np.concatenate([graph.edge_distances, graph.edge_distances], axis=0),
        triplet_edge_index=np.concatenate(
            [graph.triplet_edge_index, graph.triplet_edge_index + edge_offset],
            axis=1,
        ),
        triplets_per_atom=np.concatenate([graph.triplets_per_atom, graph.triplets_per_atom], axis=0),
        triplets_per_edge=np.concatenate([graph.triplets_per_edge, graph.triplets_per_edge], axis=0),
    )
